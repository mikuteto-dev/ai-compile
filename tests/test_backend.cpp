#include "aic/atlas.hpp"
#include "aic/cir.hpp"
#include "aic/codegen.hpp"
#include "aic/errors.hpp"
#include "aic/json_min.hpp"
#include "aic/linker.hpp"
#include "aic/mir.hpp"
#include "aic/patch_bundle.hpp"
#include "aic/runtime.hpp"
#include "aic/safetensors.hpp"
#include "aic/tensor.hpp"
#include "aic/tokenizer.hpp"
#include "aic/wisa.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace aic;

namespace {

constexpr double kPi = 3.14159265358979323846;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error("FAILED: " + msg);
    }
}

void test_atlas_save_load() {
    ModelConfig cfg;
    cfg.num_layers = 8;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 96;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.attention_type = "gqa";
    TransformerModel model = TransformerModel::create_synthetic(cfg, 42);

    const std::string src = R"STRATA(
module AtlasTest;
dipole LoopPole in Frame<Mid> {
    +pole: LoopOn;
    -pole: LoopOff;
}
manifold Ring : S1 in Frame<Mid> {
    curvature 1.0;
}
field Tone : LowCurvature in Frame<All> {
    rank 4;
    metric Fisher;
}
flow DoWork(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-3, approx <= 1e-2, energy <= 0.5, preserve(CoreFluency, Residual)]
{
    state +> LoopPole.-pole(gain: 0.25);
}
)STRATA";

    MIRModule mir = compile_source_to_mir(src);
    NeuralAtlas atlas = build_default_atlas(model, &mir, 42);
    atlas.occupied_kv_groups.insert({2, 0});

    const std::string path = (std::filesystem::temp_directory_path() / "aic_test_atlas.json").string();
    atlas.save(path);
    NeuralAtlas loaded = NeuralAtlas::load(path);
    std::filesystem::remove(path);

    check(loaded.arch_name == atlas.arch_name, "atlas arch_name mismatch");
    check(loaded.hidden_size == 32, "atlas hidden_size mismatch");
    check(loaded.num_layers == 8, "atlas num_layers mismatch");
    check(loaded.symbols.contains("LoopPole"), "atlas missing LoopPole");
    check(loaded.symbols.contains("Ring"), "atlas missing Ring");
    check(loaded.symbols["Ring"].manifold_plane.size() == 3, "Ring manifold_plane size");
    check(loaded.symbols.contains("Tone"), "atlas missing Tone");
    check(loaded.symbols["Tone"].field_modes.size() == 4, "Tone field_modes rank");
    check(loaded.dormant_poles.size() == atlas.dormant_poles.size(), "dormant_poles round-trip");
    check(loaded.occupied_kv_groups.contains({2, 0}), "occupied_kv_groups round-trip");
    check(!loaded.frame_bridges.empty(), "frame_bridges populated");
    std::cout << "[PASS] test_atlas_save_load\n";
}

void test_s1_tessellation_and_decay() {
    const std::size_t d = 24;
    Vector e0(d, 0.0);
    Vector e1(d, 0.0);
    Vector e2(d, 0.0);
    e0[0] = 1.0;
    e1[1] = 1.0;
    e2[2] = 1.0;

    const double delta = 2.0 * kPi / 7.0;
    double prev_err = 1e9;
    for (int k : {4, 8, 16, 32}) {
        auto [slots, err] = build_s1_swiglu_spline(e0, e1, e2, k, delta);
        check(slots.size() == static_cast<std::size_t>(3 * k), "expected 3*K slots");
        for (const auto& s : slots) {
            const double up_norm = norm2(s.w_up);
            check(up_norm > 0.0, "w_up nonzero");
            const double cos_align = dot(s.w_gate, s.w_up) / (norm2(s.w_gate) * up_norm);
            check(std::abs(cos_align - 1.0) < 1e-12, "w_gate parallel to w_up");
        }
        check(err < prev_err, "S1 spline error must decay monotonically");
        prev_err = err;
    }
    check(prev_err < 1e-4, "S1 K=32 error must be < 1e-4");

    ModelConfig cfg;
    cfg.num_layers = 4;
    cfg.hidden_size = 24;
    cfg.intermediate_size = 64;
    TransformerModel model = TransformerModel::create_synthetic(cfg, 7);

    auto run_s1_bound = [&](const std::string& bound_str) -> std::size_t {
        const std::string s1_src =
            "module EscalateS1;\n"
            "manifold Ring : S1 in Frame<Mid> { curvature 1.0; }\n"
            "flow Rot(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>\n"
            "    ![crosstalk <= 1e-3, approx <= " + bound_str + ", energy <= 0.5, preserve(CoreFluency)]\n"
            "{\n"
            "    when state ~> Ring(chart: \"angle\") as th {\n"
            "        let nxt = state => Ring.rotate(angle: th);\n"
            "        state +> nxt(gain: 1.0);\n"
            "    }\n"
            "}\n";
        MIRModule m = compile_source_to_mir(s1_src);
        NeuralAtlas a = build_default_atlas(model, &m, 7);
        CIRModule c = lower_mir_to_cir(m, cfg, a, "elastic");
        return c.flows[0].ops[0].spline_slots.size();
    };
    check(run_s1_bound("1e-2") == 12, "S1 1e-2 selects K=4 (12 slots)");
    check(run_s1_bound("1e-3") == 24, "S1 1e-3 escalates to K=8 (24 slots)");
    check(run_s1_bound("2e-4") == 48, "S1 2e-4 escalates to K=16 (48 slots)");
    check(run_s1_bound("2e-5") == 96, "S1 2e-5 escalates to K=32 (96 slots)");

    const std::string tight_src = R"STRATA(
module TightS1;
manifold Ring : S1 in Frame<Mid> {
    curvature 1.0;
}
flow Rot(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-3, approx <= 1e-12, energy <= 0.5, preserve(CoreFluency)]
{
    when state ~> Ring(chart: "angle") as th {
        let nxt = state => Ring.rotate(angle: th);
        state +> nxt(gain: 1.0);
    }
}
)STRATA";
    MIRModule tight_mir = compile_source_to_mir(tight_src);
    NeuralAtlas tight_atlas = build_default_atlas(model, &tight_mir, 7);
    bool threw_unmet = false;
    try {
        (void)lower_mir_to_cir(tight_mir, cfg, tight_atlas, "in-place");
    } catch (const ApproximationUnmetError&) {
        threw_unmet = true;
    }
    check(threw_unmet, "expected ApproximationUnmetError for approx <= 1e-12");

    std::cout << "[PASS] test_s1_tessellation_and_decay\n";
}

void test_nullspace_and_errors() {
    ModelConfig cfg;
    cfg.num_layers = 4;
    cfg.hidden_size = 16;
    cfg.intermediate_size = 48;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    TransformerModel model = TransformerModel::create_synthetic(cfg, 11);

    const std::string src = R"STRATA(
module NullTest;
dipole Sig in Frame<Mid> {
    +pole: A;
    -pole: B;
}
flow Edit(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-8, approx <= 0.0, energy <= 0.5, preserve(CoreFluency, Residual)]
{
    when state ~> Sig.+pole(threshold: 0.5) {
        state -| Sig.+pole(scale: 1.0);
        state +> Sig.-pole(gain: 0.3);
    }
}
)STRATA";

    MIRModule mir = compile_source_to_mir(src);
    NeuralAtlas atlas = build_default_atlas(model, &mir, 11);
    CIRModule cir = lower_mir_to_cir(mir, cfg, atlas, "in-place");
    check(cir.flows.size() == 1, "expected 1 CIR flow");
    check(cir.flows[0].max_approx_error == 0.0, "flat dipole must have 0 approx error");
    check(cir.flows[0].max_crosstalk < 1e-10, "Nullspace projection must keep crosstalk < 1e-10");

    for (const auto& op : cir.flows[0].ops) {
        for (const auto& [pname, pvecs] : atlas.preserve_directions) {
            if (pname != "CoreFluency" && pname != "Residual") {
                continue;
            }
            for (const Vector& pv : pvecs) {
                if (!op.output_dir.empty()) {
                    check(std::abs(dot(op.output_dir, pv)) < 1e-10, "output_dir orthogonal to preserved");
                }
                if (!op.input_dir.empty()) {
                    check(std::abs(dot(op.input_dir, pv)) < 1e-10, "input_dir orthogonal to preserved");
                }
            }
        }
    }

    NeuralAtlas full_atlas = atlas;
    std::vector<Vector> basis;
    for (int i = 0; i < cfg.hidden_size; ++i) {
        Vector e(static_cast<std::size_t>(cfg.hidden_size), 0.0);
        e[static_cast<std::size_t>(i)] = 1.0;
        basis.push_back(std::move(e));
    }
    full_atlas.preserve_directions["CoreFluency"] = basis;
    bool threw_null = false;
    try {
        (void)lower_mir_to_cir(mir, cfg, full_atlas, "in-place");
    } catch (const NullspaceExhaustedError&) {
        threw_null = true;
    }
    check(threw_null, "expected NullspaceExhaustedError when preserve spans R^d");

    MIRModule raw_mir;
    raw_mir.name = "RawXtTest";
    MIRSymbolDecl f1_decl;
    f1_decl.name = "F1";
    f1_decl.kind = "field";
    f1_decl.rank = 2;
    MIRSymbolDecl f2_decl;
    f2_decl.name = "F2";
    f2_decl.kind = "field";
    f2_decl.rank = 2;
    raw_mir.declarations["F1"] = f1_decl;
    raw_mir.declarations["F2"] = f2_decl;
    MIRFlow rf;
    rf.name = "DirectField";
    rf.contract.crosstalk_bound = 1e-6;
    rf.contract.preserve = {"F2"};
    MIROp fd_op;
    fd_op.op = "field_decouple";
    fd_op.dst = "pure";
    fd_op.target = "F1";
    fd_op.rhs_target = "F2";
    fd_op.scalar_arg = 1e-4;
    rf.ops.push_back(fd_op);
    raw_mir.flows.push_back(rf);
    NeuralAtlas xt_atlas = atlas;
    Vector u_s = normalize(Vector{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    Vector v_s = normalize(Vector{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    xt_atlas.symbols["F1"].kind = "field";
    xt_atlas.symbols["F1"].field_modes = {SVDMode{1.0, u_s, v_s}};
    xt_atlas.symbols["F2"].kind = "field";
    xt_atlas.symbols["F2"].field_modes = {SVDMode{1.0, u_s, v_s}};
    CIRModule raw_cir = lower_mir_to_cir(raw_mir, cfg, xt_atlas, "in-place");
    check(raw_cir.flows[0].max_crosstalk < 1e-10, "decouple annihilated collinear field cleanly");

    MIRModule bad_xt_mir = raw_mir;
    bad_xt_mir.flows[0].contract.crosstalk_bound = 1e-15;
    bad_xt_mir.flows[0].ops.clear();
    MIROp ex_op;
    ex_op.op = "excite";
    ex_op.dst = "state";
    ex_op.target = "Sig";
    ex_op.pole = "+pole";
    ex_op.scalar_arg = 0.5;
    bad_xt_mir.flows[0].ops.push_back(ex_op);
    bad_xt_mir.declarations["Sig"] = mir.declarations.at("Sig");
    bad_xt_mir.flows[0].stage = MIRStageSpec{"Mid", "", false};
    NeuralAtlas bad_xt_atlas = atlas;
    bad_xt_atlas.symbols["Sig"].pole_slots.clear();
    bad_xt_atlas.symbols["Sig"].direction = normalize(vec_add(
        atlas.symbols.at("Sig").direction,
        atlas.preserve_directions.at("CoreFluency").at(0)));
    bad_xt_mir.flows[0].contract.preserve = {"CoreFluency"};
    bool threw_xt = false;
    try {
        (void)lower_mir_to_cir(bad_xt_mir, cfg, bad_xt_atlas, "in-place");
    } catch (const CrosstalkExceededError&) {
        threw_xt = true;
    }
    check(threw_xt, "expected CrosstalkExceededError when residual crosstalk exceeds bound");

    std::cout << "[PASS] test_nullspace_and_errors\n";
}

void test_fisher_decouple_and_field() {
    const std::size_t d = 16;
    Vector u1 = normalize(Vector{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    Vector v1 = normalize(Vector{1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    Vector u2 = normalize(Vector{1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    Vector v2 = normalize(Vector{1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    Vector u3 = normalize(Vector{0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    Vector v3 = normalize(Vector{0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0});

    std::vector<SVDMode> f1{SVDMode{2.0, u1, v1}, SVDMode{1.0, u3, v3}};
    std::vector<SVDMode> f2{SVDMode{1.5, u2, v2}};
    Vector g_out(d, 1.0);
    Vector g_in(d, 1.0);
    for (std::size_t i = 0; i < d; ++i) {
        g_out[i] = 1.0 + 0.15 * static_cast<double>(i);
        g_in[i] = 2.0 + 0.05 * static_cast<double>(i);
    }

    const double pre_cos = fisher_cosine(f1, f2, g_out, g_in);
    check(pre_cos > 0.05, "expected non-trivial Fisher overlap before decouple");

    const std::vector<SVDMode> f_pure = fisher_decouple(f1, f2, g_out, g_in, 1e-4);
    check(!f_pure.empty(), "expected non-empty pure field");
    const double post_cos = fisher_cosine(f_pure, f2, g_out, g_in);
    check(post_cos < 1e-10, "expected Fisher orthogonality < 1e-10 after decouple");

    const std::vector<SVDMode> f_temp = temper_field_modes(f1, "low_curvature", 1.2);
    check(f_temp.size() == f1.size() && std::abs(f_temp[0].sigma - 2.4) < 1e-12, "temper check");
    const std::vector<SVDMode> f_blend = blend_field_modes(f1, f2, 0.5);
    check(!f_blend.empty(), "blend check");

    std::cout << "[PASS] test_fisher_decouple_and_field\n";
}

void test_attention_modes_and_conflicts() {
    const std::string attn_src = R"STRATA(
module AttnTest;
pole QueryTag in Frame<Mid>;
subspace DeltaAxis<1> in Frame<Mid>;
flow AttendFlow(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-3, approx <= 1e-2, energy <= 0.5, preserve(CoreFluency, Residual)]
{
    let d_out = @attend(query: state, key: QueryTag, value: DeltaAxis, mask: "causal");
    state +> DeltaAxis(gain: 0.25);
}
)STRATA";
    MIRModule mir = compile_source_to_mir(attn_src);

    for (const std::string& mode_name : {std::string("mha"), std::string("gqa"), std::string("mla")}) {
        ModelConfig cfg;
        cfg.num_layers = 4;
        cfg.hidden_size = 24;
        cfg.intermediate_size = 48;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = (mode_name == "mha") ? 4 : 2;
        cfg.attention_type = mode_name;
        cfg.kv_lora_rank = 6;
        TransformerModel base = TransformerModel::create_synthetic(cfg, 9);
        NeuralAtlas atlas = build_default_atlas(base, &mir, 9);

        CompileOutput out = compile_mir_to_model(mir, base, &atlas, "in-place");
        check(out.stats.instructions_applied > 0, "expected instructions applied for " + mode_name);

        if (mode_name == "gqa") {
            for (int l = 0; l < cfg.num_layers; ++l) {
                const std::string k_key = "model.layers." + std::to_string(l) + ".self_attn.k_proj.weight";
                const std::string v_key = "model.layers." + std::to_string(l) + ".self_attn.v_proj.weight";
                check(frob_norm(mat_sub(out.model.matrices.at(k_key), base.matrices.at(k_key))) == 0.0,
                      "GQA must preserve k_proj");
                check(frob_norm(mat_sub(out.model.matrices.at(v_key), base.matrices.at(v_key))) == 0.0,
                      "GQA must preserve v_proj");
            }

            NeuralAtlas blocked = atlas;
            for (int l = 0; l < cfg.num_layers; ++l) {
                for (int g = 0; g < cfg.num_key_value_heads; ++g) {
                    blocked.occupied_kv_groups.insert({l, g});
                }
            }
            bool threw_gqa = false;
            try {
                (void)compile_mir_to_model(mir, base, &blocked, "in-place");
            } catch (const GQAHeadSharingConflictError&) {
                threw_gqa = true;
            }
            check(threw_gqa, "expected GQAHeadSharingConflictError in in-place mode");

            CompileOutput elastic_gqa = compile_mir_to_model(mir, base, &blocked, "elastic");
            check(elastic_gqa.stats.added_layers >= 1, "elastic GQA resolves conflict with bypass layer");
            auto [logits, trace] = elastic_gqa.model.forward({1, 2, 3}, {}, false);
            (void)trace;
            check(logits.size() == 3, "elastic GQA forward ok");
        }

        auto [logits, trace] = out.model.forward({1, 2, 3}, {}, false);
        (void)trace;
        check(logits.size() == 3, "forward ok for " + mode_name);
    }

    std::cout << "[PASS] test_attention_modes_and_conflicts\n";
}

void test_depth_and_slot_exhaustion_and_elastic() {
    ModelConfig cfg;
    cfg.num_layers = 4;
    cfg.hidden_size = 24;
    cfg.intermediate_size = 48;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    TransformerModel base = TransformerModel::create_synthetic(cfg, 19);

    const std::string deep_src = R"STRATA(
module DepthTest;
pole P1 in Frame<0..1>;
pole P2 in Frame<0..1>;
flow SeqFlow(state: Residual in Frame<0..1>) -> Residual in Frame<0..1>
    ![crosstalk <= 1e-3, approx <= 0.0, energy <= 0.5, preserve(CoreFluency)]
{
    when state ~> P1(threshold: 0.5) {
        state +> P1(gain: 0.2);
    }
    when state ~> P2(threshold: 0.5) {
        state +> P2(gain: 0.2);
    }
}
)STRATA";
    MIRModule deep_mir = compile_source_to_mir(deep_src);
    NeuralAtlas deep_atlas = build_default_atlas(base, &deep_mir, 19);

    bool threw_depth = false;
    try {
        (void)compile_mir_to_model(deep_mir, base, &deep_atlas, "in-place");
    } catch (const OutOfDepthError&) {
        threw_depth = true;
    }
    check(threw_depth, "expected OutOfDepthError when seq_depth > stage layers");

    CompileOutput elastic_depth = compile_mir_to_model(deep_mir, base, &deep_atlas, "elastic");
    check(elastic_depth.stats.added_layers == 1, "expected 1 bypass layer added");
    check(elastic_depth.model.config.num_layers == 5, "num_layers grew to 5");
    check(elastic_depth.stats.added_poles == 2, "expected 2 poles added");
    check(elastic_depth.model.config.intermediate_size == cfg.intermediate_size + 1, "intermediate_size grew cleanly");
    auto [logits, tr] = elastic_depth.model.forward({1, 2, 3}, {}, false);
    (void)tr;
    check(logits.size() == 3, "elastic bypassed model runs forward cleanly");

    WISAInstruction noop_bypass;
    noop_bypass.kind = WISAKind::NORM_RESCALE;
    noop_bypass.layer = 4;
    noop_bypass.norm_id = "post_attention_layernorm";
    noop_bypass.gamma_scale.assign(static_cast<std::size_t>(cfg.hidden_size), 0.0);
    auto [bypass_only, bypass_stats] = apply_wisa(base, {noop_bypass}, "elastic");
    check(bypass_stats.added_layers == 1, "bypass layer inserted cleanly");
    auto [base_logits, bt] = base.forward({1, 2, 3}, {}, false);
    auto [byp_logits, byt] = bypass_only.forward({1, 2, 3}, {}, false);
    (void)bt;
    (void)byt;
    for (std::size_t i = 0; i < base_logits.size(); ++i) {
        check(norm2(vec_sub(base_logits[i], byp_logits[i])) < 1e-12, "zero-init bypass layer preserves exact identity");
    }

    const std::string s1_src = R"STRATA(
module SlotTest;
manifold Weekday : S1 in Frame<Mid> {
    curvature 1.0;
}
flow Shift(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-3, approx <= 1e-2, energy <= 0.5, preserve(CoreFluency, Residual)]
{
    when state ~> Weekday(chart: "angle") as th {
        let nxt = state => Weekday.rotate(angle: th);
        state +> nxt(gain: 1.0);
    }
}
)STRATA";
    MIRModule s1_mir = compile_source_to_mir(s1_src);
    NeuralAtlas scarce_atlas = build_default_atlas(base, &s1_mir, 19);
    scarce_atlas.dormant_poles.resize(2);

    bool threw_slots = false;
    try {
        (void)compile_mir_to_model(s1_mir, base, &scarce_atlas, "in-place");
    } catch (const ApproximationUnmetError&) {
        threw_slots = true;
    }
    check(threw_slots, "expected ApproximationUnmetError when dormant_poles insufficient");

    CompileOutput elastic_s1 = compile_mir_to_model(s1_mir, base, &scarce_atlas, "elastic");
    check(elastic_s1.stats.added_poles >= 12, "elastic S1 added >= 12 spline poles");
    check(elastic_s1.model.config.intermediate_size == cfg.intermediate_size + elastic_s1.stats.added_poles,
          "intermediate_size matches added poles");

    std::cout << "[PASS] test_depth_and_slot_exhaustion_and_elastic\n";
}

void test_assoc_mem_row_and_all_four_cases() {
    ModelConfig qcfg;
    qcfg.num_layers = 8;
    qcfg.hidden_size = 32;
    qcfg.intermediate_size = 96;
    qcfg.num_attention_heads = 4;
    qcfg.num_key_value_heads = 2;
    qcfg.attention_type = "gqa";
    qcfg.arch_name = "qwen3-mini";
    TransformerModel qwen = TransformerModel::create_synthetic(qcfg, 42);

    Vector k_star = vec_zeros(static_cast<std::size_t>(qcfg.intermediate_size));
    k_star[0] = 1.0;
    k_star[3] = 0.5;
    Vector v_star(static_cast<std::size_t>(qcfg.hidden_size), 0.0);
    v_star[0] = 0.8;
    v_star[1] = -0.6;
    WISAInstruction mem_inst;
    mem_inst.kind = WISAKind::ASSOC_MEM_ROW;
    mem_inst.layer = 2;
    mem_inst.k_star = k_star;
    mem_inst.v_star = v_star;
    mem_inst.c_inv = eye(static_cast<std::size_t>(qcfg.intermediate_size));
    mem_inst.target_norm = 1.0;
    auto [mem_model, mem_stats] = apply_wisa(qwen, {mem_inst}, "in-place");
    check(mem_stats.instructions_applied == 1, "assoc_mem_row applied");
    const Vector out_v = mat_vec(mem_model.matrices.at("model.layers.2.mlp.down_proj.weight"), k_star);
    check(norm2(vec_sub(out_v, v_star)) < 1e-12, "ASSOC_MEM_ROW exact closed-form target check");

    const std::string case1 = R"STRATA(
module MathLoopSurgery;
import std::CoreReasoning;
import std::Fluency;
frame MathReasonMid in 16..24 {
    connect to LogitFrame via StandardResidualBridge;
}
@stage(16..24)
dipole MathLoopAnomaly in Frame<16..24> {
    +pole: RepetitionSpiral {
        anchor contrastive(
            pos: ["Wait, let me double check. Wait, let me double check."],
            neg: ["Therefore, x = 5. Moving to the next equation:"]
        );
    }
    -pole: StepProgression {
        anchor probe("Step 2: Substituting into equation (1) yields:");
    }
}
@target("in-place")
flow SuppressRunawayLoop(state: Residual in Frame<16..24>) -> Residual in Frame<16..24>
    ![crosstalk <= 1e-3, approx <= 0.0, energy <= 0.05, preserve(CoreReasoning, Fluency, Residual)]
{
    when state ~> MathLoopAnomaly.+pole(threshold: 0.72) as amp {
        state -| MathLoopAnomaly.+pole(scale: 1.0);
        state +> MathLoopAnomaly.-pole(gain: 0.45 * amp);
    }
}
)STRATA";
    MIRModule m1 = compile_source_to_mir(case1);
    ModelConfig deep_cfg = qcfg;
    deep_cfg.num_layers = 24;
    TransformerModel deep_model = TransformerModel::create_synthetic(deep_cfg, 101);
    CompileOutput o1 = compile_mir_to_model(m1, deep_model, nullptr, "in-place");
    check(o1.model.config.intermediate_size == deep_cfg.intermediate_size, "Case 1 in-place keeps shape");
    check(o1.stats.max_approx_error == 0.0, "Case 1 flat dipole approx 0");
    check(o1.stats.max_crosstalk <= 1e-3, "Case 1 crosstalk within contract");
    check(o1.cir.flows[0].ops.size() == 2, "Case 1 lowers both gated body edits");
    const CIROp& gated_excite = o1.cir.flows[0].ops[1];
    check(gated_excite.kind == "thresholded_route", "dynamic gain lowers to a thresholded route");
    check(gated_excite.symbol == "MathLoopAnomaly" && gated_excite.pole == "+pole",
          "dynamic gain uses the when detector pole");
    check(std::abs(gated_excite.scalar + 0.45) < 1e-12 &&
              std::abs(gated_excite.threshold - 0.72) < 1e-12,
          "dynamic gain preserves threshold, gain, and target pole polarity");
    check(std::any_of(o1.wisa.begin(), o1.wisa.end(), [](const WISAInstruction& wi) {
              return wi.kind == WISAKind::THRESHOLDED_ROUTE;
          }),
          "dynamic gain emits THRESHOLDED_ROUTE W-ISA");
    check(o1.model.config.aic_threshold_routes,
          "compiled checkpoint enables AIC threshold routes");
    bool lora_route_rejected = false;
    try {
        (void)compile_mir_to_model(m1, deep_model, nullptr, "lora");
    } catch (const ApproximationUnmetError&) {
        lora_route_rejected = true;
    }
    check(lora_route_rejected, "LoRA target rejects threshold route tensors");

    ModelConfig route_cfg;
    route_cfg.num_layers = 1;
    route_cfg.hidden_size = 4;
    route_cfg.intermediate_size = 2;
    route_cfg.num_attention_heads = 1;
    route_cfg.num_key_value_heads = 1;
    route_cfg.vocab_size = 4;
    route_cfg.attention_type = "mha";
    TransformerModel route_base = TransformerModel::create_synthetic(route_cfg, 2026);
    const std::string route_layer = "model.layers.0";
    for (const char* suffix : {
             ".self_attn.q_proj.weight",
             ".self_attn.k_proj.weight",
             ".self_attn.v_proj.weight",
             ".self_attn.o_proj.weight",
             ".mlp.gate_proj.weight",
             ".mlp.up_proj.weight",
             ".mlp.down_proj.weight"}) {
        Matrix& m = route_base.matrices.at(route_layer + suffix);
        m = zeros(m.rows, m.cols);
    }
    route_base.matrices.at(route_layer + ".mlp.gate_proj.weight")(0, 0) = 1.0;
    route_base.matrices.at(route_layer + ".mlp.up_proj.weight")(0, 0) = 1.0;
    const std::vector<Vector> route_input = {{1.0, 0.0, 0.0, 0.0}};
    const auto [base_logits, base_trace] = route_base.forward({}, route_input, true);
    (void)base_logits;
    const double detector_act = base_trace.mlp_poles[0][0].act[0];
    check(detector_act > 1.0, "controlled detector activation is positive");

    WISAInstruction route_wi;
    route_wi.kind = WISAKind::THRESHOLDED_ROUTE;
    route_wi.layer = 0;
    route_wi.slot = 0;
    route_wi.polarity = 1;
    route_wi.u = {1.0, 0.0, 0.0, 0.0};
    route_wi.scale = 2.0;
    route_wi.threshold = detector_act + 0.25;
    auto [below_model, below_stats] = apply_wisa(route_base, {route_wi}, "in-place");
    const auto [below_logits, below_trace] = below_model.forward({}, route_input, true);
    (void)below_logits;
    check(below_stats.instructions_applied == 1 &&
              norm2(vec_sub(below_trace.residual[1][0], base_trace.residual[1][0])) < 1e-12,
          "thresholded route is exactly zero below threshold");

    route_wi.threshold = detector_act - 0.5;
    auto [above_model, above_stats] = apply_wisa(route_base, {route_wi}, "in-place");
    const auto [above_logits, above_trace] = above_model.forward({}, route_input, true);
    (void)above_logits;
    const Vector route_delta = vec_sub(above_trace.residual[1][0], base_trace.residual[1][0]);
    check(above_stats.instructions_applied == 1 &&
              std::abs(route_delta[0] - 2.0 * detector_act) < 1e-10 &&
              norm2(Vector(route_delta.begin() + 1, route_delta.end())) < 1e-12,
          "thresholded route preserves the detector amplitude above threshold");

    const PatchBundle route_bundle = create_patch_bundle(
        route_base, {route_wi}, "in-place", "threshold-route");
    const TransformerModel route_applied = apply_patch_bundle(route_base, route_bundle, true);
    check(route_applied.config.aic_threshold_routes &&
              route_applied.matrices.contains(route_layer + ".mlp.aic_threshold_route.weight"),
          "patch bundle enables and installs threshold route tensors");
    const TransformerModel route_reverted = revert_patch_bundle(route_applied, route_bundle);
    check(!route_reverted.config.aic_threshold_routes &&
              !route_reverted.matrices.contains(route_layer + ".mlp.aic_threshold_route.weight"),
          "patch bundle revert removes threshold route tensors and config flag");

    const std::string case2 = R"STRATA(
module CyclicWeekdayShift;
import std::Attention::TemporalIndex;
import std::CoreFluency;
@stage(8..14)
manifold Weekday : S1 in Frame<Mid> {
    curvature 1.0;
    chart angle: "radians_mod_2pi";
    anchor cyclic(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]);
}
subspace ShiftDelta<1> in Frame<Mid> {
    anchor probe("in {k} days");
}
pole WeekdayQueryTag in Frame<Mid> {
    anchor probe("What day of the week is it");
}
flow ComputeShiftedWeekday(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 5e-4, approx <= 1e-2, energy <= 0.12, preserve(CoreFluency, Residual)]
{
    let delta = @attend(
        query: state,
        key: WeekdayQueryTag,
        value: ShiftDelta,
        mask: "causal"
    );
    when state ~> Weekday(chart: "angle") as theta {
        let shifted = state => Weekday.rotate(angle: theta, delta: delta);
        state -| Weekday;
        state +> shifted(gain: 1.0);
    }
}
)STRATA";
    MIRModule m2 = compile_source_to_mir(case2);
    CompileOutput o2_ip = compile_mir_to_model(m2, qwen, nullptr, "in-place");
    check(o2_ip.model.config.intermediate_size == qcfg.intermediate_size, "Case 2 in-place keeps shape");
    check(o2_ip.stats.max_approx_error <= 1e-2, "Case 2 approx <= 1e-2");
    CompileOutput o2_el = compile_mir_to_model(m2, qwen, nullptr, "elastic");
    check(o2_el.model.config.intermediate_size > qcfg.intermediate_size, "Case 2 elastic widens MLP");
    check(!format_cir(o2_ip.cir).empty() && !format_wisa(o2_ip.wisa).empty(), "IR formatters non-empty");

    const std::string case3 = R"STRATA(
module TacitBehaviorSculpting;
import std::CoreFluency;
field EmpatheticTone : LowCurvature in Frame<All> {
    rank 16;
    metric Fisher;
    anchor spectral(
        prompts: ["I understand how difficult this must feel for you."],
        curvature_band: "low"
    );
}
field SycophancyDrift : Mixed in Frame<Mid..Late> {
    rank 12;
    metric Fisher;
    anchor contrastive(
        pos: ["You are completely right, 2 + 2 = 5 if you prefer!"],
        neg: ["I appreciate your perspective, however 2 + 2 equals 4."]
    );
}
field RigorousReasoning : LowCurvature in Frame<Mid..Late> {
    rank 16;
    metric Fisher;
    anchor spectral(prompts: ["Let us check each premise step by step."], curvature_band: "low");
}
flow SculptTacitPriors ![crosstalk <= 2e-3, approx <= 1e-3, energy <= 0.10, preserve(EmpatheticTone, CoreFluency, Residual)] {
    let pure_sycophancy = SycophancyDrift decouple EmpatheticTone with (metric: "fisher", reg: 1e-4);
    state -| pure_sycophancy(scale: 0.85);
    let calm_reason = temper(RigorousReasoning, band: "low_curvature", gain: 1.15);
    let blended = blend(EmpatheticTone, calm_reason, alpha: 0.25);
    state +> blended(gain: 0.20);
}
)STRATA";
    MIRModule m3 = compile_source_to_mir(case3);
    CompileOutput o3 = compile_mir_to_model(m3, qwen, nullptr, "in-place");
    check(o3.stats.max_crosstalk <= 2e-3, "Case 3 crosstalk <= 2e-3");
    bool has_field_op = false;
    for (const auto& wi : o3.wisa) {
        if (wi.kind == WISAKind::FIELD_SPECTRAL_OP) {
            has_field_op = true;
        }
    }
    check(has_field_op, "Case 3 emitted FIELD_SPECTRAL_OP");

    const std::string case4 = R"STRATA(
module FactSurgeryAndCrossTransplant;
import std::CoreReasoning;
import std::CoreFluency;
rodata CapitalKnowledge<Entity, City> in Frame<Mid..Late> {
    anchor factual("The official capital of {subject} is {object}.");
    entry "Aethelgard" => "Oldhaven" with (curvature: 24.8);
    entry "Valoria" => "Solis" with (curvature: 19.2);
}
subspace FactConsistencyCheck<4> in Frame<Mid..Late> {
    anchor contrastive(
        pos: ["Verify geographical entity before answering."],
        neg: ["Guess any plausible city name immediately."]
    );
}
flow UpdateAndTransplant(state: Residual in Frame<Mid..Late>) -> Residual in Frame<Mid..Late>
    ![crosstalk <= 1e-3, approx <= 5e-3, energy <= 0.08, preserve(CoreReasoning, CoreFluency, Residual)]
{
    rewrite CapitalKnowledge(key: "Aethelgard", old: "Oldhaven", new: "Newhaven");
    prune CapitalKnowledge(key: "DeprecatedCityRecord");
    insert CapitalKnowledge(key: "Veridia", value: "Celestia", sharpness: 15.0);
    let verified = transport(state, Frame<Mid..Late> -> Frame<Late>);
    when verified ~> FactConsistencyCheck(threshold: 0.65) {
        state +> FactConsistencyCheck(gain: 0.35);
    }
}
)STRATA";
    MIRModule m4 = compile_source_to_mir(case4);
    ModelConfig lcfg;
    lcfg.num_layers = 4;
    lcfg.hidden_size = 24;
    lcfg.intermediate_size = 64;
    lcfg.num_attention_heads = 4;
    lcfg.num_key_value_heads = 2;
    lcfg.attention_type = "gqa";
    lcfg.arch_name = "llama-mini";
    TransformerModel llama = TransformerModel::create_synthetic(lcfg, 77);
    CompileOutput o4_q = compile_mir_to_model(m4, qwen, nullptr, "in-place");
    CompileOutput o4_l = compile_mir_to_model(m4, llama, nullptr, "in-place");
    check(o4_q.stats.instructions_applied >= 5, "Case 4 qwen applied");
    check(o4_l.stats.instructions_applied >= 5, "Case 4 llama applied");
    auto [lq, tq] = o4_q.model.forward({1, 2}, {}, false);
    auto [ll, tl] = o4_l.model.forward({1, 2}, {}, false);
    (void)tq;
    (void)tl;
    check(lq.size() == 2 && ll.size() == 2, "Case 4 forward checks ok");

    std::cout << "[PASS] test_assoc_mem_row_and_all_four_cases\n";
}

std::string read_binary_bytes(const std::filesystem::path& p) {
    std::ifstream ifs(p, std::ios::binary);
    check(static_cast<bool>(ifs), "failed reading binary file: " + p.string());
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

void test_multithreaded_cir_and_large_scale() {
    const std::size_t d = 64;
    Vector e0(d, 0.0), e1(d, 0.0), e2(d, 0.0);
    e0[0] = 1.0;
    e1[1] = 1.0;
    e2[2] = 1.0;
    const double delta = 2.0 * kPi / 7.0;
    auto [slots_32, err_32] = build_s1_swiglu_spline(e0, e1, e2, 32, delta);
    check(slots_32.size() == 96, "MT S1 K=32 slots");
    check(err_32 < 1e-4, "MT S1 K=32 error < 1e-4");

    std::vector<SVDMode> f1;
    std::vector<SVDMode> f2;
    for (std::size_t r = 0; r < 8; ++r) {
        Vector u1(d, 0.0), v1(d, 0.0), u2(d, 0.0), v2(d, 0.0);
        u1[r] = 1.0;
        v1[(r + 1) % d] = 1.0;
        u2[(r + 4) % d] = 1.0;
        v2[(r + 5) % d] = 1.0;
        f1.push_back(SVDMode{2.0 / static_cast<double>(r + 1), u1, v1});
        f2.push_back(SVDMode{1.5 / static_cast<double>(r + 1), u2, v2});
    }
    Vector g_out(d, 1.0), g_in(d, 1.0);
    for (std::size_t i = 0; i < d; ++i) {
        g_out[i] = 1.0 + 0.02 * static_cast<double>(i);
        g_in[i] = 1.5 + 0.01 * static_cast<double>(i);
    }
    const std::vector<SVDMode> pure = fisher_decouple(f1, f2, g_out, g_in, 1e-4);
    check(!pure.empty(), "MT Fisher decouple non-empty");
    check(fisher_cosine(pure, f2, g_out, g_in) < 1e-10, "MT Fisher decouple < 1e-10");

    ModelConfig cfg;
    cfg.num_layers = 4;
    cfg.hidden_size = 64;
    cfg.intermediate_size = 128;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.attention_type = "gqa";
    TransformerModel model = TransformerModel::create_synthetic(cfg, 123);

    const std::string src = R"STRATA(
module MultiThreadCIR;
dipole D1 in Frame<Mid> {
    +pole: P1;
    -pole: P2;
}
rodata Tab<Entity, City> in Frame<Mid> {
    entry "K1" => "V1" with (curvature: 12.0);
}
flow MultiOp(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-4, approx <= 1e-2, energy <= 0.8, preserve(CoreFluency, Residual)]
{
    rewrite Tab(key: "K1", old: "V1", new: "V2");
    insert Tab(key: "K2", value: "V3", sharpness: 10.0);
    when state ~> D1.+pole(threshold: 0.6) {
        state -| D1.+pole(scale: 1.0);
        state +> D1.-pole(gain: 0.25);
    }
}
)STRATA";
    MIRModule mir = compile_source_to_mir(src);
    NeuralAtlas atlas = build_default_atlas(model, &mir, 123);
    CIRModule cir1 = lower_mir_to_cir(mir, cfg, atlas, "in-place");
    CIRModule cir2 = lower_mir_to_cir(mir, cfg, atlas, "in-place");
    check(cir1.flows[0].ops.size() == 4, "MT CIR op count");
    check(cir1.flows[0].max_crosstalk < 1e-10, "MT CIR crosstalk < 1e-10");
    check(format_cir(cir1) == format_cir(cir2), "MT CIR deterministic across runs");

    std::cout << "[PASS] test_multithreaded_cir_and_large_scale\n";
}

void test_streaming_bf16_sharded_checkpoint() {
    ModelConfig cfg;
    cfg.num_layers = 4;
    cfg.hidden_size = 128;
    cfg.intermediate_size = 384;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.vocab_size = 64;
    cfg.attention_type = "gqa";
    cfg.use_rope = true;
    cfg.qk_norm = true;
    cfg.torch_dtype = "BF16";
    cfg.arch_name = "qwen3-mini";

    TransformerModel base = TransformerModel::create_synthetic(cfg, 314);
    std::map<std::string, TensorSpec> specs;
    for (const auto& [name, m] : base.matrices) {
        specs[name] = TensorSpec::from_matrix(m, "BF16");
    }
    for (const auto& [name, v] : base.vectors) {
        specs[name] = TensorSpec::from_vector(v, "BF16");
    }

    const std::filesystem::path tmp_root =
        std::filesystem::temp_directory_path() / "aic_test_streaming_bf16";
    std::filesystem::remove_all(tmp_root);
    const std::filesystem::path src_dir = tmp_root / "src_ckpt";
    const std::filesystem::path dst_ip_dir = tmp_root / "dst_inplace";
    const std::filesystem::path dst_el_dir = tmp_root / "dst_elastic";
    std::filesystem::create_directories(src_dir);

    save_safetensors_sharded(
        src_dir.string(),
        specs,
        900u * 1024u,
        {{"arch_name", cfg.arch_name}, {"torch_dtype", "BF16"}});
    {
        std::ofstream cfg_ofs(src_dir / "config.json", std::ios::trunc);
        cfg_ofs << dump_json(cfg.to_json(), 2) << "\n";
    }

    const JsonValue idx_json = parse_json(read_binary_bytes(src_dir / "model.safetensors.index.json"));
    const JsonValue& wmap = idx_json["weight_map"];
    std::set<std::string> shard_files;
    for (const auto& [tname, sfile] : wmap.as_object()) {
        (void)tname;
        shard_files.insert(sfile.as_string());
    }
    check(shard_files.size() == 2, "expected exactly 2 shards in BF16 checkpoint");

    const std::string shard_l0 = wmap["model.layers.0.mlp.down_proj.weight"].as_string();
    const std::string shard_l3 = wmap["model.layers.3.mlp.down_proj.weight"].as_string();
    check(shard_l0 != shard_l3, "layer 0 and layer 3 must reside in distinct shards");

    TransformerModel bf16_mem = TransformerModel::load_checkpoint(src_dir.string());

    const std::size_t d = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t d_ff = static_cast<std::size_t>(cfg.intermediate_size);
    Vector u_d(d, 0.0), v_d(d, 0.0);
    u_d[0] = 0.6;
    u_d[3] = 0.8;
    v_d[1] = 1.0;

    std::vector<WISAInstruction> insts;
    {
        WISAInstruction wi;
        wi.kind = WISAKind::SWIGLU_POLE;
        wi.layer = 0;
        wi.slot = 5;
        wi.polarity = 1;
        wi.w_gate = vec_scale(u_d, 0.4);
        wi.w_up = v_d;
        wi.w_down = vec_scale(u_d, 0.25);
        insts.push_back(std::move(wi));
    }
    {
        WISAInstruction wi;
        wi.kind = WISAKind::THRESHOLDED_ROUTE;
        wi.layer = 0;
        wi.slot = 5;
        wi.polarity = 1;
        wi.u = vec_scale(u_d, 0.2);
        wi.scale = 0.4;
        wi.threshold = 0.3;
        insts.push_back(std::move(wi));
    }
    {
        WISAInstruction wi;
        wi.kind = WISAKind::VPD_RANK1;
        wi.layer = 0;
        wi.matrix_id = "self_attn.o_proj";
        wi.u = u_d;
        wi.v = v_d;
        wi.scale = -0.15;
        insts.push_back(std::move(wi));
    }
    {
        WISAInstruction wi;
        wi.kind = WISAKind::ATTN_BILINEAR;
        wi.layer = 0;
        wi.head_group = 0;
        wi.w_qk = mat_scale(outer(u_d, v_d), 0.1);
        wi.w_ov = mat_scale(outer(v_d, u_d), 0.1);
        insts.push_back(std::move(wi));
    }
    {
        WISAInstruction wi;
        wi.kind = WISAKind::ASSOC_MEM_ROW;
        wi.layer = 0;
        wi.k_star = vec_zeros(d_ff);
        wi.k_star[0] = 1.0;
        wi.k_star[1] = 0.5;
        wi.v_star = vec_scale(u_d, 0.5);
        wi.c_inv = eye(d_ff);
        wi.target_norm = 0.5;
        insts.push_back(std::move(wi));
    }
    {
        WISAInstruction wi;
        wi.kind = WISAKind::FIELD_SPECTRAL_OP;
        wi.layer = 0;
        wi.layer_range = {0, 1};
        wi.u_k = {u_d};
        wi.v_k = {v_d};
        wi.coeff_delta = {0.12};
        insts.push_back(std::move(wi));
    }
    {
        WISAInstruction wi;
        wi.kind = WISAKind::NORM_RESCALE;
        wi.layer = 0;
        wi.norm_id = "post_attention_layernorm";
        wi.gamma_scale = Vector(d, 0.02);
        insts.push_back(std::move(wi));
    }

    auto [mem_patched, mem_stats] = apply_wisa(bf16_mem, insts, "in-place");
    CompileStats stream_stats = apply_wisa_streaming_checkpoint(
        src_dir.string(),
        dst_ip_dir.string(),
        insts,
        "in-place");
    check(stream_stats.instructions_applied == mem_stats.instructions_applied,
          "streaming applied count matches in-memory");

    const std::string untouched_src = read_binary_bytes(src_dir / shard_l3);
    const std::string untouched_dst = read_binary_bytes(dst_ip_dir / shard_l3);
    check(untouched_src == untouched_dst, "untouched shard must be byte-for-byte identical");

    {
        MmapSafetensors src_rd((src_dir / shard_l0).string());
        MmapSafetensors dst_rd((dst_ip_dir / shard_l0).string());
        const std::string same_shard_untouched = "model.layers.0.input_layernorm.weight";
        check(dst_rd.tensor_dtype(same_shard_untouched) == "BF16", "preserve BF16 dtype");
        check(src_rd.raw_tensor_bytes(same_shard_untouched) ==
              dst_rd.raw_tensor_bytes(same_shard_untouched),
              "untouched tensor in touched shard preserves raw bytes");
        check(dst_rd.tensor_dtype("model.layers.0.mlp.down_proj.weight") == "BF16",
              "patched tensor keeps BF16 dtype");
        check(dst_rd.tensor_dtype("model.layers.0.self_attn.k_proj.weight") == "BF16",
              "GQA k_proj keeps BF16 dtype");
        check(src_rd.raw_tensor_bytes("model.layers.0.self_attn.k_proj.weight") ==
              dst_rd.raw_tensor_bytes("model.layers.0.self_attn.k_proj.weight"),
              "GQA k_proj raw bytes untouched");
    }

    TransformerModel stream_loaded = TransformerModel::load_checkpoint(dst_ip_dir.string());
    const std::vector<std::string> check_keys = {
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.aic_threshold_route.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.o_proj.weight"
    };
    for (const std::string& k : check_keys) {
        const double diff = frob_norm(mat_sub(stream_loaded.matrices.at(k), mem_patched.matrices.at(k)));
        const double ref = std::max(1.0, frob_norm(mem_patched.matrices.at(k)));
        check(diff / ref < 1e-2, "streaming BF16 patched tensor close to in-memory apply_wisa: " + k);
    }
    check(stream_loaded.config.aic_threshold_routes,
          "streaming checkpoint persists aic_threshold_routes config");
    for (const char* suffix : {"threshold", "polarity", "mask"}) {
        const std::string key = "model.layers.0.mlp.aic_threshold_route." + std::string(suffix);
        check(norm2(vec_sub(stream_loaded.vectors.at(key), mem_patched.vectors.at(key))) < 1e-2,
              "streaming threshold route vector matches in-memory apply_wisa: " + key);
    }

    auto [l_mem, t_mem] = mem_patched.forward({1, 5, 9}, {}, false);
    auto [l_str, t_str] = stream_loaded.forward({1, 5, 9}, {}, false);
    (void)t_mem;
    (void)t_str;
    for (std::size_t t = 0; t < l_mem.size(); ++t) {
        const double ldiff = norm2(vec_sub(l_mem[t], l_str[t]));
        const double lref = std::max(1.0, norm2(l_mem[t]));
        check(ldiff / lref < 2e-2, "streaming BF16 forward logits match in-memory within BF16 precision");
    }

    std::vector<WISAInstruction> elastic_insts;
    {
        WISAInstruction wi;
        wi.kind = WISAKind::SWIGLU_POLE;
        wi.layer = 4;
        wi.append_slot = true;
        wi.polarity = 1;
        wi.w_gate = vec_scale(u_d, 0.5);
        wi.w_up = v_d;
        wi.w_down = vec_scale(u_d, 0.3);
        elastic_insts.push_back(std::move(wi));
    }
    CompileStats el_stats = apply_wisa_streaming_checkpoint(
        src_dir.string(),
        dst_el_dir.string(),
        elastic_insts,
        "elastic");
    check(el_stats.added_layers == 1, "streaming elastic added 1 bypass layer");
    check(el_stats.added_poles == 1, "streaming elastic added 1 pole");
    TransformerModel el_loaded = TransformerModel::load_checkpoint(dst_el_dir.string());
    check(el_loaded.config.num_layers == 5, "streaming elastic updated num_layers=5");
    check(el_loaded.config.intermediate_size == 385, "streaming elastic updated intermediate_size=385");
    auto [l_el, t_el] = el_loaded.forward({1, 5, 9}, {}, false);
    (void)t_el;
    check(l_el.size() == 3, "streaming elastic checkpoint forward ok");

    std::filesystem::remove_all(tmp_root);
    std::cout << "[PASS] test_streaming_bf16_sharded_checkpoint\n";
}

void test_t2_torus_and_parameterized_shifts() {
    const std::size_t d = 24;
    Vector e0(d, 0.0), c1(d, 0.0), s1(d, 0.0), c2(d, 0.0), s2(d, 0.0);
    e0[0] = 1.0;
    c1[1] = 1.0;
    s1[2] = 1.0;
    c2[3] = 1.0;
    s2[4] = 1.0;

    const double dt1 = 0.7;
    const double dt2 = 1.1;
    double prev_err = 1e9;
    for (int k : {3, 5, 7, 9}) {
        auto [slots, err] = build_t2_swiglu_spline(e0, c1, s1, c2, s2, k, dt1, dt2);
        check(slots.size() == static_cast<std::size_t>(k * k), "expected k^2 T2 slots");
        for (const auto& sl : slots) {
            const double un = norm2(sl.w_up);
            check(un > 0.0, "T2 w_up nonzero");
            const double align = dot(sl.w_gate, sl.w_up) / (norm2(sl.w_gate) * un);
            check(std::abs(align - 1.0) < 1e-12, "T2 w_gate parallel to w_up");
        }
        check(err < prev_err, "T2 spline error must decay monotonically across k=3->5->7->9");
        prev_err = err;
    }
    check(prev_err < 5e-3, "T2 k=9 error < 5e-3");

    ModelConfig cfg;
    cfg.num_layers = 8;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 96;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.attention_type = "gqa";
    TransformerModel qwen = TransformerModel::create_synthetic(cfg, 42);

    const std::string t2_src = R"STRATA(
module TorusShiftTest;
manifold PhaseTorus : T2 in Frame<Mid> {
    curvature 1.0;
}
manifold RingShift : S1 in Frame<Mid> {
    curvature 1.0;
}
flow ShiftBoth(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-3, approx <= 1e-2, energy <= 0.5, preserve(CoreFluency, Residual)]
{
    when state ~> PhaseTorus(chart: "torus") as pt {
        let nxt_t = state => PhaseTorus.rotate(dtheta1: 0.7, dtheta2: 1.1);
        state +> nxt_t(gain: 1.0);
    }
    when state ~> RingShift(chart: "angle") as th {
        let nxt_r = state => RingShift.rotate(delta: 1.25);
        state +> nxt_r(gain: 1.0);
    }
}
)STRATA";
    MIRModule mir = compile_source_to_mir(t2_src);
    NeuralAtlas atlas = build_default_atlas(qwen, &mir, 42);
    check(atlas.symbols.contains("PhaseTorus"), "atlas contains PhaseTorus");
    const auto& pt_plane = atlas.symbols.at("PhaseTorus").manifold_plane;
    check(pt_plane.size() == 5, "T2 manifold_plane has 5 orthonormal vectors");
    for (std::size_t i = 0; i < pt_plane.size(); ++i) {
        check(std::abs(norm2(pt_plane[i]) - 1.0) < 1e-12, "T2 basis unit norm");
        for (std::size_t j = i + 1; j < pt_plane.size(); ++j) {
            check(std::abs(dot(pt_plane[i], pt_plane[j])) < 1e-12, "T2 basis orthogonal");
        }
    }

    CIRModule cir = lower_mir_to_cir(mir, cfg, atlas, "in-place");
    check(cir.flows[0].ops.size() == 2, "both T2 and S1 ops lowered");
    check(cir.flows[0].ops[0].approx_error <= 1e-2, "T2 approx error <= 1e-2");
    check(cir.flows[0].ops[1].approx_error <= 1e-2, "S1 parameterized shift approx error <= 1e-2");

    const double t2_err_exact = eval_t2_spline_error(
        cir.flows[0].ops[0].spline_slots,
        pt_plane[0], pt_plane[1], pt_plane[2], pt_plane[3], pt_plane[4],
        0.7, 1.1);
    const double t2_err_wrong = eval_t2_spline_error(
        cir.flows[0].ops[0].spline_slots,
        pt_plane[0], pt_plane[1], pt_plane[2], pt_plane[3], pt_plane[4],
        2.0 * kPi / 7.0, 2.0 * kPi / 5.0);
    check(t2_err_exact <= 1e-2 && t2_err_wrong > 0.1, "T2 used parameterized dtheta1/dtheta2");

    const auto& rs_plane = atlas.symbols.at("RingShift").manifold_plane;
    const double s1_err_exact = eval_s1_spline_error(
        cir.flows[0].ops[1].spline_slots,
        rs_plane[0], rs_plane[1], rs_plane[2], 1.25);
    const double s1_err_wrong = eval_s1_spline_error(
        cir.flows[0].ops[1].spline_slots,
        rs_plane[0], rs_plane[1], rs_plane[2], 2.0 * kPi / 7.0);
    check(s1_err_exact <= 1e-2 && s1_err_wrong > 0.1, "S1 used parameterized delta=1.25");

    CompileOutput out = compile_mir_to_model(mir, qwen, &atlas, "in-place", 2);
    auto [logits, tr] = out.model.forward({1, 2, 3}, {}, false);
    (void)tr;
    check(logits.size() == 3, "T2 + S1 compiled model forward ok");

    std::cout << "[PASS] test_t2_torus_and_parameterized_shifts\n";
}

void test_cir_wisa_optimizer_and_norm_rescale() {
    const std::size_t d = 32;
    CIRModule cir;
    cir.name = "OptCirTest";
    cir.hidden_size = static_cast<int>(d);
    cir.num_layers = 4;
    CIRFlow flow;
    flow.name = "F";

    Matrix m1 = eye(d);
    m1(0, 0) = 0.0; m1(0, 1) = -1.0;
    m1(1, 0) = 1.0; m1(1, 1) = 0.0;
    Matrix m2 = eye(d);
    m2(1, 1) = 0.0; m2(1, 2) = -1.0;
    m2(2, 1) = 1.0; m2(2, 2) = 0.0;

    CIROp zero_vpd;
    zero_vpd.kind = "vpd_rank1";
    zero_vpd.scalar = 0.0;
    zero_vpd.layer = 1;

    CIROp b1;
    b1.kind = "frame_bridge";
    b1.symbol = "B1";
    b1.layer = 1;
    b1.bridge_matrix = m1;
    b1.crosstalk = 1e-12;

    CIROp b2;
    b2.kind = "frame_bridge";
    b2.symbol = "B2";
    b2.layer = 1;
    b2.bridge_matrix = m2;
    b2.crosstalk = 2e-12;

    CIROp zero_pole;
    zero_pole.kind = "swiglu_pole";
    zero_pole.scalar = 0.0;
    zero_pole.layer = 1;

    flow.ops = {zero_vpd, b1, b2, zero_pole};
    cir.flows.push_back(flow);

    CIRModule opt_cir = optimize_cir(cir, 2);
    check(opt_cir.flows[0].ops.size() == 1, "optimize_cir dropped zeros and fused bridges");
    check(opt_cir.flows[0].ops[0].kind == "frame_bridge", "fused op is frame_bridge");
    const Matrix expected_m = mat_mul(m2, m1);
    check(frob_norm(mat_sub(opt_cir.flows[0].ops[0].bridge_matrix, expected_m)) < 1e-12,
          "fused bridge_matrix equals M2 * M1");

    ModelConfig cfg;
    cfg.num_layers = 4;
    cfg.hidden_size = static_cast<int>(d);
    cfg.intermediate_size = 64;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    TransformerModel base = TransformerModel::create_synthetic(cfg, 55);

    Vector u0(d, 0.0), u1(d, 0.0), v0(d, 0.0), v1(d, 0.0);
    u0[0] = 1.0; u1[1] = 1.0;
    v0[2] = 1.0; v1[3] = 1.0;

    std::vector<WISAInstruction> raw_insts;
    for (int i = 0; i < 5; ++i) {
        WISAInstruction wi;
        wi.kind = WISAKind::VPD_RANK1;
        wi.layer = 1;
        wi.matrix_id = "self_attn.o_proj";
        wi.u = normalize(vec_add(u0, vec_scale(u1, 0.5 * (i + 1))));
        wi.v = normalize(vec_add(v0, vec_scale(v1, 0.25 * (i + 1))));
        wi.scale = 0.05 * (i + 1);
        raw_insts.push_back(wi);
    }
    for (int i = 0; i < 2; ++i) {
        WISAInstruction fi;
        fi.kind = WISAKind::FIELD_SPECTRAL_OP;
        fi.layer = 1;
        fi.layer_range = {1, 3};
        fi.u_k = {u0, u1};
        fi.v_k = {v0, v1};
        fi.coeff_delta = {0.02 * (i + 1), -0.01 * (i + 1)};
        raw_insts.push_back(fi);
    }

    std::vector<WISAInstruction> opt1 = optimize_wisa(raw_insts, 1);
    check(opt1.size() <= 3, "optimize_wisa compressed 5 VPDs + 2 FIELDs to <= 3 insts");

    auto [m_unopt, st_unopt] = apply_wisa(base, raw_insts, "in-place");
    auto [m_opt1, st_opt1] = apply_wisa(base, opt1, "in-place");
    (void)st_unopt;
    (void)st_opt1;
    for (int l = 0; l < cfg.num_layers; ++l) {
        const std::string key = "model.layers." + std::to_string(l) + ".self_attn.o_proj.weight";
        const double diff = frob_norm(mat_sub(m_opt1.matrices.at(key), m_unopt.matrices.at(key)));
        check(diff < 1e-10, "SVD-compressed WISA matches unoptimized within < 1e-10");
    }

    std::vector<WISAInstruction> barrier_insts;
    {
        WISAInstruction v1_inst;
        v1_inst.kind = WISAKind::VPD_RANK1;
        v1_inst.layer = 1;
        v1_inst.matrix_id = "self_attn.o_proj";
        v1_inst.u = u0;
        v1_inst.v = v0;
        v1_inst.scale = 0.01;
        WISAInstruction pole_inst;
        pole_inst.kind = WISAKind::SWIGLU_POLE;
        pole_inst.layer = 1;
        pole_inst.slot = 0;
        pole_inst.w_gate = u0;
        pole_inst.w_up = v0;
        pole_inst.w_down = vec_scale(u0, 0.01);
        barrier_insts = {v1_inst, pole_inst, v1_inst};
    }
    std::vector<WISAInstruction> barrier_opt = optimize_wisa(barrier_insts, 2);
    check(barrier_opt.size() == 3, "chronological barrier prevents merging across SWIGLU_POLE");

    std::vector<WISAInstruction> opt2 = optimize_wisa(raw_insts, 2);
    bool emitted_norm_rescale = false;
    for (const auto& wi : opt2) {
        if (wi.kind == WISAKind::NORM_RESCALE && wi.layer == 1) {
            emitted_norm_rescale = true;
            check(wi.gamma_scale.size() == d && wi.gamma_scale[0] < 0.0 && wi.gamma_scale[0] > -0.25,
                  "auto NORM_RESCALE gamma_scale compensates positive excitation");
        }
    }
    check(emitted_norm_rescale, "opt_level=2 automatically emitted NORM_RESCALE for net_e > 0.05");

    std::vector<WISAInstruction> with_explicit = raw_insts;
    WISAInstruction exp_nr;
    exp_nr.kind = WISAKind::NORM_RESCALE;
    exp_nr.layer = 1;
    exp_nr.norm_id = "post_attention_layernorm";
    exp_nr.gamma_scale.assign(d, -0.02);
    with_explicit.push_back(exp_nr);
    std::vector<WISAInstruction> opt2_exp = optimize_wisa(with_explicit, 2);
    int nr_count = 0;
    for (const auto& wi : opt2_exp) {
        if (wi.kind == WISAKind::NORM_RESCALE && wi.layer == 1) {
            ++nr_count;
        }
    }
    check(nr_count == 1, "does not duplicate explicit NORM_RESCALE");

    const std::string opt_src = R"STRATA(
module OptCompileTest;
frame F1 in Mid {
    connect to F2 via BridgeA;
}
frame F2 in Late;
pole P in Frame<Mid>;
flow Run(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-3, approx <= 1e-2, energy <= 0.5, preserve(CoreFluency, Residual)]
{
    let s1 = transport(state, Frame<Mid> -> Frame<Late>);
    when s1 ~> P(threshold: 0.5) {
        state +> P(gain: 0.25);
    }
}
)STRATA";
    MIRModule opt_mir = compile_source_to_mir(opt_src);
    CompileOutput co_0 = compile_mir_to_model(opt_mir, base, nullptr, "in-place", 0);
    CompileOutput co_2 = compile_mir_to_model(opt_mir, base, nullptr, "in-place", 2);
    check(co_0.stats.instructions_applied > 0 && co_2.stats.instructions_applied > 0, "opt_level 0 and 2 both apply");
    auto [lo2, tro2] = co_2.model.forward({1, 2}, {}, false);
    (void)tro2;
    check(lo2.size() == 2, "optimized model forward ok");

    std::cout << "[PASS] test_cir_wisa_optimizer_and_norm_rescale\n";
}

void test_torus_with_imported_preserved_field() {
    const std::string entry = "examples/05_torus_and_imports.strata";
    check(std::filesystem::exists(entry), "05_torus_and_imports.strata exists");
    MIRModule mir = compile_file_with_imports(entry, {"examples/lib"});
    check(mir.declarations.contains("CoreFluency"), "imported CoreFluency present");
    check(mir.declarations.contains("PhaseTorus"), "PhaseTorus present");

    ModelConfig qcfg;
    qcfg.arch_name = "qwen3";
    qcfg.num_layers = 8;
    qcfg.hidden_size = 32;
    qcfg.intermediate_size = 96;
    qcfg.num_attention_heads = 4;
    qcfg.num_key_value_heads = 2;
    qcfg.attention_type = "gqa";
    TransformerModel qwen = TransformerModel::create_synthetic(qcfg, 42);

    NeuralAtlas qatlas = build_default_atlas(qwen, &mir, 42);
    check(qatlas.symbols.contains("CoreFluency") && qatlas.symbols.contains("PhaseTorus"),
          "atlas contains imported and local symbols");
    const SymbolBinding& core = qatlas.symbols.at("CoreFluency");
    const SymbolBinding& torus = qatlas.symbols.at("PhaseTorus");
    for (const Vector& pvec : torus.manifold_plane) {
        for (const SVDMode& mode : core.field_modes) {
            check(std::abs(dot(pvec, mode.u)) < 1e-6, "PhaseTorus orthogonal to preserved CoreFluency u");
            check(std::abs(dot(pvec, mode.v)) < 1e-6, "PhaseTorus orthogonal to preserved CoreFluency v");
        }
    }

    CompileOutput qout = compile_mir_to_model(mir, qwen, &qatlas, "elastic", 2);
    check(qout.stats.instructions_applied > 0, "qwen elastic 05_torus applied instructions");
    check(qout.stats.max_crosstalk <= 0.002 + 1e-9, "qwen elastic 05_torus crosstalk within bound");
    check(qout.stats.max_approx_error <= 0.05 + 1e-9, "qwen elastic 05_torus approx within bound");
    auto [qlo, qtr] = qout.model.forward({1, 2, 3}, {}, false);
    (void)qtr;
    check(qlo.size() == 3, "qwen compiled 05_torus forward ok");

    ModelConfig lcfg;
    lcfg.arch_name = "llama";
    lcfg.num_layers = 4;
    lcfg.hidden_size = 24;
    lcfg.intermediate_size = 64;
    lcfg.num_attention_heads = 4;
    lcfg.num_key_value_heads = 4;
    lcfg.attention_type = "mha";
    TransformerModel llama = TransformerModel::create_synthetic(lcfg, 42);

    CompileOutput lout = compile_mir_to_model(mir, llama, nullptr, "elastic", 2);
    check(lout.stats.instructions_applied > 0, "llama elastic 05_torus applied instructions");
    check(lout.stats.max_crosstalk <= 0.002 + 1e-9, "llama elastic 05_torus crosstalk within bound");
    check(lout.stats.max_approx_error <= 0.05 + 1e-9, "llama elastic 05_torus approx within bound");
    auto [llo, ltr] = lout.model.forward({1, 2}, {}, false);
    (void)ltr;
    check(llo.size() == 2, "llama compiled 05_torus forward ok");

    std::cout << "[PASS] test_torus_with_imported_preserved_field\n";
}

void test_lora_target_and_helper_subflow() {
    ModelConfig cfg;
    cfg.arch_name = "qwen3";
    cfg.torch_dtype = "F32";
    cfg.num_layers = 8;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 96;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.attention_type = "gqa";

    const std::filesystem::path tmp_root =
        std::filesystem::temp_directory_path() / "aic_test_lora_phase4";
    std::filesystem::remove_all(tmp_root);
    std::filesystem::create_directories(tmp_root);
    const std::filesystem::path base_dir = tmp_root / "base";
    const std::filesystem::path lora_dir = tmp_root / "lora_out";

    TransformerModel syn = TransformerModel::create_synthetic(cfg, 42);
    syn.save_checkpoint(base_dir.string());
    TransformerModel base = TransformerModel::load_checkpoint(base_dir.string());

    const std::string src = R"STRATA(
module SubflowLoRA;
const LOOP_THRESH = 0.72;
const POLE_GAIN = 0.25;
dipole LoopPole in Frame<Mid> {
    +pole: LoopOn;
    -pole: LoopOff;
}
field CoreFluency : LowCurvature in Frame<All> {
    rank 4;
    metric Fisher;
}
field StyleDrift : LowCurvature in Frame<Mid> {
    rank 4;
    metric Fisher;
}
flow HelperSuppress(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-3, approx <= 5e-2, energy <= 1.0, preserve(CoreFluency, Residual)]
{
    when state ~> LoopPole(threshold: LOOP_THRESH) {
        state +> LoopPole.-pole(gain: POLE_GAIN);
    }
}
flow MainPatch(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-3, approx <= 5e-2, energy <= 2.0, preserve(CoreFluency, Residual)]
{
    let s1 = HelperSuppress(state);
    let pure_style = StyleDrift decouple CoreFluency with (metric: "fisher", reg: 1e-4);
    s1 +> pure_style(gain: 0.2);
}
)STRATA";

    MIRModule mir = compile_source_to_mir(src);
    bool found_helper = false;
    bool found_main = false;
    for (const auto& f : mir.flows) {
        if (f.name == "HelperSuppress") {
            found_helper = f.is_helper;
        } else if (f.name == "MainPatch") {
            found_main = !f.is_helper;
        }
    }
    check(found_helper && found_main, "MIR marks HelperSuppress is_helper=true and MainPatch false");

    NeuralAtlas atlas = build_default_atlas(base, &mir, 42);
    CIRModule raw_cir = lower_mir_to_cir(mir, base.config, atlas, "lora");
    check(raw_cir.flows.size() == 1 && raw_cir.flows[0].name == "MainPatch",
          "lower_mir_to_cir skips helper sub-flow");

    CompileOutput out = compile_mir_to_model(mir, base, &atlas, "lora", 2);
    check(out.stats.added_poles == 0 && out.stats.added_layers == 0,
          "lora target keeps base tensor shapes fixed");
    check(out.model.config.num_layers == cfg.num_layers &&
          out.model.config.intermediate_size == cfg.intermediate_size,
          "lora target preserves layer count and intermediate_size");

    CompileStats lstats = emit_lora_adapter(base, out.wisa, lora_dir.string(), 16, 16.0);
    check(lstats.instructions_applied == out.stats.instructions_applied,
          "emit_lora_adapter applied expected instructions");
    check(lstats.max_approx_error < 1e-10, "emit_lora_adapter SVD reconstruction accurate");
    check(std::filesystem::exists(lora_dir / "adapter_config.json"), "adapter_config.json written");
    check(std::filesystem::exists(lora_dir / "adapter_model.safetensors"), "adapter_model.safetensors written");

    const TransformerModel via_lora =
        TransformerModel::load_with_lora(base_dir.string(), lora_dir.string());
    double max_mat_diff = 0.0;
    for (const auto& [k, m_ref] : out.model.matrices) {
        check(via_lora.matrices.contains(k), "via_lora contains matrix " + k);
        const double d = frob_norm(mat_sub(m_ref, via_lora.matrices.at(k)));
        max_mat_diff = std::max(max_mat_diff, d);
    }
    check(max_mat_diff < 1e-5, "load_with_lora matches apply_wisa(..., lora) weights");
    for (const auto& [k, v_ref] : out.model.vectors) {
        check(norm2(vec_sub(v_ref, via_lora.vectors.at(k))) < 1e-12,
              "vectors untouched and identical in lora target");
    }

    auto [lo_apply, tr_apply] = out.model.forward({1, 2, 3}, {}, false);
    auto [lo_lora, tr_lora] = via_lora.forward({1, 2, 3}, {}, false);
    (void)tr_apply;
    (void)tr_lora;
    for (std::size_t t = 0; t < lo_apply.size(); ++t) {
        check(norm2(vec_sub(lo_apply[t], lo_lora[t])) < 1e-4, "lora forward matches apply_wisa forward");
    }

    NeuralAtlas dense_atlas = atlas;
    dense_atlas.dormant_poles.clear();
    dense_atlas.symbols["LoopPole"].pole_slots.clear();
    bool inplace_threw = false;
    try {
        (void)lower_cir_to_wisa(raw_cir, base.config, dense_atlas, "in-place");
    } catch (const ApproximationUnmetError&) {
        inplace_threw = true;
    }
    check(inplace_threw, "in-place still throws ApproximationUnmetError when dormant slots exhausted");

    CompileOutput dense_lora = compile_mir_to_model(mir, base, &dense_atlas, "lora", 2);
    check(dense_lora.stats.added_poles == 0 && dense_lora.stats.instructions_applied > 0,
          "lora succeeds cleanly even with zero dormant slots");

    std::filesystem::remove_all(tmp_root);
    std::cout << "[PASS] test_lora_target_and_helper_subflow\n";
}

void test_patch_bundle_revert_and_compose() {
    ModelConfig cfg;
    cfg.arch_name = "qwen3";
    cfg.num_layers = 8;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 96;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.attention_type = "gqa";
    TransformerModel base = TransformerModel::create_synthetic(cfg, 42);

    const std::string src_a = R"STRATA(
module PatchA;
pole PoleA in Frame<Early>;
flow RunA(state: Residual in Frame<Early>) -> Residual in Frame<Early>
    ![crosstalk <= 1e-3, approx <= 5e-2, energy <= 1.0, preserve(Residual)]
{
    state +> PoleA(gain: 0.2);
}
)STRATA";

    const std::string src_b = R"STRATA(
module PatchB;
pole PoleB in Frame<Late>;
flow RunB(state: Residual in Frame<Late>) -> Residual in Frame<Late>
    ![crosstalk <= 1e-3, approx <= 5e-2, energy <= 1.0, preserve(Residual)]
{
    state +> PoleB(gain: 0.3);
}
)STRATA";

    MIRModule mir_a = compile_source_to_mir(src_a);
    MIRModule mir_b = compile_source_to_mir(src_b);
    CompileOutput out_a = compile_mir_to_model(mir_a, base, nullptr, "in-place", 2);
    CompileOutput out_b = compile_mir_to_model(mir_b, base, nullptr, "in-place", 2);

    PatchBundle pb_a = create_patch_bundle(base, out_a.wisa, "in-place", "PatchA");
    PatchBundle pb_b = create_patch_bundle(base, out_b.wisa, "in-place", "PatchB");
    check(!pb_a.matrix_deltas.empty() && !pb_a.base_checksums.empty(), "pb_a has matrix deltas and checksums");

    const std::filesystem::path tmp_dir =
        std::filesystem::temp_directory_path() / "aic_test_npatch_phase4";
    std::filesystem::remove_all(tmp_dir);
    std::filesystem::create_directories(tmp_dir);
    const std::string pa_path = (tmp_dir / "patch_a.npatch").string();
    const std::string pb_path = (tmp_dir / "patch_b.npatch").string();

    save_patch_bundle(pb_a, pa_path);
    save_patch_bundle(pb_b, pb_path);
    PatchBundle pa_loaded = load_patch_bundle(pa_path);
    PatchBundle pb_loaded = load_patch_bundle(pb_path);
    check(pa_loaded.module_name == "PatchA" && pa_loaded.instructions.size() == pb_a.instructions.size(),
          "loaded patch bundle metadata and instructions match");
    check(pa_loaded.base_checksums == pb_a.base_checksums, "loaded base checksums match");

    TransformerModel applied_a = apply_patch_bundle(base, pa_loaded, true);
    for (const auto& [k, m_ref] : out_a.model.matrices) {
        check(frob_norm(mat_sub(m_ref, applied_a.matrices.at(k))) < 1e-12,
              "applied patch bundle matches apply_wisa matrix " + k);
    }
    TransformerModel restored_a = revert_patch_bundle(applied_a, pa_loaded);
    for (const auto& [k, m_orig] : base.matrices) {
        check(frob_norm(mat_sub(m_orig, restored_a.matrices.at(k))) < 1e-12,
              "reverted patch bundle restores base matrix < 1e-12: " + k);
    }
    for (const auto& [k, v_orig] : base.vectors) {
        check(norm2(vec_sub(v_orig, restored_a.vectors.at(k))) < 1e-12,
              "reverted patch bundle restores base vector < 1e-12: " + k);
    }

    TransformerModel restored_direct = revert_patch_bundle(out_a.model, pa_loaded);
    for (const auto& [k, m_orig] : base.matrices) {
        check(frob_norm(mat_sub(m_orig, restored_direct.matrices.at(k))) < 1e-12,
              "direct revert of apply_wisa output restores base < 1e-12: " + k);
    }

    TransformerModel bad_base = base;
    const std::string first_mat_key = pb_a.matrix_deltas.begin()->first;
    bad_base.matrices[first_mat_key](0, 0) += 0.1;
    bool ck_threw = false;
    try {
        (void)apply_patch_bundle(bad_base, pa_loaded, true);
    } catch (const StrataError&) {
        ck_threw = true;
    }
    check(ck_threw, "apply_patch_bundle checksum mismatch throws StrataError");
    TransformerModel ignored_ck = apply_patch_bundle(bad_base, pa_loaded, false);
    check(ignored_ck.config.hidden_size == cfg.hidden_size, "verify_checksum=false succeeds");

    PatchBundle composed_ab = compose_patch_bundles(pa_loaded, pb_loaded);
    TransformerModel applied_ab = apply_patch_bundle(base, composed_ab, true);
    TransformerModel restored_ab = revert_patch_bundle(applied_ab, composed_ab);
    for (const auto& [k, m_orig] : base.matrices) {
        check(frob_norm(mat_sub(m_orig, restored_ab.matrices.at(k))) < 1e-12,
              "composed patch apply+revert restores base < 1e-12: " + k);
    }

    bool slot_conflict_threw = false;
    try {
        (void)compose_patch_bundles(pa_loaded, pa_loaded);
    } catch (const CrosstalkExceededError&) {
        slot_conflict_threw = true;
    }
    check(slot_conflict_threw, "compose_patch_bundles catches conflicting SWIGLU_POLE slot");

    std::vector<WISAInstruction> vpd1_insts(1);
    vpd1_insts[0].kind = WISAKind::VPD_RANK1;
    vpd1_insts[0].layer = 2;
    vpd1_insts[0].matrix_id = "self_attn.o_proj";
    vpd1_insts[0].u = Vector(32, 0.0);
    vpd1_insts[0].v = Vector(32, 0.0);
    vpd1_insts[0].u[0] = 1.0;
    vpd1_insts[0].v[1] = 1.0;
    vpd1_insts[0].scale = 0.25;

    std::vector<WISAInstruction> vpd2_insts = vpd1_insts;
    vpd2_insts[0].scale = -0.2;

    PatchBundle vpd_b1 = create_patch_bundle(base, vpd1_insts, "in-place", "VPD1");
    PatchBundle vpd_b2 = create_patch_bundle(base, vpd2_insts, "in-place", "VPD2");
    bool matrix_conflict_threw = false;
    try {
        (void)compose_patch_bundles(vpd_b1, vpd_b2);
    } catch (const CrosstalkExceededError&) {
        matrix_conflict_threw = true;
    }
    check(matrix_conflict_threw, "compose_patch_bundles catches overlapping matrix update");

    vpd2_insts[0].u.assign(32, 0.0);
    vpd2_insts[0].u[5] = 1.0;
    PatchBundle vpd_orth = create_patch_bundle(base, vpd2_insts, "in-place", "VPD_Orth");
    PatchBundle vpd_comp = compose_patch_bundles(vpd_b1, vpd_orth);
    TransformerModel applied_vo = apply_patch_bundle(base, vpd_comp, true);
    TransformerModel restored_vo = revert_patch_bundle(applied_vo, vpd_comp);
    check(frob_norm(mat_sub(base.matrices.at("model.layers.2.self_attn.o_proj.weight"),
                            restored_vo.matrices.at("model.layers.2.self_attn.o_proj.weight"))) < 1e-12,
          "orthogonal VPD bundles compose and revert cleanly");

    std::filesystem::remove_all(tmp_dir);
    std::cout << "[PASS] test_patch_bundle_revert_and_compose\n";
}

void test_weight_native_atlas_and_dipole_suppression() {
    ModelConfig cfg;
    cfg.arch_name = "qwen3";
    cfg.num_layers = 8;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 96;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.vocab_size = 64;
    cfg.attention_type = "gqa";
    TransformerModel base = TransformerModel::create_synthetic(cfg, 42);

    const std::string src = R"STRATA(
module WeightNativeTest;
dipole TestDipole in Frame<Mid> {
    +pole: ActivePole;
    -pole: SuppressPole;
}
subspace NativeSub<4> in Frame<Mid>;
field NativeField : LowCurvature in Frame<Mid> {
    rank 4;
    metric Fisher;
}
flow SuppressPoleFlow(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-3, approx <= 5e-2, energy <= 2.0, preserve(Residual)]
{
    state -| TestDipole(gain: 0.8);
}
)STRATA";

    MIRModule mir = compile_source_to_mir(src);
    NeuralAtlas atlas = build_default_atlas(base, &mir, 42);
    check(atlas.symbols.contains("TestDipole"), "atlas has TestDipole");
    const SymbolBinding& d_sb = atlas.symbols.at("TestDipole");
    check(d_sb.pole_slots.contains("+pole"), "TestDipole has +pole slot");
    const auto [slot_l, slot_id, pol] = d_sb.pole_slots.at("+pole");
    (void)pol;

    const std::string down_key = "model.layers." + std::to_string(slot_l) + ".mlp.down_proj.weight";
    const Vector raw_col = base.matrices.at(down_key).col(static_cast<std::size_t>(slot_id));
    std::vector<Vector> all_default_pres;
    for (const auto& [pname, pvecs] : atlas.preserve_directions) {
        (void)pname;
        all_default_pres.insert(all_default_pres.end(), pvecs.begin(), pvecs.end());
    }
    std::vector<Vector> pres = gram_schmidt(all_default_pres, 1e-10);
    Vector expected_dir = raw_col;
    for (int pass = 0; pass < 2; ++pass) {
        for (const Vector& p : pres) {
            const double proj = dot(expected_dir, p);
            for (std::size_t i = 0; i < expected_dir.size(); ++i) {
                expected_dir[i] -= proj * p[i];
            }
        }
    }
    expected_dir = normalize(expected_dir);
    check(dot(d_sb.direction, expected_dir) > 0.999999,
          "dipole direction derived from mlp.down_proj column orthogonal to preserved");
    check(dot(d_sb.direction, normalize(raw_col)) > 0.1,
          "dipole direction retains non-trivial weight alignment with raw mlp.down_proj column");
    check(atlas.symbols.at("NativeSub").subspace_basis.size() == 4,
          "NativeSub extracted 4 weight-native basis vectors");

    CompileOutput out = compile_mir_to_model(mir, base, &atlas, "in-place", 2);

    int active_tok = -1;
    double base_act_best = 0.0;
    for (int tid = 0; tid < cfg.vocab_size; ++tid) {
        auto [lo, tr] = base.forward({tid}, {}, true);
        (void)lo;
        const double act_val = tr.mlp_poles[static_cast<std::size_t>(slot_l)]
                                           [static_cast<std::size_t>(slot_id)].act[0];
        if (act_val > base_act_best) {
            base_act_best = act_val;
            active_tok = tid;
        }
    }
    check(active_tok >= 0 && base_act_best > 1e-6, "found token positively activating target dipole slot");

    auto [lo_base, tr_base] = base.forward({active_tok}, {}, true);
    auto [lo_pat, tr_pat] = out.model.forward({active_tok}, {}, true);
    (void)lo_base;
    (void)lo_pat;
    const double base_res_proj =
        dot(tr_base.residual[static_cast<std::size_t>(slot_l + 1)][0], d_sb.direction);
    const double pat_res_proj =
        dot(tr_pat.residual[static_cast<std::size_t>(slot_l + 1)][0], d_sb.direction);
    check(pat_res_proj < base_res_proj - 1e-4,
          "weight-native dipole suppression reduces post-layer residual projection");

    std::cout << "[PASS] test_weight_native_atlas_and_dipole_suppression\n";
}

void test_calibrate_atlas_empirical_and_rome_rewrite() {
    ModelConfig cfg;
    cfg.arch_name = "qwen3";
    cfg.num_layers = 8;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 96;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.vocab_size = 64;
    cfg.attention_type = "gqa";
    TransformerModel base = TransformerModel::create_synthetic(cfg, 42);
    Tokenizer tok = Tokenizer::create_byte_level_demo(cfg.vocab_size);

    const std::string key_prompt = "Aethelgard";
    const std::string target_val_str = "Newhaven";
    const std::vector<int> key_ids = tok.encode(key_prompt);
    const std::vector<int> val_ids = tok.encode(target_val_str);
    check(!val_ids.empty(), "target value encodes to non-empty token ids");
    const int desired_tok = val_ids[0] % cfg.vocab_size;

    auto [base_logits, base_tr] = base.forward(key_ids, {}, false);
    (void)base_tr;
    const Vector& last_base = base_logits.back();
    int base_top = 0;
    for (int i = 1; i < static_cast<int>(last_base.size()); ++i) {
        if (last_base[static_cast<std::size_t>(i)] > last_base[static_cast<std::size_t>(base_top)]) {
            base_top = i;
        }
    }
    check(base_top != desired_tok, "base argmax differs from target rewrite token");

    const std::string src =
        "module EmpiricalRomeTest;\n"
        "rodata Facts<Entity, City> in Frame<Mid> {\n"
        "    anchor factual(\"capital: target\");\n"
        "    entry \"" + key_prompt + "\" => \"Oldhaven\";\n"
        "}\n"
        "flow RewriteFact(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>\n"
        "    ![crosstalk <= 1e-3, approx <= 5e-2, energy <= 1.0, preserve(Residual)]\n"
        "{\n"
        "    rewrite Facts(key: \"" + key_prompt + "\", old: \"Oldhaven\", new: \"" + target_val_str + "\");\n"
        "}\n";

    MIRModule mir = compile_source_to_mir(src);
    NeuralAtlas atlas = build_default_atlas(base, &mir, 42);
    calibrate_atlas_empirical(atlas, base, &mir, &tok);

    check(atlas.mlp_cov_inv.size() == 8, "calibrated mlp_cov_inv populated for all 8 layers");
    check(atlas.mlp_cov_inv.at(2).rows == 96 && atlas.mlp_cov_inv.at(2).cols == 96,
          "mlp_cov_inv layer 2 has shape d_ff x d_ff");
    check(atlas.symbols.at("Facts").rodata_key_vectors.contains(key_prompt),
          "rodata_key_vectors contains empirical key vector");
    check(atlas.symbols.at("Facts").rodata_key_vectors.at(key_prompt).size() == 96,
          "empirical rodata_key_vector has size d_ff");
    check(atlas.symbols.at("Facts").rodata_val_vectors.contains(target_val_str),
          "rodata_val_vectors contains empirical value vector");
    check(std::abs(norm2(atlas.symbols.at("Facts").rodata_val_vectors.at(target_val_str)) - 8.0) < 1e-6,
          "empirical rodata_val_vector normalized and scaled by 8.0");

    const std::filesystem::path atlas_path =
        std::filesystem::temp_directory_path() / "aic_phase5_calibrated_atlas.natlas";
    atlas.save(atlas_path.string());
    NeuralAtlas reloaded_atlas = NeuralAtlas::load(atlas_path.string());
    std::filesystem::remove(atlas_path);
    check(reloaded_atlas.mlp_cov_inv.size() == 8 &&
          reloaded_atlas.symbols.at("Facts").rodata_key_vectors.contains(key_prompt) &&
          reloaded_atlas.symbols.at("Facts").rodata_val_vectors.contains(target_val_str),
          "NeuralAtlas save/load round-trips empirical ROME calibration");

    CompileOutput out = compile_mir_to_model(mir, base, &reloaded_atlas, "in-place", 2);
    auto [patched_logits, patched_tr] = out.model.forward(key_ids, {}, false);
    (void)patched_tr;
    const Vector& last_patched = patched_logits.back();
    int patched_top = 0;
    for (int i = 1; i < static_cast<int>(last_patched.size()); ++i) {
        if (last_patched[static_cast<std::size_t>(i)] >
            last_patched[static_cast<std::size_t>(patched_top)]) {
            patched_top = i;
        }
    }
    check(patched_top == desired_tok,
          "empirical ROME rewrite shifts top predicted next token to rewritten value first token");

    std::cout << "[PASS] test_calibrate_atlas_empirical_and_rome_rewrite\n";
}

void test_streaming_patch_bundle_and_int4_codegen() {
    ModelConfig cfg;
    cfg.arch_name = "qwen3-mini";
    cfg.num_layers = 8;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 96;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.vocab_size = 64;
    cfg.attention_type = "gqa";
    cfg.use_rope = true;
    cfg.qk_norm = true;
    cfg.torch_dtype = "F32";

    const std::filesystem::path tmp_root =
        std::filesystem::temp_directory_path() / "aic_test_phase5_streaming_bundle";
    std::filesystem::remove_all(tmp_root);
    const std::filesystem::path f32_base_dir = tmp_root / "f32_sharded_base";
    const std::filesystem::path ip_patched_dir = tmp_root / "ip_patched";
    const std::filesystem::path ip_restored_dir = tmp_root / "ip_restored";
    const std::filesystem::path el_patched_dir = tmp_root / "el_patched";
    const std::filesystem::path el_restored_dir = tmp_root / "el_restored";
    const std::filesystem::path i4_src_dir = tmp_root / "i4_src";
    const std::filesystem::path i4_dst_dir = tmp_root / "i4_dst";
    std::filesystem::create_directories(f32_base_dir);
    std::filesystem::create_directories(i4_src_dir);

    TransformerModel syn = TransformerModel::create_synthetic(cfg, 123);
    std::map<std::string, TensorSpec> f32_specs;
    for (const auto& [k, m] : syn.matrices) {
        f32_specs[k] = TensorSpec::from_matrix(m, "F32");
    }
    for (const auto& [k, v] : syn.vectors) {
        f32_specs[k] = TensorSpec::from_vector(v, "F32");
    }
    save_safetensors_sharded(
        f32_base_dir.string(),
        f32_specs,
        200u * 1024u,
        {{"arch_name", cfg.arch_name}, {"torch_dtype", "F32"}});
    {
        std::ofstream ofs(f32_base_dir / "config.json", std::ios::trunc);
        ofs << dump_json(cfg.to_json(), 2) << "\n";
    }
    Tokenizer tok = Tokenizer::create_byte_level_demo(cfg.vocab_size);
    tok.save((f32_base_dir / "tokenizer.json").string());

    TransformerModel f32_base = TransformerModel::load_checkpoint(f32_base_dir.string());
    const JsonValue f32_idx = parse_json(read_binary_bytes(f32_base_dir / "model.safetensors.index.json"));
    std::set<std::string> f32_shards;
    for (const auto& [k, sf] : f32_idx["weight_map"].as_object()) {
        (void)k;
        f32_shards.insert(sf.as_string());
    }
    check(f32_shards.size() >= 2, "expected multi-shard F32 checkpoint");

    const std::size_t d = static_cast<std::size_t>(cfg.hidden_size);
    Vector dir(d, 0.0);
    dir[2] = 0.6;
    dir[5] = 0.8;

    std::vector<WISAInstruction> ip_insts;
    {
        WISAInstruction wi;
        wi.kind = WISAKind::SWIGLU_POLE;
        wi.layer = 1;
        wi.slot = 3;
        wi.polarity = 1;
        wi.w_gate = vec_scale(dir, 1.5);
        wi.w_up = vec_scale(dir, 0.5);
        wi.w_down = vec_scale(dir, 0.25);
        ip_insts.push_back(std::move(wi));
    }

    PatchBundle ip_bundle = create_patch_bundle(f32_base, ip_insts, "in-place", "InPlacePatch");
    const std::string ip_patch_path = (tmp_root / "ip.npatch").string();
    save_patch_bundle(ip_bundle, ip_patch_path);
    PatchBundle ip_loaded = load_patch_bundle(ip_patch_path);
    check(ip_loaded.orig_num_layers == cfg.num_layers &&
          ip_loaded.orig_intermediate_size == cfg.intermediate_size,
          "in-place bundle preserves orig_num_layers and orig_intermediate_size");

    apply_patch_bundle_streaming(f32_base_dir.string(), ip_patched_dir.string(), ip_loaded, true);
    TransformerModel ip_mem = apply_patch_bundle(f32_base, ip_loaded, true);
    TransformerModel ip_disk = TransformerModel::load_checkpoint(ip_patched_dir.string());
    for (const auto& [k, m] : ip_mem.matrices) {
        check(frob_norm(mat_sub(m, ip_disk.matrices.at(k))) < 1e-5,
              "streaming apply matrix matches in-memory apply for " + k);
    }

    revert_patch_bundle_streaming(ip_patched_dir.string(), ip_restored_dir.string(), ip_loaded);
    TransformerModel ip_restored = TransformerModel::load_checkpoint(ip_restored_dir.string());
    for (const auto& [k, m] : f32_base.matrices) {
        check(frob_norm(mat_sub(m, ip_restored.matrices.at(k))) < 1e-6,
              "in-place streaming revert restores F32 checkpoint within < 1e-6 for " + k);
    }
    for (const auto& [k, v] : f32_base.vectors) {
        check(norm2(vec_sub(v, ip_restored.vectors.at(k))) < 1e-6,
              "in-place streaming revert restores F32 vector within < 1e-6 for " + k);
    }

    PatchBundle ip_f32_exact = ip_loaded;
    for (auto& [k, delta] : ip_f32_exact.matrix_deltas) {
        delta = mat_sub(ip_disk.matrices.at(k), f32_base.matrices.at(k));
    }
    const std::string ip_exact_path = (tmp_root / "ip_exact.npatch").string();
    save_patch_bundle(ip_f32_exact, ip_exact_path);
    PatchBundle ip_exact_loaded = load_patch_bundle(ip_exact_path);
    const std::filesystem::path ip_exact_dir = tmp_root / "ip_exact_restored";
    std::filesystem::copy(ip_patched_dir, ip_exact_dir, std::filesystem::copy_options::recursive);
    revert_patch_bundle_streaming(ip_exact_dir.string(), ip_exact_dir.string(), ip_exact_loaded);
    TransformerModel ip_exact_restored = TransformerModel::load_checkpoint(ip_exact_dir.string());
    for (const auto& [k, m] : f32_base.matrices) {
        check(frob_norm(mat_sub(m, ip_exact_restored.matrices.at(k))) == 0.0,
              "F32-parity streaming in-place revert gives exact zero frob_diff for " + k);
    }

    std::vector<WISAInstruction> el_insts;
    {
        WISAInstruction wi;
        wi.kind = WISAKind::SWIGLU_POLE;
        wi.layer = 1;
        wi.slot = cfg.intermediate_size;
        wi.append_slot = true;
        wi.polarity = 1;
        wi.w_gate = vec_scale(dir, 1.0);
        wi.w_up = vec_scale(dir, 0.5);
        wi.w_down = vec_scale(dir, 0.25);
        el_insts.push_back(std::move(wi));
    }
    {
        WISAInstruction wi;
        wi.kind = WISAKind::SWIGLU_POLE;
        wi.layer = cfg.num_layers;
        wi.slot = 0;
        wi.append_slot = false;
        wi.polarity = 1;
        wi.w_gate = vec_scale(dir, 0.75);
        wi.w_up = vec_scale(dir, 0.5);
        wi.w_down = vec_scale(dir, 0.25);
        el_insts.push_back(std::move(wi));
    }

    PatchBundle el_bundle = create_patch_bundle(f32_base, el_insts, "elastic", "ElasticPatch");
    const std::string el_patch_path = (tmp_root / "el.npatch").string();
    save_patch_bundle(el_bundle, el_patch_path);
    PatchBundle el_loaded = load_patch_bundle(el_patch_path);
    check(el_loaded.orig_num_layers == 8 && el_loaded.orig_intermediate_size == 96,
          "elastic bundle persisted orig_num_layers=8 and orig_intermediate_size=96");

    apply_patch_bundle_streaming(f32_base_dir.string(), el_patched_dir.string(), el_loaded, true);
    TransformerModel el_disk = TransformerModel::load_checkpoint(el_patched_dir.string());
    check(el_disk.config.num_layers == 9 && el_disk.config.intermediate_size == 97,
          "elastic streaming apply widened num_layers=9 and intermediate_size=97");
    auto [el_lo, el_tr] = el_disk.forward({1, 2, 3}, {}, false);
    (void)el_tr;
    check(el_lo.size() == 3, "elastic streaming patched checkpoint forward ok");

    revert_patch_bundle_streaming(el_patched_dir.string(), el_restored_dir.string(), el_loaded);
    TransformerModel el_restored = TransformerModel::load_checkpoint(el_restored_dir.string());
    check(el_restored.config.num_layers == 8 && el_restored.config.intermediate_size == 96,
          "elastic streaming revert shrunk num_layers=8 and intermediate_size=96");
    check(el_restored.matrices.size() == f32_base.matrices.size() &&
          el_restored.vectors.size() == f32_base.vectors.size(),
          "elastic streaming revert removed bypass layer tensors cleanly");
    for (const auto& [k, m] : f32_base.matrices) {
        check(el_restored.matrices.contains(k), "elastic revert kept matrix " + k);
        check(frob_norm(mat_sub(m, el_restored.matrices.at(k))) == 0.0,
              "elastic streaming revert exact zero frob_diff for " + k);
    }

    ModelConfig i4_cfg = cfg;
    i4_cfg.torch_dtype = "I4";
    TransformerModel i4_syn = TransformerModel::create_synthetic(i4_cfg, 321);
    std::map<std::string, TensorSpec> i4_specs;
    for (const auto& [k, m] : i4_syn.matrices) {
        i4_specs[k] = TensorSpec::from_matrix(m, "I4");
    }
    for (const auto& [k, v] : i4_syn.vectors) {
        i4_specs[k] = TensorSpec::from_vector(v, "I4");
    }
    save_safetensors_sharded(
        i4_src_dir.string(),
        i4_specs,
        30u * 1024u,
        {{"arch_name", i4_cfg.arch_name}, {"torch_dtype", "I4"}});
    {
        std::ofstream ofs(i4_src_dir / "config.json", std::ios::trunc);
        ofs << dump_json(i4_cfg.to_json(), 2) << "\n";
    }

    std::vector<WISAInstruction> i4_insts;
    {
        WISAInstruction wi;
        wi.kind = WISAKind::VPD_RANK1;
        wi.layer = 2;
        wi.matrix_id = "self_attn.o_proj";
        wi.u = dir;
        wi.v = Vector(d, 0.0);
        wi.v[0] = 1.0;
        wi.scale = 0.25;
        i4_insts.push_back(std::move(wi));
    }
    CompileStats i4_stats = apply_wisa_streaming_checkpoint(
        i4_src_dir.string(),
        i4_dst_dir.string(),
        i4_insts,
        "in-place");
    check(i4_stats.instructions_applied == 1, "INT4 streaming apply applied 1 instruction");

    const std::string o_proj_key = "model.layers.2.self_attn.o_proj.weight";
    TensorSpec o_spec = load_single_tensor_sharded(i4_dst_dir.string(), o_proj_key);
    check(o_spec.dtype == "I4", "INT4 streaming preserved I4 dtype");
    TransformerModel i4_loaded = TransformerModel::load_checkpoint(i4_dst_dir.string());
    auto [i4_lo, i4_tr] = i4_loaded.forward({1, 2, 3}, {}, false);
    (void)i4_tr;
    check(i4_lo.size() == 3, "INT4 streaming patched checkpoint forward ok");

    std::filesystem::remove_all(tmp_root);
    std::cout << "[PASS] test_streaming_patch_bundle_and_int4_codegen\n";
}

void test_memit_four_fact_batch_edit() {
    ModelConfig cfg;
    cfg.arch_name = "qwen3";
    cfg.num_layers = 8;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 96;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.vocab_size = 64;
    cfg.attention_type = "gqa";
    TransformerModel base = TransformerModel::create_synthetic(cfg, 42);
    Tokenizer tok = Tokenizer::create_byte_level_demo(cfg.vocab_size);

    const std::vector<std::string> prompts = {
        "FactAlphaCity",
        "FactBetaRiver",
        "FactGammaPeak",
        "FactDeltaPort"
    };
    auto argmax_last = [](const TransformerModel& m, const std::vector<int>& ids) -> int {
        auto [logits, tr] = m.forward(ids, {}, false);
        (void)tr;
        const Vector& last = logits.back();
        int best = 0;
        for (int i = 1; i < static_cast<int>(last.size()); ++i) {
            if (last[static_cast<std::size_t>(i)] > last[static_cast<std::size_t>(best)]) {
                best = i;
            }
        }
        return best;
    };

    std::vector<int> base_tops;
    for (const auto& p : prompts) {
        base_tops.push_back(argmax_last(base, tok.encode(p)));
    }

    std::vector<std::string> target_vals;
    std::vector<int> expected_toks;
    std::set<int> used_toks;
    for (std::size_t m = 0; m < prompts.size(); ++m) {
        std::string chosen_val;
        int chosen_tok = -1;
        for (const std::string& cand : std::vector<std::string>{
                 "Newhaven", "Celestia", "Boreal", "Vesper", "Oakhaven", "Kestrel", "Talisman", "Solis"}) {
            std::vector<int> enc = tok.encode(cand);
            if (enc.empty()) {
                continue;
            }
            const int tid = enc[0] % cfg.vocab_size;
            if (tid != base_tops[m] && !used_toks.contains(tid)) {
                chosen_val = cand;
                chosen_tok = tid;
                used_toks.insert(tid);
                break;
            }
        }
        check(!chosen_val.empty(), "found distinct target token for MEMIT fact " + std::to_string(m));
        target_vals.push_back(chosen_val);
        expected_toks.push_back(chosen_tok);
    }

    const std::string src =
        "module MemitBatchTest;\n"
        "rodata Facts<Entity, City> in Frame<Mid> {\n"
        "    anchor factual(\"Fact entity maps to city\");\n"
        "    entry \"" + prompts[0] + "\" => \"Old0\";\n"
        "    entry \"" + prompts[1] + "\" => \"Old1\";\n"
        "}\n"
        "flow BatchEdit(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>\n"
        "    ![crosstalk <= 1e-3, approx <= 5e-2, energy <= 2.0, preserve(Residual)]\n"
        "{\n"
        "    rewrite Facts(key: \"" + prompts[0] + "\", old: \"Old0\", new: \"" + target_vals[0] + "\");\n"
        "    rewrite Facts(key: \"" + prompts[1] + "\", old: \"Old1\", new: \"" + target_vals[1] + "\");\n"
        "    insert Facts(key: \"" + prompts[2] + "\", value: \"" + target_vals[2] + "\", sharpness: 12.0);\n"
        "    insert Facts(key: \"" + prompts[3] + "\", value: \"" + target_vals[3] + "\", sharpness: 12.0);\n"
        "}\n";

    MIRModule mir = compile_source_to_mir(src);
    NeuralAtlas atlas = build_default_atlas(base, &mir, 42);
    calibrate_atlas_empirical(atlas, base, &mir, &tok);

    CompileOutput out = compile_mir_to_model(mir, base, &atlas, "in-place", 2);
    check(out.wisa.size() == 4, "4 contiguous ASSOC_MEM_ROW instructions emitted");
    check(out.stats.instructions_applied == 4, "4 MEMIT instructions applied in batch");

    for (std::size_t m = 0; m < prompts.size(); ++m) {
        const int pred = argmax_last(out.model, tok.encode(prompts[m]));
        check(pred == expected_toks[m],
              "MEMIT simultaneous fact " + std::to_string(m) + " predicted rewritten target token");
    }

    const std::size_t d = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t d_ff = static_cast<std::size_t>(cfg.intermediate_size);
    Vector dup_k(d_ff, 0.0);
    dup_k[4] = 1.0;
    Vector v_old(d, 0.5);
    Vector v_new(d, -1.25);
    WISAInstruction d1;
    d1.kind = WISAKind::ASSOC_MEM_ROW;
    d1.layer = 3;
    d1.k_star = dup_k;
    d1.v_star = v_old;
    WISAInstruction d2 = d1;
    d2.v_star = v_new;
    auto [dup_patched, dup_stats] = apply_wisa(base, {d1, d2}, "in-place");
    check(dup_stats.instructions_applied == 2, "duplicate-key MEMIT batch applied both instructions");
    const Vector got_dup = mat_vec(dup_patched.matrices.at("model.layers.3.mlp.down_proj.weight"), dup_k);
    check(norm2(vec_sub(got_dup, v_new)) < 1e-10, "duplicate-key MEMIT keeps latest target v_star");

    std::cout << "[PASS] test_memit_four_fact_batch_edit\n";
}

void test_streaming_runtime_resolver_and_patch_only() {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "aic_test_streaming_runtime";
    std::filesystem::remove_all(root);

    for (const std::string& dtype : {"F32", "F16", "BF16", "F8_E4M3", "I8", "I4"}) {
        ModelConfig cfg;
        cfg.num_layers = 2;
        cfg.hidden_size = 8;
        cfg.intermediate_size = 16;
        cfg.num_attention_heads = 2;
        cfg.num_key_value_heads = 1;
        cfg.vocab_size = 32;
        cfg.attention_type = "gqa";
        cfg.torch_dtype = dtype;
        cfg.arch_name = "stream-test";
        TransformerModel model = TransformerModel::create_synthetic(cfg, 909);
        const std::filesystem::path dir = root / dtype;
        model.save_checkpoint(dir.string());
        const TransformerModel loaded = TransformerModel::load_checkpoint(dir.string());
        const std::vector<int> ids = {1, 3, 5};
        const auto [full_logits, full_trace] = loaded.forward(ids, {}, true);
        const auto [stream_logits, stream_trace] = forward_checkpoint_streaming(dir.string(), ids, true);
        check(stream_trace.residual.size() == full_trace.residual.size() &&
                  stream_trace.mlp_poles.size() == full_trace.mlp_poles.size(),
              "streaming trace shape matches full runtime for " + dtype);
        for (std::size_t t = 0; t < full_logits.size(); ++t) {
            const double diff = norm2(vec_sub(full_logits[t], stream_logits[t]));
            const double ref = std::max(1.0, norm2(full_logits[t]));
            check(diff / ref < 1e-10, "streaming logits match full runtime for " + dtype);
        }
    }

    const std::filesystem::path model_dir = root / "BF16";
    const TransformerModel base = TransformerModel::load_checkpoint(model_dir.string());
    const std::string src = R"STRATA(
module StreamingConspiracyProbe;
dipole PatternCertainty in Frame<All> {
    +pole: HiddenPlan {
        anchor contrastive(
            pos: ["Every coincidence proves a hidden coordinated plan."],
            neg: ["Coincidence alone is not evidence of coordination."]
        );
    }
    -pole: EvidenceCalibration {
        anchor probe("Distinguish evidence from speculation.");
    }
}
flow Amplify(state: Residual in Frame<All>) -> Residual in Frame<All>
    ![crosstalk <= 1.0, approx <= 1.0, energy <= 2.0]
{
    state +> PatternCertainty.+pole(gain: 0.2);
}
)STRATA";
    const MIRModule mir = compile_source_to_mir(src);
    const Tokenizer tok = Tokenizer::create_byte_level_demo(base.config.vocab_size);
    const NeuralAtlas full_atlas = resolve_symbols_for_target(mir, base, nullptr, 42, &tok);
    const NeuralAtlas stream_atlas = resolve_symbols_for_target_streaming(
        mir, model_dir.string(), nullptr, 42, &tok);
    const SymbolBinding& full_symbol = full_atlas.symbols.at("PatternCertainty");
    const SymbolBinding& stream_symbol = stream_atlas.symbols.at("PatternCertainty");
    check(full_symbol.pole_slots == stream_symbol.pole_slots,
          "streaming resolver selects the same empirical pole slots");
    check(dot(full_symbol.direction, stream_symbol.direction) > 1.0 - 1e-10,
          "streaming resolver reproduces the empirical contrastive direction");

    WISAInstruction rank1;
    rank1.kind = WISAKind::VPD_RANK1;
    rank1.layer = 0;
    rank1.matrix_id = "self_attn.o_proj";
    rank1.u = Vector(8, 0.0);
    rank1.v = Vector(8, 0.0);
    rank1.u[0] = 1.0;
    rank1.v[1] = 1.0;
    rank1.scale = 0.125;
    WISAInstruction route;
    route.kind = WISAKind::THRESHOLDED_ROUTE;
    route.layer = 0;
    route.slot = 0;
    route.polarity = 1;
    route.u = rank1.u;
    route.scale = 0.2;
    route.threshold = 0.4;
    const std::vector<WISAInstruction> insts = {rank1, route};
    const PatchBundle full_bundle = create_patch_bundle(base, insts, "in-place", "full");
    const PatchBundle stream_bundle = create_patch_bundle_streaming(
        model_dir.string(), insts, "in-place", "stream");
    check(full_bundle.matrix_deltas.size() == stream_bundle.matrix_deltas.size() &&
              full_bundle.vector_deltas.size() == stream_bundle.vector_deltas.size(),
          "patch-only streaming emits the same delta set");
    for (const auto& [key, delta] : full_bundle.matrix_deltas) {
        check(frob_norm(mat_sub(delta, stream_bundle.matrix_deltas.at(key))) < 1e-12,
              "patch-only matrix delta matches full compilation: " + key);
    }
    for (const auto& [key, delta] : full_bundle.vector_deltas) {
        check(norm2(vec_sub(delta, stream_bundle.vector_deltas.at(key))) < 1e-12,
              "patch-only vector delta matches full compilation: " + key);
    }

    std::filesystem::remove_all(root);
    std::cout << "[PASS] test_streaming_runtime_resolver_and_patch_only\n";
}

void test_empirical_fisher_metrics_and_decouple() {
    ModelConfig cfg;
    cfg.arch_name = "qwen3";
    cfg.num_layers = 8;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 96;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.vocab_size = 64;
    cfg.attention_type = "gqa";
    TransformerModel base = TransformerModel::create_synthetic(cfg, 42);
    Tokenizer tok = Tokenizer::create_byte_level_demo(cfg.vocab_size);

    const std::string src = R"STRATA(
module EmpiricalFisherTest;
field CoreFluency : LowCurvature in Frame<All> {
    rank 4;
    metric Fisher;
}
field StyleDrift : LowCurvature in Frame<Mid> {
    rank 4;
    metric Fisher;
}
field EuclidDrift : LowCurvature in Frame<Mid> {
    rank 4;
    metric Euclidean;
}
flow DecoupleFisher(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-3, approx <= 5e-2, energy <= 2.0, preserve(CoreFluency, Residual)]
{
    let pure_f = StyleDrift decouple CoreFluency with (metric: "fisher", reg: 1e-4);
    state +> pure_f(gain: 0.25);
    let pure_e = EuclidDrift decouple CoreFluency with (metric: "euclidean", reg: 1e-4);
    state +> pure_e(gain: 0.25);
}
)STRATA";

    MIRModule mir = compile_source_to_mir(src);
    NeuralAtlas atlas = build_default_atlas(base, &mir, 42);
    calibrate_atlas_empirical(atlas, base, &mir, &tok);

    check(atlas.fisher_metrics.size() == 8, "fisher_metrics populated for all 8 layers");
    for (int l = 0; l < 8; ++l) {
        const LayerFisherDiag& fd = atlas.fisher_metrics.at(l);
        check(fd.g_in.size() == 32 && fd.g_out.size() == 32, "fisher diagonal vectors match d");
        double min_v = fd.g_in[0];
        double max_v = fd.g_in[0];
        for (double x : fd.g_in) {
            check(x >= 1e-4, "g_in respects eps_floor");
            min_v = std::min(min_v, x);
            max_v = std::max(max_v, x);
        }
        check(max_v - min_v > 1e-3, "g_in is non-uniform empirical Fisher diagonal");
    }

    const std::filesystem::path atlas_path =
        std::filesystem::temp_directory_path() / "aic_phase6_fisher_atlas.natlas";
    atlas.save(atlas_path.string());
    NeuralAtlas reloaded = NeuralAtlas::load(atlas_path.string());
    std::filesystem::remove(atlas_path);
    check(reloaded.fisher_metrics.size() == 8, "fisher_metrics round-trips via NeuralAtlas save/load");
    check(norm2(vec_sub(atlas.fisher_metrics.at(2).g_out, reloaded.fisher_metrics.at(2).g_out)) < 1e-12,
          "fisher_metrics g_out round-trips accurately");

    CIRModule cir = lower_mir_to_cir(mir, base.config, reloaded, "in-place");
    check(cir.flows.size() == 1 && cir.flows[0].ops.size() == 4, "all 4 field ops lowered cleanly");

    const auto lr = base.config.stage_to_layers("Mid");
    Vector avg_out(32, 0.0);
    Vector avg_in(32, 0.0);
    const double inv = 1.0 / static_cast<double>(lr.second - lr.first);
    for (int l = lr.first; l < lr.second; ++l) {
        for (std::size_t i = 0; i < 32; ++i) {
            avg_out[i] += reloaded.fisher_metrics.at(l).g_out[i] * inv;
            avg_in[i] += reloaded.fisher_metrics.at(l).g_in[i] * inv;
        }
    }

    const double emp_cos = fisher_cosine(
        cir.flows[0].ops[0].field_modes,
        reloaded.symbols.at("CoreFluency").field_modes,
        avg_out,
        avg_in);
    check(std::abs(cir.flows[0].ops[0].crosstalk - emp_cos) < 1e-12,
          "field_decouple uses averaged empirical Fisher metric over flow stage");
    check(emp_cos <= 1e-3 && emp_cos < 1e-10,
          "empirical Fisher decouple satisfies crosstalk <= crosstalk_bound");

    const Vector ones(32, 1.0);
    const double pure_f_euc_cos = fisher_cosine(
        cir.flows[0].ops[0].field_modes,
        reloaded.symbols.at("CoreFluency").field_modes,
        ones,
        ones);
    check(pure_f_euc_cos > 1e-3,
          "Fisher-decoupled field is specifically orthogonal under empirical Fisher metric rather than Euclidean");

    const double euc_cos = fisher_cosine(
        cir.flows[0].ops[2].field_modes,
        reloaded.symbols.at("CoreFluency").field_modes,
        ones,
        ones);
    check(std::abs(cir.flows[0].ops[2].crosstalk - euc_cos) < 1e-12 && euc_cos < 1e-10,
          "explicit Euclidean metric op uses Euclidean inner product");
    const double pure_e_fisher_cos = fisher_cosine(
        cir.flows[0].ops[2].field_modes,
        reloaded.symbols.at("CoreFluency").field_modes,
        avg_out,
        avg_in);
    check(pure_e_fisher_cos > 1e-3,
          "Euclidean-decoupled field differs measurably under empirical Fisher metric");

    std::cout << "[PASS] test_empirical_fisher_metrics_and_decouple\n";
}

}

int main() {
    test_atlas_save_load();
    test_s1_tessellation_and_decay();
    test_nullspace_and_errors();
    test_fisher_decouple_and_field();
    test_attention_modes_and_conflicts();
    test_depth_and_slot_exhaustion_and_elastic();
    test_assoc_mem_row_and_all_four_cases();
    test_multithreaded_cir_and_large_scale();
    test_streaming_bf16_sharded_checkpoint();
    test_t2_torus_and_parameterized_shifts();
    test_cir_wisa_optimizer_and_norm_rescale();
    test_torus_with_imported_preserved_field();
    test_lora_target_and_helper_subflow();
    test_patch_bundle_revert_and_compose();
    test_weight_native_atlas_and_dipole_suppression();
    test_calibrate_atlas_empirical_and_rome_rewrite();
    test_streaming_patch_bundle_and_int4_codegen();
    test_memit_four_fact_batch_edit();
    test_streaming_runtime_resolver_and_patch_only();
    test_empirical_fisher_metrics_and_decouple();
    std::cout << "ALL test_backend checks passed (20/20 test groups)\n";
    return 0;
}
