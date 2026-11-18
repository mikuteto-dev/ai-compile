#include "aic/decompiler.hpp"
#include "aic/errors.hpp"
#include "aic/json_min.hpp"
#include "aic/linker.hpp"
#include "aic/mir.hpp"
#include "aic/nelf.hpp"
#include "aic/safetensors.hpp"
#include "aic/tokenizer.hpp"
#include "aic/tracer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace aic;

namespace {

std::string read_file_str(const std::string& path) {
    std::ifstream ifs(path);
    assert(ifs.good());
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

std::string insert_into_lifted_flow(const std::string& src, const std::string& body) {
    const std::size_t flow_pos = src.find("flow LiftedProgram");
    assert(flow_pos != std::string::npos);
    const std::size_t open_pos = src.find('{', flow_pos);
    assert(open_pos != std::string::npos);
    const std::size_t close_pos = src.find('}', open_pos);
    assert(close_pos != std::string::npos);
    std::string out = src;
    out.insert(close_pos, body);
    return out;
}

double max_model_frob_diff(const TransformerModel& a, const TransformerModel& b) {
    assert(a.matrices.size() == b.matrices.size());
    assert(a.vectors.size() == b.vectors.size());
    double max_diff = 0.0;
    for (const auto& [k, ma] : a.matrices) {
        assert(b.matrices.contains(k));
        const Matrix& mb = b.matrices.at(k);
        assert(ma.rows == mb.rows && ma.cols == mb.cols);
        max_diff = std::max(max_diff, frob_norm(mat_sub(ma, mb)));
    }
    for (const auto& [k, va] : a.vectors) {
        assert(b.vectors.contains(k));
        const Vector& vb = b.vectors.at(k);
        assert(va.size() == vb.size());
        max_diff = std::max(max_diff, norm2(vec_sub(va, vb)));
    }
    return max_diff;
}

double max_logits_diff(
    const TransformerModel& a,
    const TransformerModel& b,
    const std::vector<int>& tokens) {
    auto [la, ta] = a.forward(tokens, {}, false);
    auto [lb, tb] = b.forward(tokens, {}, false);
    (void)ta;
    (void)tb;
    assert(la.size() == lb.size());
    double max_d = 0.0;
    for (std::size_t t = 0; t < la.size(); ++t) {
        max_d = std::max(max_d, norm2(vec_sub(la[t], lb[t])));
    }
    return max_d;
}

ModelConfig make_qwen3_mini() {
    ModelConfig cfg;
    cfg.num_layers = 8;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 96;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.vocab_size = 64;
    cfg.attention_type = "gqa";
    cfg.arch_name = "qwen3-mini";
    return cfg;
}

ModelConfig make_llama_mini() {
    ModelConfig cfg;
    cfg.num_layers = 4;
    cfg.hidden_size = 24;
    cfg.intermediate_size = 64;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.vocab_size = 64;
    cfg.attention_type = "gqa";
    cfg.arch_name = "llama-mini";
    return cfg;
}

ModelConfig make_mla_mini() {
    ModelConfig cfg;
    cfg.num_layers = 4;
    cfg.hidden_size = 24;
    cfg.intermediate_size = 64;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 4;
    cfg.kv_lora_rank = 8;
    cfg.vocab_size = 64;
    cfg.attention_type = "mla";
    cfg.arch_name = "deepseek-mini";
    return cfg;
}

void test_nelf_save_load_and_reconstruct() {
    TransformerModel model = TransformerModel::create_synthetic(make_qwen3_mini(), 42);
    auto [nelf, strata_src] = lift_model(model, "TestSaveLoad");
    (void)strata_src;

    const TransformerModel recon = nelf.reconstruct_model();
    assert(max_model_frob_diff(model, recon) < 1e-12);

    const std::filesystem::path tmp_dir =
        std::filesystem::temp_directory_path() / "aic_test_nelf_save_load";
    std::filesystem::remove_all(tmp_dir);
    nelf.save(tmp_dir.string());

    assert(std::filesystem::exists(tmp_dir / ".text.safetensors"));
    assert(std::filesystem::exists(tmp_dir / ".rodata.safetensors"));
    assert(std::filesystem::exists(tmp_dir / ".field.safetensors"));
    assert(std::filesystem::exists(tmp_dir / ".residual.safetensors"));
    assert(std::filesystem::exists(tmp_dir / "passthrough.safetensors"));
    assert(std::filesystem::exists(tmp_dir / "atlas.json"));
    assert(std::filesystem::exists(tmp_dir / "config.json"));
    assert(std::filesystem::exists(tmp_dir / "manifest.json"));

    const NeuralELF loaded = NeuralELF::load(tmp_dir.string());
    assert(loaded.config.arch_name == "qwen3-mini");
    assert(loaded.config.hidden_size == 32);
    assert(loaded.text_weights.size() == nelf.text_weights.size());
    assert(loaded.residual_weights.size() == nelf.residual_weights.size());
    assert(loaded.passthrough_vectors.size() == nelf.passthrough_vectors.size());

    const TransformerModel loaded_recon = loaded.reconstruct_model();
    assert(max_model_frob_diff(model, loaded_recon) < 1e-5);

    std::filesystem::remove_all(tmp_dir);
    std::cout << "[PASS] test_nelf_save_load_and_reconstruct\n";
}

void test_lift_model_and_zero_diff_roundtrip() {
    for (const auto& cfg : {make_qwen3_mini(), make_llama_mini(), make_mla_mini()}) {
        TransformerModel model = TransformerModel::create_synthetic(cfg, 77);
        auto [nelf, strata_src] = lift_model(model, "Lifted_" + cfg.arch_name);

        const std::string slash2(2, '/');
        const std::string block_open = std::string("/") + "*";
        assert(strata_src.find(slash2) == std::string::npos);
        assert(strata_src.find(block_open) == std::string::npos);

        const auto ratios = nelf.section_energy_ratios();
        const double r_text = ratios.at(".text");
        const double r_rodata = ratios.at(".rodata");
        const double r_field = ratios.at(".field");
        const double r_res = ratios.at(".residual");
        const double r_sum = r_text + r_rodata + r_field + r_res;

        assert(std::abs(r_sum - 1.0) < 1e-9);
        assert(r_res >= 0.02 && r_res <= 0.05);
        assert(r_text >= 0.50 && r_text <= 0.78);
        assert(r_rodata >= 0.08 && r_rodata <= 0.25);
        assert(r_field >= 0.08 && r_field <= 0.25);

        for (const auto& [k, w_res] : nelf.residual_weights) {
            const Matrix sum = mat_add(
                nelf.text_weights.at(k),
                mat_add(nelf.rodata_weights.at(k), mat_add(nelf.field_weights.at(k), w_res)));
            assert(frob_norm(mat_sub(model.matrices.at(k), sum)) < 1e-12);
        }

        const MIRModule mir = compile_source_to_mir(strata_src);
        assert(!mir.declarations.empty());
        assert(!mir.flows.empty());

        const TransformerModel restored = recompile_nelf_with_patch(nelf, strata_src, "in-place");
        assert(max_model_frob_diff(model, restored) < 1e-12);
        assert(max_logits_diff(model, restored, {1, 5, 11, 19}) < 1e-12);
    }
    std::cout << "[PASS] test_lift_model_and_zero_diff_roundtrip\n";
}

void test_differential_edit_isolation() {
    TransformerModel model = TransformerModel::create_synthetic(make_qwen3_mini(), 99);
    auto [nelf, base_src] = lift_model(model, "DiffIsoModel");

    {
        const std::string edited_rodata = insert_into_lifted_flow(
            base_src,
            "    rewrite LiftedRodata(key: \"anchor_0\", old: \"val_0\", new: \"val_1\");\n");

        const TransformerModel patched = recompile_nelf_with_patch(nelf, edited_rodata, "in-place");

        int changed_layers = 0;
        for (int l = 0; l < model.config.num_layers; ++l) {
            const std::string prefix = "model.layers." + std::to_string(l);
            assert(frob_norm(mat_sub(
                patched.matrices.at(prefix + ".self_attn.q_proj.weight"),
                model.matrices.at(prefix + ".self_attn.q_proj.weight"))) < 1e-12);
            assert(frob_norm(mat_sub(
                patched.matrices.at(prefix + ".self_attn.o_proj.weight"),
                model.matrices.at(prefix + ".self_attn.o_proj.weight"))) < 1e-12);
            assert(frob_norm(mat_sub(
                patched.matrices.at(prefix + ".mlp.gate_proj.weight"),
                model.matrices.at(prefix + ".mlp.gate_proj.weight"))) < 1e-12);
            assert(frob_norm(mat_sub(
                patched.matrices.at(prefix + ".mlp.up_proj.weight"),
                model.matrices.at(prefix + ".mlp.up_proj.weight"))) < 1e-12);

            const Matrix delta_down = mat_sub(
                patched.matrices.at(prefix + ".mlp.down_proj.weight"),
                model.matrices.at(prefix + ".mlp.down_proj.weight"));
            if (frob_norm(delta_down) > 1e-9) {
                ++changed_layers;
                for (const Vector& r_dir : nelf.atlas.preserve_directions.at("Residual")) {
                    const Vector res_proj = mat_vec(transpose(delta_down), r_dir);
                    assert(norm2(res_proj) / frob_norm(delta_down) < 1e-8);
                }
            }
        }
        assert(changed_layers == 1);
        assert(frob_norm(mat_sub(
            patched.matrices.at("model.embed_tokens.weight"),
            model.matrices.at("model.embed_tokens.weight"))) < 1e-12);
        assert(frob_norm(mat_sub(
            patched.matrices.at("lm_head.weight"),
            model.matrices.at("lm_head.weight"))) < 1e-12);
    }

    {
        const std::string edited_text = insert_into_lifted_flow(
            base_src,
            "    when state ~> LiftedDipole.+pole(threshold: 0.65) {\n"
            "        state -| LiftedDipole.+pole(scale: 0.5);\n"
            "        state +> LiftedDipole.-pole(gain: 0.25);\n"
            "    }\n");

        const TransformerModel patched = recompile_nelf_with_patch(nelf, edited_text, "in-place");

        int touched_mlp_layers = 0;
        for (int l = 0; l < model.config.num_layers; ++l) {
            const std::string prefix = "model.layers." + std::to_string(l);
            const Matrix delta_down = mat_sub(
                patched.matrices.at(prefix + ".mlp.down_proj.weight"),
                model.matrices.at(prefix + ".mlp.down_proj.weight"));
            if (frob_norm(delta_down) > 1e-9) {
                ++touched_mlp_layers;
                for (const Vector& r_dir : nelf.atlas.preserve_directions.at("Residual")) {
                    const Vector res_proj = mat_vec(transpose(delta_down), r_dir);
                    assert(norm2(res_proj) / frob_norm(delta_down) < 1e-8);
                }
            }
        }
        assert(touched_mlp_layers >= 1);
    }

    std::cout << "[PASS] test_differential_edit_isolation\n";
}

void test_static_and_dynamic_symbol_resolution() {
    const std::string src_04 = read_file_str("examples/04_rodata_and_transplant.strata");
    const MIRModule mir_04 = compile_source_to_mir(src_04);

    TransformerModel llama = TransformerModel::create_synthetic(make_llama_mini(), 123);
    NeuralAtlas prebuilt = build_default_atlas(llama, &mir_04, 123);
    const Vector custom_dir = normalize(Vector(24, 1.0));
    prebuilt.symbols["FactConsistencyCheck"].direction = custom_dir;

    const NeuralAtlas static_res = resolve_symbols_for_target(mir_04, llama, &prebuilt, 0);
    assert(norm2(vec_sub(static_res.symbols.at("FactConsistencyCheck").direction, custom_dir)) < 1e-12);

    const NeuralAtlas dyn_res = resolve_symbols_for_target(mir_04, llama, nullptr, 7);
    assert(dyn_res.hidden_size == 24);
    assert(dyn_res.num_layers == 4);
    assert(dyn_res.symbols.contains("FactConsistencyCheck"));
    assert(dyn_res.symbols.contains("CapitalKnowledge"));
    const auto& fcc = dyn_res.symbols.at("FactConsistencyCheck");
    assert(fcc.subspace_basis.size() == 4);
    for (std::size_t i = 0; i < fcc.subspace_basis.size(); ++i) {
        assert(std::abs(norm2(fcc.subspace_basis[i]) - 1.0) < 1e-9);
        for (std::size_t j = i + 1; j < fcc.subspace_basis.size(); ++j) {
            assert(std::abs(dot(fcc.subspace_basis[i], fcc.subspace_basis[j])) < 1e-9);
        }
    }

    const std::string src_02 = read_file_str("examples/02_cyclic_weekday_s1.strata");
    const MIRModule mir_02 = compile_source_to_mir(src_02);
    const NeuralAtlas dyn_02 = resolve_symbols_for_target(mir_02, llama, nullptr, 11);
    assert(dyn_02.symbols.contains("Weekday"));
    const auto& wd = dyn_02.symbols.at("Weekday");
    assert(wd.manifold_plane.size() == 3);
    for (std::size_t i = 0; i < 3; ++i) {
        assert(std::abs(norm2(wd.manifold_plane[i]) - 1.0) < 1e-9);
        for (std::size_t j = i + 1; j < 3; ++j) {
            assert(std::abs(dot(wd.manifold_plane[i], wd.manifold_plane[j])) < 1e-9);
        }
    }

    std::cout << "[PASS] test_static_and_dynamic_symbol_resolution\n";
}

void test_cross_compile_across_models() {
    TransformerModel qwen = TransformerModel::create_synthetic(make_qwen3_mini(), 42);
    TransformerModel llama = TransformerModel::create_synthetic(make_llama_mini(), 99);
    TransformerModel mla = TransformerModel::create_synthetic(make_mla_mini(), 55);

    const std::string src_04 = read_file_str("examples/04_rodata_and_transplant.strata");
    const CompileOutput out_llama_04 = cross_compile(src_04, llama, nullptr, "in-place");
    assert(out_llama_04.model.config.num_layers == 4);
    assert(out_llama_04.model.config.hidden_size == 24);
    assert(out_llama_04.model.config.intermediate_size == 64);
    assert(out_llama_04.stats.instructions_applied >= 4);
    assert(out_llama_04.stats.max_crosstalk <= 1e-3 + 1e-9);
    auto [logits_04, tr_04] = out_llama_04.model.forward({1, 2, 3, 4}, {}, false);
    (void)tr_04;
    assert(logits_04.size() == 4);

    const CompileOutput out_mla_04 = cross_compile(src_04, mla, nullptr, "in-place");
    assert(out_mla_04.model.config.attention_type == "mla");
    assert(out_mla_04.stats.instructions_applied >= 4);

    const std::string src_02 = read_file_str("examples/02_cyclic_weekday_s1.strata");
    const CompileOutput out_qwen_02 = cross_compile(src_02, qwen, nullptr, "in-place");
    assert(out_qwen_02.stats.max_approx_error <= 1e-2 + 1e-9);
    assert(out_qwen_02.stats.instructions_applied >= 10);

    const std::string src_03 = read_file_str("examples/03_tacit_field_decouple.strata");
    const CompileOutput out_llama_03 = cross_compile(src_03, llama, nullptr, "in-place");
    assert(out_llama_03.stats.instructions_applied >= 1);
    assert(!out_llama_03.wisa.empty());
    assert(out_llama_03.wisa[0].kind == WISAKind::FIELD_SPECTRAL_OP);
    assert(out_llama_03.stats.max_crosstalk <= 2e-3 + 1e-9);

    const std::string src_01 = read_file_str("examples/01_math_loop_surgery.strata");
    bool threw_ood = false;
    try {
        (void)cross_compile(src_01, llama, nullptr, "in-place");
    } catch (const OutOfDepthError&) {
        threw_ood = true;
    }
    assert(threw_ood);

    const CompileOutput out_elastic_01 = cross_compile(src_01, llama, nullptr, "elastic");
    assert(out_elastic_01.stats.added_layers >= 1);
    assert(out_elastic_01.model.config.num_layers > 4);
    auto [logits_01, tr_01] = out_elastic_01.model.forward({2, 4, 6}, {}, false);
    (void)tr_01;
    assert(logits_01.size() == 3);

    std::cout << "[PASS] test_cross_compile_across_models\n";
}

void test_tokenizer_backed_cross_compile_rope_qknorm_tied() {
    ModelConfig q3_cfg;
    q3_cfg.num_layers = 4;
    q3_cfg.hidden_size = 32;
    q3_cfg.intermediate_size = 96;
    q3_cfg.num_attention_heads = 4;
    q3_cfg.num_key_value_heads = 2;
    q3_cfg.vocab_size = 272;
    q3_cfg.attention_type = "gqa";
    q3_cfg.arch_name = "qwen3-rope-mini";
    q3_cfg.model_type = "qwen3";
    q3_cfg.use_rope = true;
    q3_cfg.rope_theta = 10000.0;
    q3_cfg.qk_norm = true;
    q3_cfg.tie_word_embeddings = true;

    const TransformerModel q3_model = TransformerModel::create_synthetic(q3_cfg, 314);
    assert(!q3_model.matrices.contains("lm_head.weight"));
    assert(q3_model.vectors.contains("model.layers.0.self_attn.q_norm.weight"));
    assert(q3_model.vectors.contains("model.layers.0.self_attn.k_norm.weight"));

    const Tokenizer tok = Tokenizer::create_byte_level_demo(q3_cfg.vocab_size);
    const std::string src_04 = read_file_str("examples/04_rodata_and_transplant.strata");
    const MIRModule mir_04 = compile_source_to_mir(src_04);

    const NeuralAtlas resolved = resolve_symbols_for_target(mir_04, q3_model, nullptr, 0, &tok);
    assert(resolved.symbols.contains("FactConsistencyCheck"));
    assert(resolved.symbols.at("FactConsistencyCheck").subspace_basis.size() == 4);

    const CompileOutput out_04 = cross_compile(src_04, q3_model, nullptr, "in-place", &tok);
    assert(out_04.stats.instructions_applied >= 4);
    assert(out_04.stats.max_crosstalk <= 1e-3 + 1e-9);
    assert(out_04.model.config.use_rope);
    assert(out_04.model.config.qk_norm);
    assert(out_04.model.config.tie_word_embeddings);
    const std::vector<int> gen = out_04.model.generate_greedy(tok.encode("The capital of Aethelgard"), 4);
    assert(gen.size() >= 4);

    const std::string src_02 = read_file_str("examples/02_cyclic_weekday_s1.strata");
    const CompileOutput out_02 = cross_compile(src_02, q3_model, nullptr, "in-place", &tok);
    assert(out_02.stats.max_approx_error <= 1e-2 + 1e-9);
    assert(out_02.stats.instructions_applied >= 10);

    ModelConfig small_cfg = make_qwen3_mini();
    small_cfg.num_layers = 4;
    small_cfg.vocab_size = 64;
    small_cfg.use_rope = true;
    small_cfg.qk_norm = true;
    small_cfg.tie_word_embeddings = true;
    const TransformerModel small_vocab_model = TransformerModel::create_synthetic(small_cfg, 271);
    const Tokenizer tok_256 = Tokenizer::create_byte_level_demo(256);
    const CompileOutput out_clamped = cross_compile(src_04, small_vocab_model, nullptr, "in-place", &tok_256);
    assert(out_clamped.stats.instructions_applied >= 4);

    std::cout << "[PASS] test_tokenizer_backed_cross_compile_rope_qknorm_tied\n";
}

void test_layer_filtered_lift_model() {
    ModelConfig cfg = make_qwen3_mini();
    cfg.model_type = "qwen3";
    cfg.use_rope = true;
    cfg.qk_norm = true;
    cfg.tie_word_embeddings = true;

    const TransformerModel model = TransformerModel::create_synthetic(cfg, 808);
    assert(!model.matrices.contains("lm_head.weight"));

    auto [nelf_sub, strata_sub] = lift_model(model, "SelectiveLifted", {2, 5});

    const std::string slash2(2, '/');
    const std::string block_open = std::string("/") + "*";
    assert(strata_sub.find(slash2) == std::string::npos);
    assert(strata_sub.find(block_open) == std::string::npos);

    const MIRModule mir_sub = compile_source_to_mir(strata_sub);
    assert(!mir_sub.declarations.empty());
    assert(!mir_sub.flows.empty());

    for (int l = 0; l < cfg.num_layers; ++l) {
        const std::string qk = "model.layers." + std::to_string(l) + ".self_attn.q_proj.weight";
        const std::string dk = "model.layers." + std::to_string(l) + ".mlp.down_proj.weight";
        if (l >= 2 && l < 5) {
            assert(nelf_sub.text_weights.contains(qk));
            assert(nelf_sub.text_weights.contains(dk));
            assert(nelf_sub.residual_weights.contains(dk));
            assert(!nelf_sub.passthrough_matrices.contains(qk));
        } else {
            assert(!nelf_sub.text_weights.contains(qk));
            assert(!nelf_sub.rodata_weights.contains(qk));
            assert(!nelf_sub.field_weights.contains(qk));
            assert(!nelf_sub.residual_weights.contains(qk));
            assert(nelf_sub.passthrough_matrices.contains(qk));
            assert(nelf_sub.passthrough_matrices.contains(dk));
        }
    }
    assert(nelf_sub.passthrough_matrices.contains("model.embed_tokens.weight"));
    assert(nelf_sub.passthrough_vectors.contains("model.layers.0.self_attn.q_norm.weight"));
    assert(nelf_sub.passthrough_vectors.contains("model.layers.3.self_attn.k_norm.weight"));

    const auto ratios = nelf_sub.section_energy_ratios();
    assert(ratios.at(".residual") >= 0.02 && ratios.at(".residual") <= 0.05);

    const TransformerModel recon = nelf_sub.reconstruct_model();
    assert(max_model_frob_diff(model, recon) < 1e-12);

    const TransformerModel restored = recompile_nelf_with_patch(nelf_sub, strata_sub, "in-place");
    assert(max_model_frob_diff(model, restored) < 1e-12);

    const std::string edited_sub = insert_into_lifted_flow(
        strata_sub,
        "    rewrite LiftedRodata(key: \"anchor_0\", old: \"val_0\", new: \"val_2\");\n");
    const TransformerModel patched_sub = recompile_nelf_with_patch(nelf_sub, edited_sub, "in-place");

    int changed_in_filter = 0;
    for (int l = 0; l < cfg.num_layers; ++l) {
        const std::string dk = "model.layers." + std::to_string(l) + ".mlp.down_proj.weight";
        const double diff = frob_norm(mat_sub(patched_sub.matrices.at(dk), model.matrices.at(dk)));
        if (l >= 2 && l < 5) {
            if (diff > 1e-9) {
                ++changed_in_filter;
            }
        } else {
            assert(diff < 1e-12);
        }
    }
    assert(changed_in_filter == 1);

    const std::filesystem::path tmp_dir =
        std::filesystem::temp_directory_path() / "aic_test_nelf_filtered";
    std::filesystem::remove_all(tmp_dir);
    nelf_sub.save(tmp_dir.string());
    const NeuralELF reloaded = NeuralELF::load(tmp_dir.string());
    assert(max_model_frob_diff(model, reloaded.reconstruct_model()) < 1e-5);
    std::filesystem::remove_all(tmp_dir);

    std::cout << "[PASS] test_layer_filtered_lift_model\n";
}

void test_streaming_lift_sharded_bf16() {
    ModelConfig cfg = make_qwen3_mini();
    cfg.model_type = "qwen3";
    cfg.use_rope = true;
    cfg.qk_norm = true;
    cfg.torch_dtype = "BF16";
    const TransformerModel model = TransformerModel::create_synthetic(cfg, 404);

    const std::filesystem::path ckpt_dir =
        std::filesystem::temp_directory_path() / "aic_test_stream_lift_bf16";
    std::filesystem::remove_all(ckpt_dir);
    std::filesystem::create_directories(ckpt_dir);

    std::map<std::string, TensorSpec> specs;
    std::size_t total_bytes = 0;
    for (const auto& [k, m] : model.matrices) {
        TensorSpec sp = TensorSpec::from_matrix(m, "BF16");
        total_bytes += sp.data.size() * 2u;
        specs[k] = std::move(sp);
    }
    for (const auto& [k, v] : model.vectors) {
        TensorSpec sp = TensorSpec::from_vector(v, "BF16");
        total_bytes += sp.data.size() * 2u;
        specs[k] = std::move(sp);
    }

    save_safetensors_sharded(
        ckpt_dir.string(),
        specs,
        std::max<std::size_t>(1, total_bytes / 2u),
        {{"arch_name", cfg.arch_name}});
    {
        std::ofstream ofs((ckpt_dir / "config.json").string(), std::ios::trunc);
        assert(ofs.good());
        ofs << dump_json(cfg.to_json(), 2) << '\n';
    }
    assert(std::filesystem::exists(ckpt_dir / "model.safetensors.index.json"));

    auto [nelf_sub, strata_sub] = lift_checkpoint_streaming(ckpt_dir.string(), "StreamSub", {2, 5});

    const std::string slash2(2, '/');
    const std::string block_open = std::string("/") + "*";
    assert(strata_sub.find(slash2) == std::string::npos);
    assert(strata_sub.find(block_open) == std::string::npos);

    const MIRModule mir_sub = compile_source_to_mir(strata_sub);
    assert(!mir_sub.declarations.empty());
    assert(!mir_sub.flows.empty());

    for (int l = 0; l < cfg.num_layers; ++l) {
        const std::string qk = "model.layers." + std::to_string(l) + ".self_attn.q_proj.weight";
        const std::string dk = "model.layers." + std::to_string(l) + ".mlp.down_proj.weight";
        if (l >= 2 && l < 5) {
            assert(nelf_sub.text_weights.contains(qk));
            assert(nelf_sub.text_weights.contains(dk));
            assert(nelf_sub.residual_weights.contains(dk));
        } else {
            assert(!nelf_sub.text_weights.contains(qk));
            assert(!nelf_sub.rodata_weights.contains(qk));
            assert(!nelf_sub.field_weights.contains(qk));
            assert(!nelf_sub.residual_weights.contains(qk));
            assert(!nelf_sub.passthrough_matrices.contains(qk));
        }
    }

    const auto ratios = nelf_sub.section_energy_ratios();
    assert(ratios.at(".residual") >= 0.02 && ratios.at(".residual") <= 0.05);

    const TransformerModel recon_sub = nelf_sub.reconstruct_model();
    for (int l = 2; l < 5; ++l) {
        const std::string dk = "model.layers." + std::to_string(l) + ".mlp.down_proj.weight";
        const Matrix bf16_ref = load_single_tensor_sharded(ckpt_dir.string(), dk).to_matrix();
        assert(frob_norm(mat_sub(recon_sub.matrices.at(dk), bf16_ref)) < 1e-12);
    }

    auto [nelf_full, strata_full] = lift_checkpoint_streaming(ckpt_dir.string(), "StreamFull", {-1, -1});
    assert(strata_full.find(slash2) == std::string::npos);
    assert(strata_full.find(block_open) == std::string::npos);
    const TransformerModel loaded_full = TransformerModel::load_checkpoint(ckpt_dir.string());
    assert(max_model_frob_diff(loaded_full, nelf_full.reconstruct_model()) < 1e-12);

    std::filesystem::remove_all(ckpt_dir);
    std::cout << "[PASS] test_streaming_lift_sharded_bf16\n";
}

void test_auto_s1_ring_discovery() {
    constexpr double kPi = 3.14159265358979323846;
    ModelConfig cfg = make_qwen3_mini();
    TransformerModel ring_model = TransformerModel::create_synthetic(cfg, 515);

    const std::size_t d = static_cast<std::size_t>(cfg.hidden_size);
    for (int l = 0; l < 3; ++l) {
        const std::string prefix = "model.layers." + std::to_string(l);
        ring_model.matrices[prefix + ".self_attn.o_proj.weight"] =
            zeros(d, static_cast<std::size_t>(cfg.hidden_size));
        ring_model.matrices[prefix + ".mlp.down_proj.weight"] =
            zeros(d, static_cast<std::size_t>(cfg.intermediate_size));
    }

    Vector e0(d, 0.0);
    Vector e1(d, 0.0);
    Vector e2(d, 0.0);
    e0[0] = 1.0;
    e1[1] = 1.0;
    e2[2] = 1.0;
    Matrix& emb = ring_model.matrices["model.embed_tokens.weight"];
    for (std::size_t tid = 0; tid < emb.rows; ++tid) {
        const double theta = (2.0 * kPi * static_cast<double>(tid % 16)) / 16.0;
        Vector row = vec_add(
            vec_scale(e0, 1.5),
            vec_add(vec_scale(e1, 2.0 * std::cos(theta)), vec_scale(e2, 2.0 * std::sin(theta))));
        emb.set_row(tid, row);
    }

    auto [nelf_ring, strata_ring] = lift_model(ring_model, "RingDiscovered", {2, 3});
    const std::string slash2(2, '/');
    const std::string block_open = std::string("/") + "*";
    assert(strata_ring.find(slash2) == std::string::npos);
    assert(strata_ring.find(block_open) == std::string::npos);
    assert(strata_ring.find("manifold RingManifold_L2 : S1 in Frame<2..3>") != std::string::npos);

    const MIRModule mir_ring = compile_source_to_mir(strata_ring);
    assert(mir_ring.declarations.contains("RingManifold_L2"));
    assert(mir_ring.declarations.at("RingManifold_L2").topology == "S1");
    assert(nelf_ring.atlas.symbols.contains("RingManifold_L2"));
    const auto& plane = nelf_ring.atlas.symbols.at("RingManifold_L2").manifold_plane;
    assert(plane.size() == 3);
    for (std::size_t i = 0; i < 3; ++i) {
        assert(std::abs(norm2(plane[i]) - 1.0) < 1e-9);
        for (std::size_t j = i + 1; j < 3; ++j) {
            assert(std::abs(dot(plane[i], plane[j])) < 1e-9);
        }
    }

    const std::string edited_ring = insert_into_lifted_flow(
        strata_ring,
        "    when state ~> RingManifold_L2(chart: \"angle\") as theta {\n"
        "        let rot = state => RingManifold_L2.rotate(angle: theta, delta: 0.5);\n"
        "        state -| RingManifold_L2;\n"
        "        state +> rot(gain: 1.0);\n"
        "    }\n");
    const TransformerModel patched_ring = recompile_nelf_with_patch(nelf_ring, edited_ring, "elastic");
    assert(patched_ring.config.intermediate_size > ring_model.config.intermediate_size);

    for (std::uint64_t seed : {11ULL, 42ULL, 99ULL, 123ULL}) {
        const TransformerModel plain = TransformerModel::create_synthetic(make_qwen3_mini(), seed);
        auto [plain_nelf, plain_strata] = lift_model(plain, "Plain_" + std::to_string(seed));
        (void)plain_nelf;
        assert(plain_strata.find("RingManifold_") == std::string::npos);
    }

    std::cout << "[PASS] test_auto_s1_ring_discovery\n";
}

void test_verify_nelf_faithfulness() {
    ModelConfig cfg = make_qwen3_mini();
    cfg.model_type = "qwen3";
    cfg.use_rope = true;
    cfg.qk_norm = true;
    const TransformerModel model = TransformerModel::create_synthetic(cfg, 606);

    auto [nelf, strata_src] = lift_model(model, "FaithfulModel");
    const TransformerModel recompiled = recompile_nelf_with_patch(nelf, strata_src, "in-place");

    const FaithfulnessReport rep_zero = verify_nelf_faithfulness(model, recompiled, &nelf);
    assert(rep_zero.weight_frob_diff < 1e-12);
    assert(rep_zero.mean_kl_div < 1e-12);
    assert(rep_zero.max_kl_div < 1e-12);
    assert(rep_zero.max_logit_diff < 1e-12);
    assert(rep_zero.residual_energy_ratio >= 0.02 && rep_zero.residual_energy_ratio <= 0.05);

    const std::string edited = insert_into_lifted_flow(
        strata_src,
        "    rewrite LiftedRodata(key: \"anchor_0\", old: \"val_0\", new: \"val_3\");\n");
    const TransformerModel patched = recompile_nelf_with_patch(nelf, edited, "in-place");
    const FaithfulnessReport rep_patch = verify_nelf_faithfulness(model, patched, &nelf);
    assert(rep_patch.weight_frob_diff > 1e-6);
    assert(std::isfinite(rep_patch.mean_kl_div) && rep_patch.mean_kl_div >= 0.0);

    std::cout << "[PASS] test_verify_nelf_faithfulness\n";
}

void test_tracer_before_and_after_pole_patch() {
    ModelConfig cfg = make_qwen3_mini();
    cfg.num_layers = 4;
    cfg.vocab_size = 272;
    cfg.use_rope = true;
    cfg.qk_norm = true;
    const TransformerModel base_model = TransformerModel::create_synthetic(cfg, 707);
    const Tokenizer tok = Tokenizer::create_byte_level_demo(cfg.vocab_size);

    const std::string patch_src =
        "module TraceLoopPatch;\n"
        "dipole LoopAnomaly in Frame<Mid> {\n"
        "    +pole: Spiral {\n"
        "        anchor contrastive(\n"
        "            pos: [\"Wait, let me double check. Wait, let me double check.\"],\n"
        "            neg: [\"Therefore, x = 5.\"]\n"
        "        );\n"
        "    }\n"
        "    -pole: Progress {\n"
        "        anchor probe(\"Step 2:\");\n"
        "    }\n"
        "}\n"
        "flow SuppressLoop(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>\n"
        "    ![crosstalk <= 1e-2, approx <= 0.0, energy <= 0.25, preserve(CoreFluency, Residual)]\n"
        "{\n"
        "    when state ~> LoopAnomaly.+pole(threshold: 0.5) {\n"
        "        state -| LoopAnomaly.+pole(scale: 1.5);\n"
        "        state +> LoopAnomaly.-pole(gain: 0.45);\n"
        "    }\n"
        "}\n";

    const MIRModule mir = compile_source_to_mir(patch_src);
    const NeuralAtlas atlas = resolve_symbols_for_target(mir, base_model, nullptr, 42, &tok);
    const CompileOutput patched = cross_compile(patch_src, base_model, &atlas, "in-place", &tok);

    const std::vector<int> prompt_ids = tok.encode("Wait, let me double check.");
    const ExecutionTraceReport rep_before = trace_model_execution(base_model, atlas, prompt_ids, &mir);
    const ExecutionTraceReport rep_after = trace_model_execution(patched.model, atlas, prompt_ids, &mir);

    assert(rep_before.layer_residual_rms.size() == static_cast<std::size_t>(cfg.num_layers + 1));
    assert(rep_after.layer_residual_rms.size() == static_cast<std::size_t>(cfg.num_layers + 1));
    assert(rep_before.predicted_next_token >= 0 && rep_before.predicted_next_token < cfg.vocab_size);
    assert(rep_after.predicted_next_token >= 0 && rep_after.predicted_next_token < cfg.vocab_size);

    auto find_act = [](const ExecutionTraceReport& rep, const std::string& sym) -> SymbolActivation {
        for (const auto& a : rep.activations) {
            if (a.symbol == sym) {
                return a;
            }
        }
        assert(false);
        return SymbolActivation{};
    };

    const SymbolActivation act_before = find_act(rep_before, "LoopAnomaly");
    const SymbolActivation act_after = find_act(rep_after, "LoopAnomaly");
    const double delta_mean = std::abs(act_after.mean_proj - act_before.mean_proj);
    const double delta_max = std::abs(act_after.max_proj - act_before.max_proj);
    assert(delta_mean > 1e-6 || delta_max > 1e-6);

    const std::string formatted = format_execution_trace(rep_after, &tok);
    assert(formatted.find("ExecutionTrace") != std::string::npos);
    assert(formatted.find("Layer Residual RMS:") != std::string::npos);
    assert(formatted.find("LoopAnomaly") != std::string::npos);

    std::cout << "[PASS] test_tracer_before_and_after_pole_patch\n";
}

void test_diff_nelf_identical_and_text_edit() {
    const TransformerModel model = TransformerModel::create_synthetic(make_qwen3_mini(), 909);
    auto [nelf_a, strata_a] = lift_model(model, "DiffNelfBase");
    (void)strata_a;

    const NelfDiffReport same_rep = diff_nelf(nelf_a, nelf_a);
    assert(same_rep.text_frob_diff == 0.0);
    assert(same_rep.rodata_frob_diff == 0.0);
    assert(same_rep.field_frob_diff == 0.0);
    assert(same_rep.residual_frob_diff == 0.0);
    assert(same_rep.changed_symbols == 0);

    NeuralELF nelf_b = nelf_a;
    Matrix& tw = nelf_b.text_weights.at("model.layers.2.mlp.down_proj.weight");
    tw(0, 0) += 0.25;
    nelf_b.atlas.symbols["LiftedDipole"].direction[0] += 0.1;

    const NelfDiffReport edit_rep = diff_nelf(nelf_a, nelf_b);
    assert(std::abs(edit_rep.text_frob_diff - 0.25) < 1e-12);
    assert(edit_rep.rodata_frob_diff == 0.0);
    assert(edit_rep.field_frob_diff == 0.0);
    assert(edit_rep.residual_frob_diff == 0.0);
    assert(edit_rep.changed_symbols == 1);

    const std::string formatted = format_nelf_diff(edit_rep);
    assert(formatted.find("text_frob_diff:") != std::string::npos);
    assert(formatted.find("changed_symbols: 1") != std::string::npos);

    std::cout << "[PASS] test_diff_nelf_identical_and_text_edit\n";
}

void test_trace_generation_and_compare() {
    ModelConfig cfg = make_qwen3_mini();
    cfg.num_layers = 4;
    cfg.vocab_size = 272;
    cfg.use_rope = true;
    cfg.qk_norm = true;
    const TransformerModel base_model = TransformerModel::create_synthetic(cfg, 707);
    const Tokenizer tok = Tokenizer::create_byte_level_demo(cfg.vocab_size);

    const std::string patch_src =
        "module TraceGenModule;\n"
        "dipole LoopAnomaly in Frame<Mid> {\n"
        "    +pole: Spiral {\n"
        "        anchor contrastive(\n"
        "            pos: [\"Wait, let me double check. Wait, let me double check.\"],\n"
        "            neg: [\"Therefore, x = 5.\"]\n"
        "        );\n"
        "    }\n"
        "    -pole: Progress {\n"
        "        anchor probe(\"Step 2:\");\n"
        "    }\n"
        "}\n"
        "flow SuppressLoop(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>\n"
        "    ![crosstalk <= 1e-2, approx <= 0.1, energy <= 0.25, preserve(CoreFluency, Residual)]\n"
        "{\n"
        "    state -| LoopAnomaly.+pole(scale: 1.5);\n"
        "    state +> LoopAnomaly.-pole(gain: 0.45);\n"
        "}\n";

    const MIRModule mir = compile_source_to_mir(patch_src);
    const NeuralAtlas atlas = resolve_symbols_for_target(mir, base_model, nullptr, 42, &tok);
    const CompileOutput patched = cross_compile(patch_src, base_model, &atlas, "in-place", &tok);

    const std::vector<int> prompt_ids = tok.encode("Wait, let me double check.");
    const GenerationTraceReport gen_rep = trace_generation(base_model, atlas, prompt_ids, 4, &mir);

    assert(gen_rep.prompt_ids == prompt_ids);
    assert(gen_rep.generated_ids.size() == 4);
    assert(gen_rep.decode_steps.size() == 4);
    assert(gen_rep.prefill_trace.token_ids == prompt_ids);

    const std::vector<int> greedy_ids = base_model.generate_greedy(prompt_ids, 4);
    assert(greedy_ids.size() == prompt_ids.size() + 4);
    for (int s = 0; s < 4; ++s) {
        const std::size_t idx = static_cast<std::size_t>(s);
        assert(gen_rep.generated_ids[idx] == greedy_ids[prompt_ids.size() + idx]);
        assert(gen_rep.decode_steps[idx].step == s);
        assert(gen_rep.decode_steps[idx].token_id == gen_rep.generated_ids[idx]);
        assert(!gen_rep.decode_steps[idx].activations.empty());
        for (const auto& act : gen_rep.decode_steps[idx].activations) {
            assert(std::isfinite(act.mean_proj));
            assert(std::isfinite(act.max_proj));
            assert(std::isfinite(act.active_ratio));
        }
    }

    const GenerationTraceReport zero_gen = trace_generation(base_model, atlas, prompt_ids, 0, &mir);
    assert(zero_gen.generated_ids.empty());
    assert(zero_gen.decode_steps.empty());

    const std::string gen_fmt = format_generation_trace(gen_rep, &tok);
    assert(gen_fmt.find("GenerationTrace") != std::string::npos);
    assert(gen_fmt.find("Prompt:") != std::string::npos);
    assert(gen_fmt.find("Generated:") != std::string::npos);
    assert(gen_fmt.find("Decode Steps (4):") != std::string::npos);
    assert(gen_fmt.find("LoopAnomaly") != std::string::npos);

    const TraceComparisonReport cmp = compare_execution_traces(
        base_model, patched.model, atlas, prompt_ids, &mir);
    assert(cmp.token_ids == prompt_ids);
    assert(cmp.base_next_token >= 0 && cmp.base_next_token < cfg.vocab_size);
    assert(cmp.patched_next_token >= 0 && cmp.patched_next_token < cfg.vocab_size);
    const double expected_kl = compute_model_divergence(base_model, patched.model, {prompt_ids}).mean_kl;
    assert(std::abs(cmp.kl_divergence - expected_kl) < 1e-12);
    assert(!cmp.deltas.empty());

    bool found_loop = false;
    for (const auto& d : cmp.deltas) {
        assert(std::abs(d.delta_mean_proj - (d.patched_mean_proj - d.base_mean_proj)) < 1e-12);
        assert(std::abs(d.delta_max_proj - (d.patched_max_proj - d.base_max_proj)) < 1e-12);
        if (d.symbol == "LoopAnomaly") {
            found_loop = true;
            assert(std::abs(d.delta_mean_proj) > 1e-6 || std::abs(d.delta_max_proj) > 1e-6);
        }
    }
    assert(found_loop);

    const TraceComparisonReport same_cmp = compare_execution_traces(
        base_model, base_model, atlas, prompt_ids, &mir);
    assert(same_cmp.kl_divergence == 0.0);
    for (const auto& d : same_cmp.deltas) {
        assert(std::abs(d.delta_mean_proj) < 1e-12);
        assert(std::abs(d.delta_max_proj) < 1e-12);
    }

    const std::string cmp_fmt = format_trace_comparison(cmp, &tok);
    assert(cmp_fmt.find("TraceComparison") != std::string::npos);
    assert(cmp_fmt.find("kl_divergence=") != std::string::npos);
    assert(cmp_fmt.find("Symbol Deltas") != std::string::npos);
    assert(cmp_fmt.find("LoopAnomaly") != std::string::npos);

    std::cout << "[PASS] test_trace_generation_and_compare\n";
}

void test_run_strata_tests_suite() {
    ModelConfig cfg = make_qwen3_mini();
    cfg.num_layers = 4;
    cfg.vocab_size = 272;
    cfg.use_rope = true;
    cfg.qk_norm = true;
    const TransformerModel base_model = TransformerModel::create_synthetic(cfg, 808);
    const Tokenizer tok = Tokenizer::create_byte_level_demo(cfg.vocab_size);

    const std::string src =
        "module BehavioralSuiteTest;\n"
        "const POS_PROMPT = \"Wait, let me double check. Wait, let me double check.\";\n"
        "dipole LoopAnomaly in Frame<Mid> {\n"
        "    +pole: Spiral {\n"
        "        anchor contrastive(\n"
        "            pos: [\"Wait, let me double check. Wait, let me double check.\"],\n"
        "            neg: [\"Therefore, x = 5.\"]\n"
        "        );\n"
        "    }\n"
        "    -pole: Progress {\n"
        "        anchor probe(\"Step 2:\");\n"
        "    }\n"
        "}\n"
        "subspace GuardSub<2> in Frame<Mid> {\n"
        "    anchor probe(\"Hello world\");\n"
        "}\n"
        "flow SuppressLoop(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>\n"
        "    ![crosstalk <= 1e-2, approx <= 0.1, energy <= 0.25, preserve(Residual)]\n"
        "{\n"
        "    state -| LoopAnomaly.+pole(scale: 1.5);\n"
        "    state +> LoopAnomaly.-pole(gain: 0.45);\n"
        "}\n"
        "test PassContracts {\n"
        "    expect resonance(POS_PROMPT, LoopAnomaly.+pole) < base;\n"
        "    expect resonance(\"Step 2:\", LoopAnomaly.-pole) >= base - 10.0;\n"
        "    expect resonance(\"Hello world\", GuardSub) >= 0.0;\n"
        "    expect kl_drift([POS_PROMPT, \"Hello world\"]) < 5.0;\n"
        "    expect crosstalk <= 0.5;\n"
        "    expect approx <= 0.5;\n"
        "}\n"
        "test FailContracts {\n"
        "    expect resonance(POS_PROMPT, LoopAnomaly.+pole) > base + 100.0;\n"
        "    expect resonance(\"Step 2:\", LoopAnomaly.-pole) == -999.0;\n"
        "    expect kl_drift(\"Hello world\") < 0.0;\n"
        "    expect crosstalk < 0.0;\n"
        "    expect approx < 0.0;\n"
        "}\n";

    const MIRModule mir = compile_source_to_mir(src);
    const NeuralAtlas atlas = resolve_symbols_for_target(mir, base_model, nullptr, 42, &tok);
    const CompileOutput compiled = cross_compile(src, base_model, &atlas, "in-place", &tok);

    const StrataTestSuiteReport rep = run_strata_tests(mir, base_model, compiled, atlas, &tok);
    assert(rep.total_tests == 2);
    assert(rep.passed_tests == 1);
    assert(rep.total_assertions == 11);
    assert(rep.passed_assertions == 6);
    assert(rep.test_results.size() == 2);
    assert(rep.test_results[0].first == "PassContracts");
    assert(rep.test_results[0].second.size() == 6);
    for (const auto& ar : rep.test_results[0].second) {
        assert(ar.passed);
    }
    assert(rep.test_results[1].first == "FailContracts");
    assert(rep.test_results[1].second.size() == 5);
    for (const auto& ar : rep.test_results[1].second) {
        assert(!ar.passed);
    }

    const std::string rep_str = format_strata_test_report(rep);
    assert(rep_str.find("StrataTestSuite (1/2 tests passed, 6/11 assertions passed)") != std::string::npos);
    assert(rep_str.find("TEST PassContracts: PASS (6/6)") != std::string::npos);
    assert(rep_str.find("TEST FailContracts: FAIL (0/5)") != std::string::npos);
    assert(rep_str.find("[PASS]") != std::string::npos);
    assert(rep_str.find("[FAIL]") != std::string::npos);

    const NeuralAtlas empty_atlas;
    const StrataTestSuiteReport auto_rep = run_strata_tests(
        mir, base_model, compiled, empty_atlas, nullptr);
    assert(auto_rep.total_tests == 2);
    assert(auto_rep.total_assertions == 11);

    std::cout << "[PASS] test_run_strata_tests_suite\n";
}

void test_top_token_and_logit_margin_tests() {
    ModelConfig cfg;
    cfg.arch_name = "qwen3";
    cfg.num_layers = 8;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 96;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.vocab_size = 64;
    cfg.attention_type = "gqa";
    const TransformerModel base_model = TransformerModel::create_synthetic(cfg, 42);
    const Tokenizer tok = Tokenizer::create_byte_level_demo(cfg.vocab_size);

    const std::string prompt = "Aethelgard";
    const std::string target_str = "Newhaven";
    const std::string foil_str = "Oldhaven";

    const std::string src =
        "module TopTokenAndMarginTest;\n"
        "rodata Facts<Entity, City> in Frame<Mid> {\n"
        "    anchor factual(\"capital: target\");\n"
        "    entry \"" + prompt + "\" => \"" + foil_str + "\";\n"
        "}\n"
        "flow RewriteFact(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>\n"
        "    ![crosstalk <= 1e-3, approx <= 5e-2, energy <= 1.0, preserve(Residual)]\n"
        "{\n"
        "    rewrite Facts(key: \"" + prompt + "\", old: \"" + foil_str + "\", new: \"" + target_str + "\");\n"
        "}\n"
        "test VerifyFactAndMargin {\n"
        "    expect top_token(\"" + prompt + "\") == \"" + target_str + "\";\n"
        "    expect top_token(\"" + prompt + "\") != \"" + foil_str + "\";\n"
        "    expect logit_margin(\"" + prompt + "\", \"" + target_str + "\", \"" + foil_str + "\") > 0.0;\n"
        "    expect logit_margin(\"" + prompt + "\", \"" + target_str + "\", \"" + foil_str + "\") > base;\n"
        "}\n"
        "test FailFactAndMargin {\n"
        "    expect top_token(\"" + prompt + "\") == \"" + foil_str + "\";\n"
        "    expect top_token(\"" + prompt + "\") != \"" + target_str + "\";\n"
        "    expect logit_margin(\"" + prompt + "\", \"" + target_str + "\", \"" + foil_str + "\") < 0.0;\n"
        "    expect logit_margin(\"" + prompt + "\", \"" + target_str + "\", \"" + foil_str + "\") < base;\n"
        "}\n";

    const MIRModule mir = compile_source_to_mir(src);
    NeuralAtlas atlas = build_default_atlas(base_model, &mir, 42);
    calibrate_atlas_empirical(atlas, base_model, &mir, &tok);
    const CompileOutput compiled = compile_mir_to_model(mir, base_model, &atlas, "in-place", 2);

    const StrataTestSuiteReport rep = run_strata_tests(mir, base_model, compiled, atlas, &tok);
    assert(rep.total_tests == 2);
    assert(rep.passed_tests == 1);
    assert(rep.total_assertions == 8);
    assert(rep.passed_assertions == 4);
    for (const auto& ar : rep.test_results[0].second) {
        assert(ar.passed);
    }
    for (const auto& ar : rep.test_results[1].second) {
        assert(!ar.passed);
    }

    const std::string formatted = format_strata_test_report(rep);
    assert(formatted.find("top_token(") != std::string::npos);
    assert(formatted.find("logit_margin(") != std::string::npos);
    assert(formatted.find("TEST VerifyFactAndMargin: PASS (4/4)") != std::string::npos);
    assert(formatted.find("TEST FailFactAndMargin: FAIL (0/4)") != std::string::npos);

    std::cout << "[PASS] test_top_token_and_logit_margin_tests\n";
}

void test_auto_t2_torus_and_lifted_test_block() {
    constexpr double kPi = 3.14159265358979323846;
    ModelConfig cfg = make_qwen3_mini();
    TransformerModel torus_model = TransformerModel::create_synthetic(cfg, 626);

    const std::size_t d = static_cast<std::size_t>(cfg.hidden_size);
    for (int l = 0; l < 3; ++l) {
        const std::string prefix = "model.layers." + std::to_string(l);
        torus_model.matrices[prefix + ".self_attn.o_proj.weight"] =
            zeros(d, static_cast<std::size_t>(cfg.hidden_size));
        torus_model.matrices[prefix + ".mlp.down_proj.weight"] =
            zeros(d, static_cast<std::size_t>(cfg.intermediate_size));
    }

    Matrix& emb = torus_model.matrices["model.embed_tokens.weight"];
    for (std::size_t tid = 0; tid < emb.rows; ++tid) {
        const int iu = static_cast<int>(tid % 4);
        const int iv = static_cast<int>((tid / 4) % 4);
        const double theta_u = (2.0 * kPi * static_cast<double>(iu)) / 4.0;
        const double theta_v = (2.0 * kPi * static_cast<double>(iv)) / 4.0;
        Vector row(d, 0.0);
        row[9] = 1.5;
        row[10] = 2.0 * std::cos(theta_u);
        row[11] = 2.0 * std::sin(theta_u);
        row[12] = 1.4 * std::cos(theta_v);
        row[13] = 1.4 * std::sin(theta_v);
        emb.set_row(tid, row);
    }

    auto [nelf_torus, strata_torus] = lift_model(torus_model, "TorusDiscovered", {2, 3});
    const std::string slash2(2, '/');
    const std::string block_open = std::string("/") + "*";
    assert(strata_torus.find(slash2) == std::string::npos);
    assert(strata_torus.find(block_open) == std::string::npos);
    assert(strata_torus.find("manifold TorusManifold_L2 : T2 in Frame<2..3>") != std::string::npos);
    assert(strata_torus.find("test VerifyLiftedFaithfulness") != std::string::npos);

    const MIRModule mir_torus = compile_source_to_mir(strata_torus);
    assert(mir_torus.declarations.contains("TorusManifold_L2"));
    assert(mir_torus.declarations.at("TorusManifold_L2").topology == "T2");
    assert(mir_torus.tests.size() == 1);
    assert(mir_torus.tests[0].name == "VerifyLiftedFaithfulness");
    assert(nelf_torus.atlas.symbols.contains("TorusManifold_L2"));
    const auto& plane = nelf_torus.atlas.symbols.at("TorusManifold_L2").manifold_plane;
    assert(plane.size() == 5);
    for (std::size_t i = 0; i < 5; ++i) {
        assert(std::abs(norm2(plane[i]) - 1.0) < 1e-9);
        for (std::size_t j = i + 1; j < 5; ++j) {
            assert(std::abs(dot(plane[i], plane[j])) < 1e-9);
        }
    }

    for (std::uint64_t seed : {11ULL, 42ULL, 99ULL, 123ULL}) {
        const TransformerModel plain = TransformerModel::create_synthetic(make_qwen3_mini(), seed);
        auto [plain_nelf, plain_strata] = lift_model(plain, "PlainTorus_" + std::to_string(seed));
        assert(plain_strata.find("TorusManifold_") == std::string::npos);
        assert(plain_strata.find("test VerifyLiftedFaithfulness") != std::string::npos);
        assert(plain_strata.find(slash2) == std::string::npos);
        assert(plain_strata.find(block_open) == std::string::npos);

        const TransformerModel restored = recompile_nelf_with_patch(plain_nelf, plain_strata, "in-place");
        assert(max_model_frob_diff(plain, restored) < 1e-12);

        const MIRModule plain_mir = compile_source_to_mir(plain_strata);
        const CompileOutput zero_compiled = compile_mir_to_model(
            plain_mir, plain, &plain_nelf.atlas, "in-place");
        const StrataTestSuiteReport suite = run_strata_tests(
            plain_mir, plain, zero_compiled, plain_nelf.atlas);
        assert(suite.total_tests == 1);
        assert(suite.passed_tests == 1);
        assert(suite.total_assertions == 2);
        assert(suite.passed_assertions == 2);
    }

    std::cout << "[PASS] test_auto_t2_torus_and_lifted_test_block\n";
}

}

int main() {
    test_nelf_save_load_and_reconstruct();
    test_lift_model_and_zero_diff_roundtrip();
    test_differential_edit_isolation();
    test_static_and_dynamic_symbol_resolution();
    test_cross_compile_across_models();
    test_tokenizer_backed_cross_compile_rope_qknorm_tied();
    test_layer_filtered_lift_model();
    test_streaming_lift_sharded_bf16();
    test_auto_s1_ring_discovery();
    test_verify_nelf_faithfulness();
    test_tracer_before_and_after_pole_patch();
    test_diff_nelf_identical_and_text_edit();
    test_trace_generation_and_compare();
    test_run_strata_tests_suite();
    test_top_token_and_logit_margin_tests();
    test_auto_t2_torus_and_lifted_test_block();
    std::cout << "ALL test_nelf_lift_link checks passed (16/16 test groups)\n";
    return 0;
}
