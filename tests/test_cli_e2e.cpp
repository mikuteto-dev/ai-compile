#include "aic/atlas.hpp"
#include "aic/json_min.hpp"
#include "aic/linker.hpp"
#include "aic/nelf.hpp"
#include "aic/patch_bundle.hpp"
#include "aic/runtime.hpp"
#include "aic/safetensors.hpp"
#include "aic/tokenizer.hpp"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct CmdResult {
    int exit_code = -1;
    std::string stdout_str;
    std::string stderr_str;
};

std::string get_aic_bin() {
#ifdef AIC_BIN_PATH
    if (fs::exists(AIC_BIN_PATH)) {
        return AIC_BIN_PATH;
    }
#endif
    if (fs::exists("build/aic")) {
        return "build/aic";
    }
    if (fs::exists("./aic")) {
        return "./aic";
    }
    throw std::runtime_error("Could not locate aic executable");
}

std::string get_example_path(const std::string& rel) {
#ifdef AIC_SOURCE_DIR
    const fs::path p = fs::path(AIC_SOURCE_DIR) / rel;
    if (fs::exists(p)) {
        return p.string();
    }
#endif
    if (fs::exists(rel)) {
        return rel;
    }
    const fs::path parent_rel = fs::path("..") / rel;
    if (fs::exists(parent_rel)) {
        return parent_rel.string();
    }
    throw std::runtime_error("Could not locate example file: " + rel);
}

std::string read_file_str(const fs::path& p) {
    std::ifstream ifs(p);
    assert(ifs.good());
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

void write_file_str(const fs::path& p, const std::string& content) {
    std::ofstream ofs(p, std::ios::trunc);
    assert(ofs.good());
    ofs << content;
    assert(ofs.good());
}

std::string read_fd_all(int fd) {
    std::string out;
    char buf[4096];
    for (;;) {
        const ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n > 0) {
            out.append(buf, static_cast<std::size_t>(n));
        } else {
            break;
        }
    }
    return out;
}

CmdResult run_aic(const std::vector<std::string>& args) {
    const std::string bin = get_aic_bin();
    int out_pipe[2] = {-1, -1};
    int err_pipe[2] = {-1, -1};
    if (::pipe(out_pipe) != 0 || ::pipe(err_pipe) != 0) {
        throw std::runtime_error("pipe failed");
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);
        ::close(err_pipe[0]);
        ::close(err_pipe[1]);
        throw std::runtime_error("fork failed");
    }

    if (pid == 0) {
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::dup2(err_pipe[1], STDERR_FILENO);
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);
        ::close(err_pipe[0]);
        ::close(err_pipe[1]);

        std::vector<char*> argv_exec;
        std::vector<std::string> owned;
        owned.reserve(args.size() + 1);
        owned.push_back(bin);
        for (const auto& a : args) {
            owned.push_back(a);
        }
        argv_exec.reserve(owned.size() + 1);
        for (auto& s : owned) {
            argv_exec.push_back(s.data());
        }
        argv_exec.push_back(nullptr);

        ::execv(bin.c_str(), argv_exec.data());
        ::_exit(127);
    }

    ::close(out_pipe[1]);
    ::close(err_pipe[1]);

    CmdResult res;
    res.stdout_str = read_fd_all(out_pipe[0]);
    res.stderr_str = read_fd_all(err_pipe[0]);
    ::close(out_pipe[0]);
    ::close(err_pipe[0]);

    int status = 0;
    ::waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
        res.exit_code = WEXITSTATUS(status);
    } else {
        res.exit_code = 128;
    }
    return res;
}

void test_init_demo_models(const fs::path& tmp) {
    const fs::path qwen_dir = tmp / "qwen_demo";
    const fs::path llama_dir = tmp / "llama_demo";
    const fs::path qwen_bf16_dir = tmp / "qwen_bf16_sharded";

    const CmdResult r1 = run_aic({
        "init-demo-model", "--out", qwen_dir.string(), "--arch", "qwen3-mini", "--seed", "42"
    });
    assert(r1.exit_code == 0);
    assert(fs::exists(qwen_dir / "config.json"));
    assert(fs::exists(qwen_dir / "model.safetensors"));
    assert(fs::exists(qwen_dir / "model.natlas"));
    assert(fs::exists(qwen_dir / "tokenizer.json"));

    const aic::TransformerModel qwen = aic::TransformerModel::load_checkpoint(qwen_dir.string());
    assert(qwen.config.arch_name == "qwen3-mini");
    assert(qwen.config.num_layers == 8);
    assert(qwen.config.hidden_size == 32);
    assert(qwen.config.intermediate_size == 96);
    assert(qwen.config.use_rope);
    assert(qwen.config.qk_norm);

    const aic::NeuralAtlas q_atlas = aic::NeuralAtlas::load((qwen_dir / "model.natlas").string());
    assert(q_atlas.hidden_size == 32);
    assert(q_atlas.num_layers == 8);
    assert(!q_atlas.dormant_poles.empty());

    const aic::Tokenizer q_tok = aic::Tokenizer::load_from_file((qwen_dir / "tokenizer.json").string());
    assert(q_tok.vocab_size() == qwen.config.vocab_size);

    const CmdResult r2 = run_aic({
        "init-demo-model", "--out", llama_dir.string(), "--arch", "llama-mini", "--seed", "77"
    });
    assert(r2.exit_code == 0);
    assert(fs::exists(llama_dir / "tokenizer.json"));
    const aic::TransformerModel llama = aic::TransformerModel::load_checkpoint(llama_dir.string());
    assert(llama.config.arch_name == "llama-mini");
    assert(llama.config.num_layers == 4);
    assert(llama.config.hidden_size == 24);
    assert(llama.config.intermediate_size == 64);
    assert(llama.config.use_rope);
    assert(!llama.config.qk_norm);

    const CmdResult r3 = run_aic({
        "init-demo-model",
        "--out", qwen_bf16_dir.string(),
        "--arch", "qwen3-mini",
        "--dtype", "BF16",
        "--shards", "2",
        "--seed", "42"
    });
    assert(r3.exit_code == 0);
    assert(fs::exists(qwen_bf16_dir / "model.safetensors.index.json"));
    assert(fs::exists(qwen_bf16_dir / "model-00001-of-00002.safetensors"));
    assert(fs::exists(qwen_bf16_dir / "model-00002-of-00002.safetensors"));
    assert(fs::exists(qwen_bf16_dir / "tokenizer.json"));
    assert(fs::exists(qwen_bf16_dir / "model.natlas"));

    const aic::MmapSafetensors shard1((qwen_bf16_dir / "model-00001-of-00002.safetensors").string());
    const auto s1_names = shard1.tensor_names();
    assert(!s1_names.empty());
    assert(shard1.tensor_dtype(s1_names[0]) == "BF16");

    const aic::TransformerModel qwen_bf16 = aic::TransformerModel::load_checkpoint(qwen_bf16_dir.string());
    assert(qwen_bf16.config.num_layers == 8);
    assert(qwen_bf16.config.hidden_size == 32);

    std::cout << "[PASS] test_init_demo_models\n";
}

void test_check_command(const fs::path& tmp) {
    const std::vector<std::string> examples = {
        "examples/01_math_loop_surgery.strata",
        "examples/02_cyclic_weekday_s1.strata",
        "examples/03_tacit_field_decouple.strata",
        "examples/04_rodata_and_transplant.strata"
    };
    for (const auto& ex : examples) {
        const std::string path = get_example_path(ex);
        const std::string content = read_file_str(path);
        const std::string slash2(2, '/');
        const std::string block_open = std::string(1, '/') + "*";
        assert(content.find(slash2) == std::string::npos);
        assert(content.find(block_open) == std::string::npos);

        const CmdResult r = run_aic({"check", path});
        assert(r.exit_code == 0);
        assert(r.stdout_str.find("OK: module") != std::string::npos);
    }

    const fs::path bad_file = tmp / "bad_unipolar.strata";
    write_file_str(
        bad_file,
        "module Bad;\n"
        "pole P in Frame<Mid>;\n"
        "flow F(s: Residual in Frame<Mid>) -> Residual in Frame<Mid> {\n"
        "    s +> P(gain: -0.5);\n"
        "}\n");
    const CmdResult r_bad = run_aic({"check", bad_file.string()});
    assert(r_bad.exit_code != 0);
    assert(r_bad.stderr_str.find("UnipolarViolationError") != std::string::npos);

    const CmdResult r_missing = run_aic({"check", (tmp / "nonexistent.strata").string()});
    assert(r_missing.exit_code != 0);
    assert(!r_missing.stderr_str.empty());

    std::cout << "[PASS] test_check_command\n";
}

void test_ir_command(const fs::path& tmp) {
    const fs::path qwen_dir = tmp / "qwen_demo";
    const std::string ex1 = get_example_path("examples/01_math_loop_surgery.strata");
    const std::string ex2 = get_example_path("examples/02_cyclic_weekday_s1.strata");
    const std::string ex3 = get_example_path("examples/03_tacit_field_decouple.strata");

    const CmdResult r_mir = run_aic({"ir", ex1, "--emit", "mir"});
    assert(r_mir.exit_code == 0);
    assert(r_mir.stdout_str.find("mir.module") != std::string::npos);

    const CmdResult r_cir = run_aic({
        "ir", ex2, "--emit", "cir", "--model", qwen_dir.string()
    });
    assert(r_cir.exit_code == 0);
    assert(r_cir.stdout_str.find("cir.module") != std::string::npos);

    const CmdResult r_wisa = run_aic({
        "ir", ex2, "--emit", "wisa",
        "--model", qwen_dir.string(),
        "--atlas", (qwen_dir / "model.natlas").string(),
        "--target", "elastic"
    });
    assert(r_wisa.exit_code == 0);
    assert(r_wisa.stdout_str.find("SWIGLU_POLE") != std::string::npos);
    assert(r_wisa.stdout_str.find("ATTN_BILINEAR") != std::string::npos);

    const CmdResult r_field_wisa = run_aic({
        "ir", ex3, "--emit", "wisa", "--model", qwen_dir.string()
    });
    assert(r_field_wisa.exit_code == 0);
    assert(r_field_wisa.stdout_str.find("FIELD_SPECTRAL_OP") != std::string::npos);

    const CmdResult r_ood = run_aic({
        "ir", ex1, "--emit", "cir", "--model", qwen_dir.string(), "--target", "in-place"
    });
    assert(r_ood.exit_code != 0);
    assert(r_ood.stderr_str.find("OutOfDepthError") != std::string::npos);

    std::cout << "[PASS] test_ir_command\n";
}

void test_compile_and_streaming_command(const fs::path& tmp) {
    const fs::path qwen_dir = tmp / "qwen_demo";
    const fs::path qwen_bf16_dir = tmp / "qwen_bf16_sharded";
    const fs::path out_inplace = tmp / "qwen_compiled_inplace";
    const fs::path out_elastic = tmp / "qwen_compiled_elastic";
    const fs::path out_field = tmp / "qwen_compiled_field";
    const fs::path out_stream_ip = tmp / "qwen_stream_inplace";
    const fs::path out_stream_el = tmp / "qwen_stream_elastic";
    const fs::path out_contrastive = tmp / "qwen_stream_contrastive";

    const std::string ex2 = get_example_path("examples/02_cyclic_weekday_s1.strata");
    const std::string ex3 = get_example_path("examples/03_tacit_field_decouple.strata");

    const CmdResult r_ip = run_aic({
        "compile", ex2,
        "--model", qwen_dir.string(),
        "--atlas", (qwen_dir / "model.natlas").string(),
        "--out", out_inplace.string(),
        "--target", "in-place"
    });
    assert(r_ip.exit_code == 0);
    const aic::TransformerModel m_ip = aic::TransformerModel::load_checkpoint(out_inplace.string());
    assert(m_ip.config.num_layers == 8);
    assert(m_ip.config.intermediate_size == 96);
    auto [logits_ip, tr_ip] = m_ip.forward({1, 2, 3}, {}, false);
    (void)tr_ip;
    assert(logits_ip.size() == 3);

    const CmdResult r_el = run_aic({
        "compile", ex2,
        "--model", qwen_dir.string(),
        "--out", out_elastic.string(),
        "--target=elastic"
    });
    assert(r_el.exit_code == 0);
    const aic::TransformerModel m_el = aic::TransformerModel::load_checkpoint(out_elastic.string());
    assert(m_el.config.num_layers == 8);
    assert(m_el.config.intermediate_size > 96);
    auto [logits_el, tr_el] = m_el.forward({1, 2, 3}, {}, false);
    (void)tr_el;
    assert(logits_el.size() == 3);

    const CmdResult r_field = run_aic({
        "compile", ex3,
        "--model", qwen_dir.string(),
        "--out", out_field.string(),
        "--target=in-place"
    });
    assert(r_field.exit_code == 0);
    const aic::TransformerModel m_field = aic::TransformerModel::load_checkpoint(out_field.string());
    assert(m_field.config.intermediate_size == 96);

    const CmdResult r_stream_ip = run_aic({
        "compile", ex2,
        "--model", qwen_bf16_dir.string(),
        "--out", out_stream_ip.string(),
        "--target", "in-place",
        "--streaming"
    });
    assert(r_stream_ip.exit_code == 0);
    assert(r_stream_ip.stdout_str.find("streaming") != std::string::npos);
    assert(fs::exists(out_stream_ip / "model.safetensors.index.json"));
    assert(fs::exists(out_stream_ip / "tokenizer.json"));
    const aic::TransformerModel m_stream_ip = aic::TransformerModel::load_checkpoint(out_stream_ip.string());
    assert(m_stream_ip.config.intermediate_size == 96);
    const aic::MmapSafetensors stream_s1((out_stream_ip / "model-00001-of-00002.safetensors").string());
    assert(stream_s1.tensor_dtype(stream_s1.tensor_names().front()) == "BF16");

    const CmdResult r_stream_el = run_aic({
        "compile", ex2,
        "--model", qwen_bf16_dir.string(),
        "--out", out_stream_el.string(),
        "--target", "elastic",
        "--streaming"
    });
    assert(r_stream_el.exit_code == 0);
    const aic::TransformerModel m_stream_el = aic::TransformerModel::load_checkpoint(out_stream_el.string());
    assert(m_stream_el.config.intermediate_size > 96);
    auto [logits_se, tr_se] = m_stream_el.forward({1, 2, 3}, {}, false);
    (void)tr_se;
    assert(logits_se.size() == 3);

    const fs::path contrastive_src = tmp / "contrastive_compile.strata";
    write_file_str(
        contrastive_src,
        "module ContrastiveCompile;\n"
        "pole ContrastiveSignal in Frame<All> {\n"
        "    anchor contrastive(pos: [\"positive signal\"], neg: [\"negative signal\"]);\n"
        "}\n"
        "flow RemoveSignal(state: Residual in Frame<All>) -> Residual in Frame<All> {\n"
        "    state -| ContrastiveSignal(scale: 0.1);\n"
        "}\n");
    const CmdResult r_contrastive = run_aic({
        "compile", contrastive_src.string(),
        "--model", qwen_bf16_dir.string(),
        "--out", out_contrastive.string(),
        "--target", "in-place",
        "--streaming"
    });
    assert(r_contrastive.exit_code == 0);

    const aic::TransformerModel contrastive_base =
        aic::TransformerModel::load_checkpoint(qwen_bf16_dir.string());
    const aic::Tokenizer contrastive_tok = aic::Tokenizer::load_from_file(
        (qwen_bf16_dir / "tokenizer.json").string());
    const aic::MIRModule contrastive_mir = aic::compile_source_to_mir(
        read_file_str(contrastive_src), contrastive_src.string());
    const aic::NeuralAtlas expected_contrastive = aic::resolve_symbols_for_target(
        contrastive_mir, contrastive_base, nullptr, 42, &contrastive_tok);
    const aic::NeuralAtlas actual_contrastive = aic::NeuralAtlas::load(
        (out_contrastive / "model.natlas").string());
    assert(actual_contrastive.symbols.at("ContrastiveSignal").pole_slots ==
           expected_contrastive.symbols.at("ContrastiveSignal").pole_slots);
    assert(aic::dot(
        actual_contrastive.symbols.at("ContrastiveSignal").direction,
        expected_contrastive.symbols.at("ContrastiveSignal").direction) > 0.999999);

    const fs::path patch_only_path = tmp / "contrastive_patch_only.npatch";
    const CmdResult r_patch_only = run_aic({
        "compile", contrastive_src.string(),
        "--model", qwen_bf16_dir.string(),
        "--target", "in-place",
        "--patch-only",
        "--emit-patch", patch_only_path.string()
    });
    assert(r_patch_only.exit_code == 0);
    assert(r_patch_only.stdout_str.find("patch-only streaming") != std::string::npos);
    assert(fs::exists(patch_only_path));
    const aic::PatchBundle patch_only = aic::load_patch_bundle(patch_only_path.string());
    assert(!patch_only.instructions.empty());

    std::cout << "[PASS] test_compile_and_streaming_command\n";
}

void test_lift_command(const fs::path& tmp) {
    const fs::path qwen_dir = tmp / "qwen_demo";
    const fs::path nelf_dir = tmp / "qwen_nelf";
    const fs::path strata_out = tmp / "qwen_lifted.strata";
    const fs::path nelf_sub_dir = tmp / "qwen_nelf_sub";
    const fs::path strata_sub_out = tmp / "qwen_sub.strata";

    const CmdResult r_lift = run_aic({
        "lift",
        "--model", qwen_dir.string(),
        "--out", nelf_dir.string(),
        "--strata-out", strata_out.string()
    });
    assert(r_lift.exit_code == 0);
    assert(fs::exists(nelf_dir / ".text.safetensors"));
    assert(fs::exists(nelf_dir / ".rodata.safetensors"));
    assert(fs::exists(nelf_dir / ".field.safetensors"));
    assert(fs::exists(nelf_dir / ".residual.safetensors"));
    assert(fs::exists(nelf_dir / "passthrough.safetensors"));
    assert(fs::exists(nelf_dir / "atlas.json"));
    assert(fs::exists(nelf_dir / "config.json"));
    assert(fs::exists(nelf_dir / "manifest.json"));
    assert(fs::exists(strata_out));

    const std::string lifted_src = read_file_str(strata_out);
    const std::string slash2(2, '/');
    const std::string block_open = std::string(1, '/') + "*";
    assert(lifted_src.find(slash2) == std::string::npos);
    assert(lifted_src.find(block_open) == std::string::npos);

    const CmdResult r_chk = run_aic({"check", strata_out.string()});
    assert(r_chk.exit_code == 0);

    const aic::NeuralELF nelf = aic::NeuralELF::load(nelf_dir.string());
    const auto ratios = nelf.section_energy_ratios();
    assert(ratios.at(".residual") >= 0.02 && ratios.at(".residual") <= 0.05);
    const aic::TransformerModel recon = nelf.reconstruct_model();
    auto [logits, tr] = recon.forward({1, 4, 9}, {}, false);
    (void)tr;
    assert(logits.size() == 3);

    const CmdResult r_lift_sub = run_aic({
        "lift",
        "--model", qwen_dir.string(),
        "--out", nelf_sub_dir.string(),
        "--strata-out", strata_sub_out.string(),
        "--layers", "2..6"
    });
    assert(r_lift_sub.exit_code == 0);
    assert(fs::exists(strata_sub_out));
    const std::string sub_src = read_file_str(strata_sub_out);
    assert(sub_src.find(slash2) == std::string::npos);
    assert(sub_src.find(block_open) == std::string::npos);
    const CmdResult r_chk_sub = run_aic({"check", strata_sub_out.string()});
    assert(r_chk_sub.exit_code == 0);

    const aic::NeuralELF nelf_sub = aic::NeuralELF::load(nelf_sub_dir.string());
    assert(nelf_sub.text_weights.contains("model.layers.2.mlp.down_proj.weight"));
    assert(!nelf_sub.text_weights.contains("model.layers.0.mlp.down_proj.weight"));
    assert(nelf_sub.passthrough_matrices.contains("model.layers.0.mlp.down_proj.weight"));
    assert(nelf_sub.passthrough_matrices.contains("model.layers.7.mlp.down_proj.weight"));

    const fs::path nelf_default_dir = tmp / "qwen_nelf_default";
    const CmdResult r_lift2 = run_aic({
        "lift", "--model", qwen_dir.string(), "--out", nelf_default_dir.string()
    });
    assert(r_lift2.exit_code == 0);
    assert(fs::exists(nelf_default_dir / "lifted.strata"));

    std::cout << "[PASS] test_lift_command\n";
}

void test_link_command(const fs::path& tmp) {
    const fs::path llama_dir = tmp / "llama_demo";
    const fs::path out_llama = tmp / "llama_linked";
    const fs::path out_llama_el = tmp / "llama_linked_elastic";

    const std::string ex4 = get_example_path("examples/04_rodata_and_transplant.strata");
    const std::string ex1 = get_example_path("examples/01_math_loop_surgery.strata");

    const CmdResult r_link = run_aic({
        "link", ex4,
        "--target-model", llama_dir.string(),
        "--out", out_llama.string(),
        "--target", "in-place"
    });
    assert(r_link.exit_code == 0);
    assert(r_link.stdout_str.find("tokenizer=yes") != std::string::npos);
    assert(fs::exists(out_llama / "tokenizer.json"));
    const aic::TransformerModel linked = aic::TransformerModel::load_checkpoint(out_llama.string());
    assert(linked.config.arch_name == "llama-mini");
    assert(linked.config.num_layers == 4);
    assert(linked.config.hidden_size == 24);
    assert(linked.config.intermediate_size == 64);
    auto [logits, tr] = linked.forward({1, 2, 3, 4}, {}, false);
    (void)tr;
    assert(logits.size() == 4);

    const CmdResult r_link_el = run_aic({
        "link", ex1,
        "--target-model", llama_dir.string(),
        "--atlas", (llama_dir / "model.natlas").string(),
        "--out", out_llama_el.string(),
        "--target=elastic"
    });
    assert(r_link_el.exit_code == 0);
    const aic::TransformerModel linked_el = aic::TransformerModel::load_checkpoint(out_llama_el.string());
    assert(linked_el.config.num_layers > 4);
    auto [logits_el, tr_el] = linked_el.forward({2, 4, 6}, {}, false);
    (void)tr_el;
    assert(logits_el.size() == 3);

    std::cout << "[PASS] test_link_command\n";
}

void test_inspect_run_bench(const fs::path& tmp) {
    const fs::path qwen_bf16_dir = tmp / "qwen_bf16_sharded";
    const fs::path qwen_dir = tmp / "qwen_demo";

    const CmdResult r_insp = run_aic({"inspect", "--model", qwen_bf16_dir.string()});
    assert(r_insp.exit_code == 0);
    assert(r_insp.stdout_str.find("arch: qwen3-mini") != std::string::npos);
    assert(r_insp.stdout_str.find("use_rope: true") != std::string::npos);
    assert(r_insp.stdout_str.find("qk_norm: true") != std::string::npos);
    assert(r_insp.stdout_str.find("dtype: BF16") != std::string::npos);
    assert(r_insp.stdout_str.find("shards: 2") != std::string::npos);
    assert(r_insp.stdout_str.find("tokenizer: present") != std::string::npos);
    assert(r_insp.stdout_str.find("accelerate:") != std::string::npos);
    assert(r_insp.stdout_str.find("metal_available:") != std::string::npos);

    const CmdResult r_run_greedy = run_aic({
        "run", "--model", qwen_bf16_dir.string(), "--prompt", "Step 1:", "--max-tokens", "6"
    });
    assert(r_run_greedy.exit_code == 0);
    assert(r_run_greedy.stdout_str.find("prompt_ids:") != std::string::npos);
    assert(r_run_greedy.stdout_str.find("generated_ids:") != std::string::npos);
    assert(r_run_greedy.stdout_str.find("output_text:") != std::string::npos);

    const CmdResult r_run_streaming = run_aic({
        "run", "--model", qwen_bf16_dir.string(), "--prompt", "Step 1:",
        "--max-tokens", "1", "--streaming"
    });
    assert(r_run_streaming.exit_code == 0);
    assert(r_run_streaming.stdout_str.find("generated_ids:") != std::string::npos);

    const CmdResult r_run_sampled = run_aic({
        "run", "--model", qwen_dir.string(),
        "--prompt", "Step 1:", "--max-tokens", "12",
        "--temp", "0.6", "--top-p", "0.9"
    });
    assert(r_run_sampled.exit_code == 0);
    assert(r_run_sampled.stdout_str.find("prompt_ids:") != std::string::npos);
    assert(r_run_sampled.stdout_str.find("generated_ids:") != std::string::npos);

    const CmdResult r_bench = run_aic({"bench", "--dim", "128", "--layers", "2"});
    assert(r_bench.exit_code == 0);
    assert(r_bench.stdout_str.find("accelerate_matmul:") != std::string::npos);
    assert(r_bench.stdout_str.find("metal_matmul") != std::string::npos);
    assert(r_bench.stdout_str.find("GFLOPS") != std::string::npos);
    assert(r_bench.stdout_str.find("s1_tessellation_nullspace_cir:") != std::string::npos);
    assert(r_bench.stdout_str.find("streaming_bf16_patch:") != std::string::npos);
    assert(r_bench.stdout_str.find("KVCache forward_step:") != std::string::npos);

    std::cout << "[PASS] test_inspect_run_bench\n";
}

void test_phase3_new_commands(const fs::path& tmp) {
    const fs::path qwen_dir = tmp / "qwen_demo";
    const fs::path qwen_bf16_dir = tmp / "qwen_bf16_sharded";
    const fs::path out_compiled = tmp / "qwen_compiled_inplace";

    const std::string ex01 = get_example_path("examples/01_math_loop_surgery.strata");
    const std::string ex05 = get_example_path("examples/05_torus_and_imports.strata");
    const std::string lib_dir = get_example_path("examples/lib");

    const CmdResult r_fmt_check = run_aic({"fmt", "--check", ex01});
    assert(r_fmt_check.exit_code == 0);

    const fs::path fmt_copy = tmp / "fmt_test_copy.strata";
    write_file_str(fmt_copy, read_file_str(ex01));
    const CmdResult r_fmt_write = run_aic({"fmt", "--write", fmt_copy.string()});
    assert(r_fmt_write.exit_code == 0);
    const CmdResult r_fmt_check2 = run_aic({"fmt", "--check", fmt_copy.string()});
    assert(r_fmt_check2.exit_code == 0);

    const CmdResult r_check_inc = run_aic({"check", "-I", lib_dir, ex05});
    assert(r_check_inc.exit_code == 0);
    assert(r_check_inc.stdout_str.find("OK: module") != std::string::npos);

    const CmdResult r_ir_torus = run_aic({
        "ir", ex05,
        "-I", lib_dir,
        "--emit", "mir"
    });
    assert(r_ir_torus.exit_code == 0);
    assert(r_ir_torus.stdout_str.find("mir.module") != std::string::npos);

    const fs::path out_torus = tmp / "qwen_torus_elastic";
    const CmdResult r_compile_torus = run_aic({
        "compile", ex05,
        "-I", lib_dir,
        "--model", qwen_dir.string(),
        "--out", out_torus.string(),
        "--target", "elastic",
        "--opt-level", "2"
    });
    assert(r_compile_torus.exit_code == 0);
    assert(fs::exists(out_torus / "config.json"));

    const std::string ex02 = get_example_path("examples/02_cyclic_weekday_s1.strata");
    const fs::path out_stream_opt1 = tmp / "qwen_stream_opt1";
    const CmdResult r_stream_opt1 = run_aic({
        "compile", ex02,
        "--model", qwen_bf16_dir.string(),
        "--out", out_stream_opt1.string(),
        "--target", "in-place",
        "--streaming",
        "--opt-level", "1"
    });
    assert(r_stream_opt1.exit_code == 0);
    assert(r_stream_opt1.stdout_str.find("streaming") != std::string::npos);

    const CmdResult r_verify = run_aic({
        "verify",
        "--orig", qwen_dir.string(),
        "--patched", out_compiled.string(),
        "--samples", "4"
    });
    assert(r_verify.exit_code == 0);
    assert(r_verify.stdout_str.find("mean_kl:") != std::string::npos);
    assert(r_verify.stdout_str.find("max_logit_diff:") != std::string::npos);

    const CmdResult r_verify_elastic = run_aic({
        "verify",
        "--orig", qwen_dir.string(),
        "--patched", out_torus.string(),
        "--samples", "4"
    });
    assert(r_verify_elastic.exit_code == 0);
    assert(r_verify_elastic.stdout_str.find("untouched_layers:") != std::string::npos);
    assert(r_verify_elastic.stdout_str.find("mean_kl:") != std::string::npos);

    const fs::path nelf_stream_dir = tmp / "qwen_nelf_streaming";
    const fs::path strata_stream_out = tmp / "qwen_stream_lifted.strata";
    const CmdResult r_lift_stream = run_aic({
        "lift",
        "--streaming",
        "--layers", "2..6",
        "--model", qwen_dir.string(),
        "--out", nelf_stream_dir.string(),
        "--strata-out", strata_stream_out.string()
    });
    assert(r_lift_stream.exit_code == 0);
    assert(fs::exists(strata_stream_out));
    const std::string stream_src = read_file_str(strata_stream_out);
    const std::string slash2(2, '/');
    const std::string block_open = std::string(1, '/') + "*";
    assert(stream_src.find(slash2) == std::string::npos);
    assert(stream_src.find(block_open) == std::string::npos);
    const CmdResult r_chk_stream = run_aic({"check", strata_stream_out.string()});
    assert(r_chk_stream.exit_code == 0);

    std::cout << "[PASS] test_phase3_new_commands\n";
}

void test_phase4_commands(const fs::path& tmp) {
    const fs::path qwen_dir = tmp / "qwen_demo";
    const fs::path qwen_fp8_dir = tmp / "qwen_fp8";
    const std::string ex02 = get_example_path("examples/02_cyclic_weekday_s1.strata");
    const std::string ex06 = get_example_path("examples/06_modular_flows_and_lora.strata");
    const std::string lib_dir = get_example_path("examples/lib");

    const CmdResult r_init_fp8 = run_aic({
        "init-demo-model", "--out", qwen_fp8_dir.string(),
        "--arch", "qwen3-mini", "--dtype", "FP8", "--shards", "1"
    });
    assert(r_init_fp8.exit_code == 0);
    assert(fs::exists(qwen_fp8_dir / "config.json"));

    const CmdResult r_fmt_check_06 = run_aic({"fmt", "--check", ex06});
    assert(r_fmt_check_06.exit_code == 0);

    const CmdResult r_lint_06 = run_aic({"lint", ex06});
    assert(r_lint_06.exit_code == 0);
    assert(r_lint_06.stdout_str.find("lint OK") != std::string::npos);

    const fs::path lora_out = tmp / "qwen_lora_out";
    const CmdResult r_compile_lora = run_aic({
        "compile", ex06,
        "--model", qwen_dir.string(),
        "--out", lora_out.string(),
        "--target", "lora"
    });
    assert(r_compile_lora.exit_code == 0);
    assert(fs::exists(lora_out / "adapter_model.safetensors"));
    assert(fs::exists(lora_out / "adapter_config.json"));

    const CmdResult r_run_lora = run_aic({
        "run",
        "--model", qwen_dir.string(),
        "--prompt", "Step 1:",
        "--max-tokens", "4",
        "--lora", lora_out.string()
    });
    assert(r_run_lora.exit_code == 0);
    assert(r_run_lora.stdout_str.find("generated_ids:") != std::string::npos);

    const fs::path patch_out = tmp / "qwen_compiled_p1";
    const fs::path npatch_path = tmp / "p1.npatch";
    const CmdResult r_compile_patch = run_aic({
        "compile", ex02,
        "--model", qwen_dir.string(),
        "--out", patch_out.string(),
        "--target", "in-place",
        "--emit-patch", npatch_path.string()
    });
    assert(r_compile_patch.exit_code == 0);
    assert(fs::exists(npatch_path));

    const fs::path restored_dir = tmp / "qwen_restored";
    const CmdResult r_patch_revert = run_aic({
        "patch-revert", npatch_path.string(),
        "--model", patch_out.string(),
        "--out", restored_dir.string()
    });
    assert(r_patch_revert.exit_code == 0);
    assert(fs::exists(restored_dir / "config.json"));

    const CmdResult r_verify_restored = run_aic({
        "verify",
        "--orig", qwen_dir.string(),
        "--patched", restored_dir.string(),
        "--samples", "4"
    });
    assert(r_verify_restored.exit_code == 0);
    assert(r_verify_restored.stdout_str.find("mean_kl: 0.000000") != std::string::npos);
    assert(r_verify_restored.stdout_str.find("max_kl: 0.000000") != std::string::npos);
    assert(r_verify_restored.stdout_str.find("weight_frob_diff: 0.000000") != std::string::npos);
    assert(r_verify_restored.stdout_str.find("untouched_layers: 8/8") != std::string::npos);

    const fs::path patch_out2 = tmp / "qwen_compiled_p2";
    const fs::path npatch2_path = tmp / "p2.npatch";
    const std::string ex04 = get_example_path("examples/04_rodata_and_transplant.strata");
    const CmdResult r_compile_patch2 = run_aic({
        "compile", ex04,
        "--model", qwen_dir.string(),
        "--out", patch_out2.string(),
        "--target", "in-place",
        "--emit-patch", npatch2_path.string()
    });
    assert(r_compile_patch2.exit_code == 0);
    assert(fs::exists(npatch2_path));

    const fs::path npatch_composed = tmp / "composed.npatch";
    const CmdResult r_compose = run_aic({
        "patch-compose", npatch_path.string(), npatch2_path.string(),
        "--out", npatch_composed.string()
    });
    assert(r_compose.exit_code == 0);
    assert(fs::exists(npatch_composed));

    const CmdResult r_patch_apply = run_aic({
        "patch-apply", npatch_path.string(),
        "--model", qwen_dir.string(),
        "--out", (tmp / "qwen_applied").string()
    });
    assert(r_patch_apply.exit_code == 0);
    assert(fs::exists((tmp / "qwen_applied") / "config.json"));

    const CmdResult r_trace = run_aic({
        "trace",
        "--model", qwen_dir.string(),
        "--prompt", "Wait, let me double check.",
        "--strata", ex06
    });
    assert(r_trace.exit_code == 0);
    assert(r_trace.stdout_str.find("layer_residual_rms") != std::string::npos
           || r_trace.stdout_str.find("predicted_next_token") != std::string::npos
           || r_trace.stdout_str.find("tokens") != std::string::npos);

    const fs::path nelf_dir = tmp / "qwen_nelf_p4";
    const CmdResult r_lift = run_aic({
        "lift", "--model", qwen_dir.string(), "--out", nelf_dir.string()
    });
    assert(r_lift.exit_code == 0);

    const CmdResult r_diff = run_aic({
        "diff-nelf",
        "--lhs", nelf_dir.string(),
        "--rhs", nelf_dir.string()
    });
    assert(r_diff.exit_code == 0);
    assert(r_diff.stdout_str.find("text_frob_diff") != std::string::npos
           || r_diff.stdout_str.find("0.000000") != std::string::npos);

    const CmdResult r_bench = run_aic({"bench", "--dim", "600", "--layers", "4"});
    assert(r_bench.exit_code == 0);
    assert(r_bench.stdout_str.find("randomized_svd") != std::string::npos);

    std::cout << "[PASS] test_phase4_commands\n";
}

void test_phase5_commands(const fs::path& tmp) {
    const std::string qwen_dir = (tmp / "qwen_p5").string();
    const CmdResult r_init = run_aic({
        "init-demo-model", "--out", qwen_dir
    });
    assert(r_init.exit_code == 0);

    const std::string int4_dir = (tmp / "qwen_int4").string();
    const CmdResult r_int4 = run_aic({
        "init-demo-model", "--out", int4_dir, "--dtype", "INT4", "--shards", "2"
    });
    assert(r_int4.exit_code == 0);
    assert(r_int4.stdout_str.find("I4") != std::string::npos);

    const CmdResult r_inspect_int4 = run_aic({"inspect", "--model", int4_dir});
    assert(r_inspect_int4.exit_code == 0);
    assert(r_inspect_int4.stdout_str.find("I4") != std::string::npos);
    assert(r_inspect_int4.stdout_str.find("shards: 2") != std::string::npos);
    assert(fs::exists(tmp / "qwen_int4" / "model.safetensors.index.json"));

    const std::string int4_stream_dir = (tmp / "qwen_int4_compiled").string();
    const CmdResult r_int4_compile = run_aic({
        "compile", get_example_path("examples/06_modular_flows_and_lora.strata"),
        "--model", int4_dir, "--out", int4_stream_dir, "--streaming"
    });
    assert(r_int4_compile.exit_code == 0);
    assert(fs::exists(tmp / "qwen_int4_compiled" / "model.safetensors.index.json"));

    const CmdResult r_int4_run = run_aic({
        "run", "--model", int4_stream_dir, "--prompt", "Step 1:", "--max-tokens", "4"
    });
    assert(r_int4_run.exit_code == 0);
    assert(r_int4_run.stdout_str.find("output_text") != std::string::npos);

    const std::string ex07 = get_example_path("examples/07_verified_rome_and_tests.strata");

    const CmdResult r_fmt_ex07 = run_aic({"fmt", "--check", ex07});
    assert(r_fmt_ex07.exit_code == 0);

    const CmdResult r_lint_ex07 = run_aic({"lint", ex07});
    assert(r_lint_ex07.exit_code == 0);
    assert(r_lint_ex07.stdout_str.find("0 warnings") != std::string::npos);

    const CmdResult r_test_ex07 = run_aic({
        "test", ex07, "--model", qwen_dir
    });
    if (r_test_ex07.exit_code != 0) {
        std::cerr << r_test_ex07.stdout_str << r_test_ex07.stderr_str;
    }
    assert(r_test_ex07.exit_code == 0);
    assert(r_test_ex07.stdout_str.find("5/5 assertions passed") != std::string::npos);

    const CmdResult r_test_lora = run_aic({
        "test", ex07, "--model", qwen_dir, "--target", "lora"
    });
    assert(r_test_lora.exit_code == 0);
    assert(r_test_lora.stdout_str.find("5/5 assertions passed") != std::string::npos);

    const fs::path fail_strata = tmp / "fail_test.strata";
    write_file_str(fail_strata,
        "module FailTest;\n"
        "pole FP in Frame<Mid> {\n"
        "    curvature 0.1;\n"
        "    chart \"linear\";\n"
        "    anchor probe(\"test\");\n"
        "}\n"
        "flow TrivialPatch(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>\n"
        "    ![crosstalk <= 1e-3, approx <= 0.1, preserve(Residual)]\n"
        "{\n"
        "    when state ~> FP(threshold: 0.9) {\n"
        "        state +> FP(gain: 0.01);\n"
        "    }\n"
        "}\n"
        "test AlwaysFail {\n"
        "    expect crosstalk > 999.0;\n"
        "}\n"
    );
    const CmdResult r_test_fail = run_aic({
        "test", fail_strata.string(), "--model", qwen_dir
    });
    assert(r_test_fail.exit_code != 0);
    assert(r_test_fail.stdout_str.find("FAIL") != std::string::npos);

    const std::string patch_path = (tmp / "p5.npatch").string();
    const std::string patched_dir = (tmp / "patched_p5").string();
    const CmdResult r_compile_patch = run_aic({
        "compile", get_example_path("examples/02_cyclic_weekday_s1.strata"),
        "--model", qwen_dir, "--out", patched_dir, "--emit-patch", patch_path
    });
    assert(r_compile_patch.exit_code == 0);

    const std::string stream_applied_dir = (tmp / "stream_applied_p5").string();
    const CmdResult r_stream_apply = run_aic({
        "patch-apply", patch_path, "--model", qwen_dir,
        "--out", stream_applied_dir, "--streaming"
    });
    assert(r_stream_apply.exit_code == 0);
    assert(r_stream_apply.stdout_str.find("streaming") != std::string::npos);

    const std::string stream_reverted_dir = (tmp / "stream_reverted_p5").string();
    const CmdResult r_stream_revert = run_aic({
        "patch-revert", patch_path, "--model", stream_applied_dir,
        "--out", stream_reverted_dir, "--streaming"
    });
    assert(r_stream_revert.exit_code == 0);
    assert(r_stream_revert.stdout_str.find("streaming") != std::string::npos);

    const CmdResult r_verify_streaming = run_aic({
        "verify", "--orig", qwen_dir, "--patched", stream_reverted_dir, "--samples", "4"
    });
    assert(r_verify_streaming.exit_code == 0);
    assert(r_verify_streaming.stdout_str.find("weight_frob_diff: 0.000000") != std::string::npos);
    assert(r_verify_streaming.stdout_str.find("untouched_layers: 8/8") != std::string::npos);
    assert(r_verify_streaming.stdout_str.find("mean_kl: 0.000000") != std::string::npos);

    const CmdResult r_gen_trace = run_aic({
        "trace", "--model", qwen_dir, "--prompt", "Hello world", "--gen-tokens", "4"
    });
    assert(r_gen_trace.exit_code == 0);
    assert(r_gen_trace.stdout_str.find("GenerationTrace") != std::string::npos
           || r_gen_trace.stdout_str.find("generated_tokens") != std::string::npos);

    const CmdResult r_compare_trace = run_aic({
        "trace", "--model", qwen_dir, "--prompt", "Hello world", "--compare", patched_dir
    });
    assert(r_compare_trace.exit_code == 0);
    assert(r_compare_trace.stdout_str.find("TraceComparison") != std::string::npos
           || r_compare_trace.stdout_str.find("kl_divergence") != std::string::npos
           || r_compare_trace.stdout_str.find("base_next_token") != std::string::npos);

    const std::string rome_dir = (tmp / "rome_p5").string();
    const CmdResult r_rome_compile = run_aic({
        "compile", ex07, "--model", qwen_dir, "--out", rome_dir
    });
    assert(r_rome_compile.exit_code == 0);
    const CmdResult r_rome_run = run_aic({
        "run", "--model", rome_dir, "--prompt", "The capital city of Velmoria is", "--max-tokens", "4"
    });
    assert(r_rome_run.exit_code == 0);
    assert(r_rome_run.stdout_str.find("output_text") != std::string::npos);

    std::cout << "[PASS] test_phase5_commands\n";
}

double extract_metric_val(const std::string& out, const std::string& key) {
    const std::size_t pos = out.find(key);
    assert(pos != std::string::npos);
    return std::stod(out.substr(pos + key.size()));
}

void test_phase6_commands(const fs::path& tmp) {
    const std::string lib_dir = get_example_path("examples/lib");
    const std::string qwen_dir = (tmp / "qwen_p6").string();
    const CmdResult r_init = run_aic({"init-demo-model", "--out", qwen_dir, "--seed", "42"});
    assert(r_init.exit_code == 0);

    const std::string nelf_dir = (tmp / "nelf_p6").string();
    const fs::path lifted_strata = tmp / "lifted_p6.strata";
    const CmdResult r_lift = run_aic({
        "lift", "--model", qwen_dir, "--out", nelf_dir,
        "--strata-out", lifted_strata.string()
    });
    assert(r_lift.exit_code == 0);

    const CmdResult r_fmt_lifted = run_aic({"fmt", "--check", lifted_strata.string()});
    assert(r_fmt_lifted.exit_code == 0);

    const CmdResult r_test_lifted = run_aic({
        "test", lifted_strata.string(), "--model", qwen_dir
    });
    assert(r_test_lifted.exit_code == 0);
    assert(r_test_lifted.stdout_str.find("VerifyLiftedFaithfulness: PASS") != std::string::npos);

    const std::string rebuilt_dir = (tmp / "rebuilt_p6").string();
    const CmdResult r_build_nelf = run_aic({
        "build-nelf", lifted_strata.string(),
        "--nelf", nelf_dir,
        "--out", rebuilt_dir,
        "--verify-orig", qwen_dir
    });
    assert(r_build_nelf.exit_code == 0);
    assert(r_build_nelf.stdout_str.find("mean_kl_div: 0.000000") != std::string::npos);
    const double roundtrip_frob = extract_metric_val(r_build_nelf.stdout_str, "weight_frob_diff: ");
    assert(roundtrip_frob < 1e-4);
    const double orig_res_ratio = extract_metric_val(r_build_nelf.stdout_str, "residual_energy_ratio: ");
    assert(orig_res_ratio > 0.01 && orig_res_ratio < 0.10);

    std::string lifted_src = read_file_str(lifted_strata);
    const std::size_t flow_pos = lifted_src.find("flow LiftedProgram");
    assert(flow_pos != std::string::npos);
    const std::size_t open_pos = lifted_src.find('{', flow_pos);
    const std::size_t close_pos = lifted_src.find('}', open_pos);
    assert(open_pos != std::string::npos && close_pos != std::string::npos);
    lifted_src.insert(close_pos, "    state +> LiftedDipole.+pole(gain: 0.1);\n");
    const fs::path mutated_strata = tmp / "lifted_mutated_p6.strata";
    write_file_str(mutated_strata, lifted_src);

    const std::string mutated_dir = (tmp / "rebuilt_mutated_p6").string();
    const CmdResult r_build_mutated = run_aic({
        "build-nelf", mutated_strata.string(),
        "--nelf", nelf_dir,
        "--out", mutated_dir,
        "--verify-orig", qwen_dir
    });
    assert(r_build_mutated.exit_code == 0);
    const double mutated_frob = extract_metric_val(r_build_mutated.stdout_str, "weight_frob_diff: ");
    assert(mutated_frob > 0.01);
    const double mutated_res_ratio = extract_metric_val(r_build_mutated.stdout_str, "residual_energy_ratio: ");
    assert(mutated_res_ratio > 0.01);

    const std::string ex08 = get_example_path("examples/08_memit_batch_and_fisher.strata");
    const CmdResult r_fmt_08 = run_aic({"fmt", "--check", ex08});
    assert(r_fmt_08.exit_code == 0);

    const CmdResult r_lint_08 = run_aic({"lint", ex08, "-I", lib_dir});
    assert(r_lint_08.exit_code == 0);
    assert(r_lint_08.stdout_str.find("0 warnings") != std::string::npos);

    const CmdResult r_test_08 = run_aic({
        "test", ex08, "--model", qwen_dir, "-I", lib_dir
    });
    if (r_test_08.exit_code != 0) {
        std::cerr << r_test_08.stdout_str << r_test_08.stderr_str;
    }
    assert(r_test_08.exit_code == 0);
    assert(r_test_08.stdout_str.find("8/8 assertions passed") != std::string::npos);
    assert(r_test_08.stdout_str.find("VerifyBatchMemitFacts: PASS") != std::string::npos);
    assert(r_test_08.stdout_str.find("VerifyFisherFieldStability: PASS") != std::string::npos);

    std::cout << "[PASS] test_phase6_commands\n";
}

}

int main() {
    const fs::path tmp = fs::temp_directory_path() / ("aic_cli_e2e_test_" + std::to_string(::getpid()));
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    try {
        test_init_demo_models(tmp);
        test_check_command(tmp);
        test_ir_command(tmp);
        test_compile_and_streaming_command(tmp);
        test_lift_command(tmp);
        test_link_command(tmp);
        test_inspect_run_bench(tmp);
        test_phase3_new_commands(tmp);
        test_phase4_commands(tmp);
        test_phase5_commands(tmp);
        test_phase6_commands(tmp);
    } catch (...) {
        fs::remove_all(tmp);
        throw;
    }

    fs::remove_all(tmp);
    std::cout << "ALL test_cli_e2e checks passed (11/11 test groups)\n";
    return 0;
}
