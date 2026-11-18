#include "aic/accel_metal.hpp"
#include "aic/json_min.hpp"
#include "aic/runtime.hpp"
#include "aic/safetensors.hpp"
#include "aic/tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void expect_near(double a, double b, double tol, const char* msg) {
    if (std::abs(a - b) > tol) {
        std::cerr << "FAIL " << msg << ": got " << a << ", expected " << b << " (tol " << tol << ")\n";
        std::abort();
    }
}

void expect_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL " << msg << "\n";
        std::abort();
    }
}

template <typename Fn>
void expect_throws(Fn&& fn, const char* msg) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception&) {
        threw = true;
    }
    if (!threw) {
        std::cerr << "FAIL expected exception: " << msg << "\n";
        std::abort();
    }
}

void test_basic_tensor_ops() {
    using namespace aic;

    Matrix z = zeros(2, 3);
    expect_true(z.rows == 2 && z.cols == 3, "zeros dims");
    expect_near(frob_norm(z), 0.0, 1e-15, "zeros norm");

    Matrix id = eye(4);
    expect_near(frob_norm(id), 2.0, 1e-12, "eye frob");
    expect_near(id(2, 2), 1.0, 1e-15, "eye diag");
    expect_near(id(1, 3), 0.0, 1e-15, "eye offdiag");

    id.set_row(1, {1.0, 2.0, 3.0, 4.0});
    expect_near(id(1, 3), 4.0, 1e-15, "set_row");
    Vector r1 = id.row(1);
    expect_near(r1[2], 3.0, 1e-15, "row");
    id.set_col(2, {5.0, 6.0, 7.0, 8.0});
    Vector c2 = id.col(2);
    expect_near(c2[0], 5.0, 1e-15, "col 0");
    expect_near(c2[3], 8.0, 1e-15, "col 3");

    expect_throws([&]() { (void)id(4, 0); }, "matrix oob");

    Vector a{3.0, 4.0};
    Vector b{1.0, -2.0};
    expect_near(dot(a, b), -5.0, 1e-12, "dot");
    expect_near(norm2(a), 5.0, 1e-12, "norm2");
    Vector na = normalize(a);
    expect_near(norm2(na), 1.0, 1e-12, "normalize");
    expect_near(norm2(normalize(vec_zeros(3))), 0.0, 1e-15, "normalize zero");

    Vector sum = vec_add(a, b);
    Vector diff = vec_sub(a, b);
    Vector sc = vec_scale(a, 0.5);
    expect_near(sum[0], 4.0, 1e-12, "vec_add");
    expect_near(diff[1], 6.0, 1e-12, "vec_sub");
    expect_near(sc[0], 1.5, 1e-12, "vec_scale");

    Matrix op = outer({1.0, 2.0}, {3.0, 4.0, 5.0});
    expect_true(op.rows == 2 && op.cols == 3, "outer shape");
    expect_near(op(1, 2), 10.0, 1e-12, "outer value");

    Matrix A(2, 2);
    A(0, 0) = 2.0;
    A(0, 1) = 1.0;
    A(1, 0) = 5.0;
    A(1, 1) = 3.0;
    Vector rhs{4.0, 11.0};
    Vector sol = solve_linear(A, rhs);
    Vector check = mat_vec(A, sol);
    expect_near(check[0], rhs[0], 1e-11, "solve_linear 0");
    expect_near(check[1], rhs[1], 1e-11, "solve_linear 1");

    Matrix inv = mat_inv(A);
    Matrix prod = mat_mul(A, inv);
    expect_near(frob_norm(mat_sub(prod, eye(2))), 0.0, 1e-11, "mat_inv");

    Matrix singular = zeros(2, 2);
    expect_throws([&]() { (void)solve_linear(singular, {1.0, 2.0}); }, "singular solve");
    expect_throws([&]() { (void)mat_inv(singular); }, "singular inv");

    std::vector<Vector> vecs{
        {1.0, 1.0, 0.0},
        {2.0, 2.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 1.0}};
    std::vector<Vector> ortho = gram_schmidt(vecs);
    expect_true(ortho.size() == 3, "gram_schmidt rank");
    for (std::size_t i = 0; i < ortho.size(); ++i) {
        expect_near(norm2(ortho[i]), 1.0, 1e-12, "gs unit");
        for (std::size_t j = i + 1; j < ortho.size(); ++j) {
            expect_near(dot(ortho[i], ortho[j]), 0.0, 1e-12, "gs orthogonal");
        }
    }

    expect_near(silu(0.0), 0.0, 1e-15, "silu 0");
    expect_near(silu(-1000.0), 0.0, 1e-15, "silu neg large");
    expect_true(silu(2.0) > 1.7 && silu(2.0) < 1.8, "silu 2");

    Vector rn = rms_norm({3.0, 4.0}, {1.0, 2.0}, 0.0);
    const double rms = std::sqrt(12.5);
    expect_near(rn[0], 3.0 / rms, 1e-12, "rms_norm 0");
    expect_near(rn[1], 8.0 / rms, 1e-12, "rms_norm 1");

    Vector sm = softmax({1000.0, 1001.0, 1000.0});
    double sm_sum = sm[0] + sm[1] + sm[2];
    expect_near(sm_sum, 1.0, 1e-12, "softmax sum");
    expect_true(sm[1] > sm[0] && std::abs(sm[0] - sm[2]) < 1e-12, "softmax order");
}

void test_svd_and_procrustes() {
    using namespace aic;

    std::mt19937_64 rng(123);
    std::normal_distribution<double> dist(0.0, 1.0);

    auto check_svd = [&](std::size_t m, std::size_t n, std::size_t k) {
        Matrix A(m, n);
        for (double& x : A.data) {
            x = dist(rng);
        }
        std::vector<SVDMode> modes = svd_top_k(A, k, 60, 7);
        const std::size_t expected_k = std::min({k, m, n});
        expect_true(modes.size() == expected_k, "svd mode count");
        for (std::size_t i = 0; i < modes.size(); ++i) {
            expect_true(modes[i].u.size() == m, "svd u len");
            expect_true(modes[i].v.size() == n, "svd v len");
            expect_near(norm2(modes[i].u), 1.0, 1e-10, "svd u unit");
            expect_near(norm2(modes[i].v), 1.0, 1e-10, "svd v unit");
            if (i + 1 < modes.size()) {
                expect_true(modes[i].sigma >= modes[i + 1].sigma - 1e-12, "svd sorted");
            }
            for (std::size_t j = i + 1; j < modes.size(); ++j) {
                expect_near(dot(modes[i].u, modes[j].u), 0.0, 1e-10, "svd u ortho");
                expect_near(dot(modes[i].v, modes[j].v), 0.0, 1e-10, "svd v ortho");
            }
            Vector av = mat_vec(A, modes[i].v);
            Vector su = vec_scale(modes[i].u, modes[i].sigma);
            expect_near(norm2(vec_sub(av, su)), 0.0, 1e-7, "svd Av == sigma u");
        }
        if (expected_k == std::min(m, n)) {
            Matrix recon = zeros(m, n);
            for (const auto& mode : modes) {
                recon = mat_add(recon, mat_scale(outer(mode.u, mode.v), mode.sigma));
            }
            expect_near(frob_norm(mat_sub(A, recon)), 0.0, 1e-9, "svd full reconstruction");
        }
    };

    check_svd(12, 5, 5);
    check_svd(5, 12, 5);
    check_svd(8, 8, 8);
    check_svd(120, 110, 4);

    {
        const std::size_t hm = 560;
        const std::size_t hn = 540;
        Vector u0 = normalize(Vector(hm, 1.0));
        Vector v0 = normalize(Vector(hn, 1.0));
        Vector u1(hm, 0.0);
        Vector v1(hn, 0.0);
        for (std::size_t i = 0; i < hm; ++i) {
            u1[i] = (i % 2 == 0) ? 1.0 : -1.0;
        }
        for (std::size_t j = 0; j < hn; ++j) {
            v1[j] = (j % 2 == 0) ? 1.0 : -1.0;
        }
        u1 = normalize(u1);
        v1 = normalize(v1);
        Matrix huge = mat_add(
            mat_scale(outer(u0, v0), 10.0),
            mat_scale(outer(u1, v1), 3.0));
        std::vector<SVDMode> h_modes = svd_top_k(huge, 2, 30, 11);
        expect_true(h_modes.size() == 2, "huge svd mode count");
        expect_near(h_modes[0].sigma, 10.0, 1e-4, "huge svd sigma 0");
        expect_near(h_modes[1].sigma, 3.0, 1e-4, "huge svd sigma 1");
    }

    Matrix rank1 = outer({1.0, 2.0, 3.0}, {4.0, -5.0});
    auto r1_modes = svd_top_k(rank1, 2);
    expect_true(r1_modes.size() == 1, "rank1 svd drops zero mode");
    expect_near(r1_modes[0].sigma, std::sqrt(14.0 * 41.0), 1e-10, "rank1 sigma");

    expect_true(svd_top_k(zeros(4, 4), 2).empty(), "zero matrix svd");

    const double theta = 0.73;
    Matrix R_true(3, 3, 0.0);
    R_true(0, 0) = std::cos(theta);
    R_true(0, 1) = -std::sin(theta);
    R_true(1, 0) = std::sin(theta);
    R_true(1, 1) = std::cos(theta);
    R_true(2, 2) = 1.0;

    Matrix src(10, 3);
    for (double& x : src.data) {
        x = dist(rng);
    }
    Matrix tgt = mat_mul(src, R_true);
    Matrix R_est = orthogonal_procrustes(src, tgt);
    expect_near(frob_norm(mat_sub(mat_mul(transpose(R_est), R_est), eye(3))), 0.0, 1e-10, "procrustes ortho");
    expect_near(frob_norm(mat_sub(mat_mul(src, R_est), tgt)), 0.0, 1e-10, "procrustes fit");
}

void test_metal_accel() {
    using namespace aic;

    const bool avail = metal_is_available();
    const std::string dev_name = metal_device_name();
    expect_true(!dev_name.empty(), "metal device name non-empty");
    (void)avail;

    std::mt19937_64 rng(777);
    std::normal_distribution<double> dist(0.0, 1.0);
    Matrix A(17, 23);
    Matrix B(23, 19);
    for (double& x : A.data) {
        x = dist(rng);
    }
    for (double& x : B.data) {
        x = dist(rng);
    }

    Matrix cpu_c = mat_mul(A, B);
    Matrix gpu_c = metal_mat_mul(A, B);
    expect_true(gpu_c.rows == 17 && gpu_c.cols == 19, "metal_mat_mul dims");
    expect_near(frob_norm(mat_sub(cpu_c, gpu_c)), 0.0, 1e-3, "metal_mat_mul parity");

    Matrix W_cpu = zeros(11, 13);
    Matrix W_gpu = zeros(11, 13);
    Vector u(11);
    Vector v(13);
    for (std::size_t i = 0; i < 11; ++i) {
        u[i] = dist(rng);
    }
    for (std::size_t j = 0; j < 13; ++j) {
        v[j] = dist(rng);
    }
    W_cpu = mat_add(W_cpu, mat_scale(outer(u, v), -0.35));
    metal_add_rank1(W_gpu, u, v, -0.35);
    expect_near(frob_norm(mat_sub(W_cpu, W_gpu)), 0.0, 1e-4, "metal_add_rank1 parity");

    Vector gp{-2.0, 0.0, 1.5, 3.0};
    Vector up{1.0, -3.0, 2.0, -0.5};
    Vector sw = metal_swiglu_poles(gp, up);
    expect_true(sw.size() == 4, "metal_swiglu_poles len");
    for (std::size_t i = 0; i < 4; ++i) {
        expect_near(sw[i], silu(gp[i]) * up[i], 1e-5, "metal_swiglu_poles parity");
    }
}

void test_json_min() {
    using namespace aic;

    JsonValue root;
    root["null_v"] = nullptr;
    root["bool_v"] = true;
    root["int_v"] = 42;
    root["size_v"] = static_cast<std::size_t>(100);
    root["dbl_v"] = 1e-6;
    root["str_v"] = "hello\n\"world\"";
    JsonValue arr = JsonValue::array();
    arr.as_array().emplace_back(1);
    arr.as_array().emplace_back("two");
    arr.as_array().emplace_back(false);
    root["arr_v"] = arr;

    expect_true(root.is_object(), "json is_object");
    expect_true(root.has("int_v"), "json has");
    expect_true(!root.has("missing"), "json !has");
    expect_true(root["null_v"].is_null(), "json null");
    expect_true(root["bool_v"].as_bool(), "json bool");
    expect_true(root["int_v"].as_int() == 42, "json int");
    expect_near(root["dbl_v"].as_number(), 1e-6, 1e-15, "json double");
    expect_true(root["str_v"].as_string() == "hello\n\"world\"", "json string");

    const JsonValue& croot = root;
    expect_throws([&]() { (void)croot["nope"]; }, "const json missing key");

    const std::string dumped = dump_json(root, 2);
    const JsonValue parsed = parse_json(dumped);
    expect_true(parsed == root, "json round-trip");

    const std::string compact = dump_json(root, -1);
    expect_true(compact.find('\n') == std::string::npos, "compact json no newlines");
    expect_true(parse_json(compact) == root, "compact json round-trip");

    const JsonValue unicode_test = parse_json("{\"txt\":\"A\\u0042\\uD83D\\uDE00\"}");
    expect_true(unicode_test["txt"].as_string() == "AB\xF0\x9F\x98\x80", "json unicode surrogate");

    expect_throws([]() { (void)parse_json("01"); }, "json leading zero");
    expect_throws([]() { (void)parse_json("{\"a\":1} extra"); }, "json trailing");
    expect_throws([]() { (void)dump_json(JsonValue(std::nan(""))); }, "json nan dump");
}

void test_safetensors_and_mmap_sharded() {
    using namespace aic;

    const std::filesystem::path tmp_path =
        std::filesystem::temp_directory_path() / "aic_test_tensor.safetensors";

    Matrix m(2, 3);
    m(0, 0) = 1.25;
    m(0, 1) = -3.5;
    m(0, 2) = 0.0;
    m(1, 0) = 65504.0;
    m(1, 1) = 5.960464477539063e-8;
    m(1, 2) = -0.125;

    Vector v{0.5, -1.75, 3.125, 1024.0};

    std::map<std::string, TensorSpec> tensors;
    tensors["mat_f32"] = TensorSpec::from_matrix(m, "F32");
    tensors["mat_f16"] = TensorSpec::from_matrix(m, "F16");
    tensors["vec_bf16"] = TensorSpec::from_vector(v, "BF16");

    std::map<std::string, std::string> meta{
        {"arch", "qwen3-mini"},
        {"format", "pt"}};

    save_safetensors(tmp_path.string(), tensors, meta);

    std::ifstream raw_in(tmp_path, std::ios::binary);
    std::uint8_t hdr_len_bytes[8];
    raw_in.read(reinterpret_cast<char*>(hdr_len_bytes), 8);
    std::uint64_t hdr_len = 0;
    for (int i = 0; i < 8; ++i) {
        hdr_len |= static_cast<std::uint64_t>(hdr_len_bytes[i]) << (8 * i);
    }
    expect_true(hdr_len % 8 == 0, "safetensors header aligned to 8 bytes");
    std::string hdr_json(hdr_len, '\0');
    raw_in.read(hdr_json.data(), static_cast<std::streamsize>(hdr_len));
    JsonValue hdr_parsed = parse_json(hdr_json);
    expect_true(hdr_parsed.has("__metadata__"), "safetensors __metadata__");
    expect_true(hdr_parsed["mat_f32"]["dtype"].as_string() == "F32", "safetensors dtype field");
    expect_true(hdr_parsed["mat_f16"]["data_offsets"].as_array().size() == 2, "safetensors offsets");

    auto [loaded_tensors, loaded_meta] = load_safetensors(tmp_path.string());
    expect_true(loaded_meta == meta, "safetensors metadata round-trip");

    Matrix m_f32 = loaded_tensors.at("mat_f32").to_matrix();
    expect_near(frob_norm(mat_sub(m_f32, m)), 0.0, 1e-6, "safetensors F32");

    Matrix m_f16 = loaded_tensors.at("mat_f16").to_matrix();
    expect_near(m_f16(0, 0), 1.25, 1e-6, "safetensors F16 normal");
    expect_near(m_f16(1, 0), 65504.0, 1e-6, "safetensors F16 max");
    expect_near(m_f16(1, 1), 5.960464477539063e-8, 1e-12, "safetensors F16 subnormal");

    Vector v_bf16 = loaded_tensors.at("vec_bf16").to_vector();
    for (std::size_t i = 0; i < v.size(); ++i) {
        expect_near(v_bf16[i], v[i], 1e-2 * std::abs(v[i]) + 1e-6, "safetensors BF16");
    }

    {
        MmapSafetensors mm1(tmp_path.string());
        MmapSafetensors mm = std::move(mm1);
        expect_true(mm.has_tensor("mat_f32"), "mmap has mat_f32");
        expect_true(!mm.has_tensor("missing"), "mmap !has missing");
        expect_true(mm.tensor_names().size() == 3, "mmap tensor_names");
        expect_true(mm.tensor_shape("mat_f16") == std::vector<std::size_t>({2, 3}), "mmap shape");
        expect_true(mm.tensor_dtype("vec_bf16") == "BF16", "mmap dtype");
        expect_true(mm.metadata() == meta, "mmap metadata");
        auto [r0, r1] = mm.raw_byte_range("mat_f32");
        expect_true(r1 - r0 == 24, "mmap raw byte range F32");
        std::string_view raw_sv = mm.raw_tensor_bytes("vec_bf16");
        expect_true(raw_sv.size() == 8, "mmap raw_tensor_bytes BF16");
        Matrix mm_mat = mm.read_tensor("mat_f32").to_matrix();
        expect_near(frob_norm(mat_sub(mm_mat, m)), 0.0, 1e-6, "mmap read_tensor");
    }

    std::filesystem::remove(tmp_path);

    const std::filesystem::path shard_dir =
        std::filesystem::temp_directory_path() / "aic_test_sharded_st";
    std::filesystem::remove_all(shard_dir);

    save_safetensors_sharded(shard_dir.string(), tensors, 16, meta);
    expect_true(std::filesystem::exists(shard_dir / "model.safetensors.index.json"), "index json created");
    expect_true(!std::filesystem::exists(shard_dir / "model.safetensors"), "single model.safetensors absent when sharded");

    auto [sharded_loaded, sharded_meta] = load_safetensors_sharded(shard_dir.string());
    expect_true(sharded_loaded.size() == 3, "sharded loaded count");
    expect_true(sharded_meta == meta, "sharded metadata");
    expect_near(frob_norm(mat_sub(sharded_loaded.at("mat_f32").to_matrix(), m)), 0.0, 1e-6, "sharded F32");

    TensorSpec single_v = load_single_tensor_sharded(shard_dir.string(), "vec_bf16");
    expect_true(single_v.dtype == "BF16", "single tensor sharded dtype");
    expect_true(single_v.shape == std::vector<std::size_t>({4}), "single tensor sharded shape");

    std::filesystem::remove_all(shard_dir);
}

void test_runtime_and_checkpoints() {
    using namespace aic;

    ModelConfig cfg;
    expect_true(cfg.stage_to_layers("Early") == std::pair<int, int>(0, 2), "stage Early");
    expect_true(cfg.stage_to_layers("Mid") == std::pair<int, int>(2, 6), "stage Mid");
    expect_true(cfg.stage_to_layers("Late") == std::pair<int, int>(6, 7), "stage Late");
    expect_true(cfg.stage_to_layers("Logit") == std::pair<int, int>(7, 8), "stage Logit");
    expect_true(cfg.stage_to_layers("All") == std::pair<int, int>(0, 8), "stage All");
    expect_true(cfg.stage_to_layers("Mid", "Late") == std::pair<int, int>(2, 7), "stage Mid..Late");
    expect_true(cfg.stage_to_layers("2", "5", false) == std::pair<int, int>(2, 5), "stage 2..5 excl");
    expect_true(cfg.stage_to_layers("2", "5", true) == std::pair<int, int>(2, 6), "stage 2..5 incl");
    expect_true(cfg.stage_to_layers("3") == std::pair<int, int>(3, 4), "stage single int");
    expect_true(cfg.stage_to_layers("16", "24") == std::pair<int, int>(8, 8), "stage clamped 16..24");

    JsonValue cfg_json = cfg.to_json();
    ModelConfig cfg_rt = ModelConfig::from_json(cfg_json);
    expect_true(cfg_rt.num_layers == cfg.num_layers, "config round-trip layers");
    expect_true(cfg_rt.arch_name == cfg.arch_name, "config round-trip arch");

    const JsonValue hf_json = parse_json(
        "{\"architectures\":[\"Qwen3ForCausalLM\"],\"model_type\":\"qwen3\","
        "\"num_hidden_layers\":3,\"hidden_size\":24,\"intermediate_size\":64,"
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":8,"
        "\"vocab_size\":48,\"rms_norm_eps\":1e-6,\"rope_theta\":1000000.0,"
        "\"tie_word_embeddings\":true,\"torch_dtype\":\"bfloat16\"}");
    ModelConfig hf_cfg = ModelConfig::from_json(hf_json);
    expect_true(hf_cfg.num_layers == 3, "hf num_hidden_layers");
    expect_true(hf_cfg.attention_type == "gqa", "hf attention_type");
    expect_true(hf_cfg.use_rope, "hf qwen3 default use_rope");
    expect_true(hf_cfg.head_dim == 8, "hf head_dim");
    expect_true(hf_cfg.tie_word_embeddings, "hf tie_word_embeddings");
    expect_true(hf_cfg.torch_dtype == "BF16", "hf torch_dtype BF16");
    expect_near(hf_cfg.rope_theta, 1000000.0, 1e-6, "hf rope_theta");

    for (const std::string& attn : {"mha", "gqa", "mla"}) {
        ModelConfig c;
        c.num_layers = 4;
        c.hidden_size = 24;
        c.intermediate_size = 48;
        c.num_attention_heads = 4;
        c.num_key_value_heads = (attn == "mha") ? 4 : 2;
        c.vocab_size = 32;
        c.attention_type = attn;
        c.kv_lora_rank = 6;
        c.arch_name = "test-" + attn;

        TransformerModel model = TransformerModel::create_synthetic(c, 99);
        auto [logits_a, trace_a] = model.forward({1, 5, 9}, {}, true);
        auto [logits_b, trace_b] = model.forward({1, 5, 20}, {}, false);
        (void)trace_b;

        expect_true(logits_a.size() == 3, "logits seq len");
        expect_true(logits_a[0].size() == 32, "logits vocab len");
        expect_true(trace_a.residual.size() == 5, "residual trace L+1");
        expect_true(trace_a.mlp_poles.size() == 4, "mlp poles trace L");
        expect_true(trace_a.mlp_poles[0].size() == 48, "mlp poles trace d_ff");

        for (std::size_t t = 0; t < 2; ++t) {
            expect_near(norm2(vec_sub(logits_a[t], logits_b[t])), 0.0, 1e-12, "causal mask invariant");
        }

        const double g0 = trace_a.mlp_poles[0][0].gate_pre[1];
        const double u0 = trace_a.mlp_poles[0][0].up_val[1];
        const double a0 = trace_a.mlp_poles[0][0].act[1];
        expect_near(a0, silu(g0) * u0, 1e-12, "pole trace swiglu consistency");

        std::vector<Vector> emb_inputs;
        const Matrix& emb = model.matrices.at("model.embed_tokens.weight");
        emb_inputs.push_back(emb.row(1));
        emb_inputs.push_back(emb.row(5));
        emb_inputs.push_back(emb.row(9));
        auto [logits_h, trace_h] = model.forward({}, emb_inputs, false);
        (void)trace_h;
        for (std::size_t t = 0; t < 3; ++t) {
            expect_near(norm2(vec_sub(logits_a[t], logits_h[t])), 0.0, 1e-12, "hidden_inputs parity");
        }
    }

    {
        ModelConfig rope_cfg = hf_cfg;
        rope_cfg.qk_norm = true;
        rope_cfg.qkv_bias = true;
        rope_cfg.torch_dtype = "F32";
        TransformerModel qwen3_mod = TransformerModel::create_synthetic(rope_cfg, 17);
        expect_true(!qwen3_mod.matrices.contains("lm_head.weight"), "tied lm_head omitted");
        expect_true(qwen3_mod.vectors.contains("model.layers.0.self_attn.q_norm.weight"), "qk_norm weights present");
        expect_true(qwen3_mod.vectors.contains("model.layers.0.self_attn.q_proj.bias"), "qkv_bias weights present");

        auto [logits_r1, tr_r1] = qwen3_mod.forward({3, 5, 5}, {}, false);
        auto [logits_r2, tr_r2] = qwen3_mod.forward({3, 5, 11}, {}, false);
        (void)tr_r1;
        (void)tr_r2;
        expect_near(norm2(vec_sub(logits_r1[0], logits_r2[0])), 0.0, 1e-12, "rope causal t=0");
        expect_near(norm2(vec_sub(logits_r1[1], logits_r2[1])), 0.0, 1e-12, "rope causal t=1");
        expect_true(norm2(vec_sub(logits_r1[1], logits_r1[2])) > 1e-6, "rope distinguishes positions");

        TransformerModel no_rope_mod = qwen3_mod;
        no_rope_mod.config.use_rope = false;
        auto [logits_nr, tr_nr] = no_rope_mod.forward({3, 5, 5}, {}, false);
        (void)tr_nr;
        expect_near(norm2(vec_sub(logits_r1[0], logits_nr[0])), 0.0, 1e-12, "rope t=0 invariant");
        expect_true(norm2(vec_sub(logits_r1[1], logits_nr[1])) > 1e-6, "rope modifies multi-token attention");

        std::vector<int> gen = qwen3_mod.generate_greedy({2, 7}, 3);
        expect_true(gen.size() == 5, "generate_greedy length");
        expect_true(gen[0] == 2 && gen[1] == 7, "generate_greedy preserves prompt");

        const std::filesystem::path sharded_ckpt =
            std::filesystem::temp_directory_path() / "aic_test_sharded_ckpt";
        std::filesystem::remove_all(sharded_ckpt);
        std::filesystem::create_directories(sharded_ckpt);
        {
            std::ofstream ofs(sharded_ckpt / "config.json");
            ofs << dump_json(rope_cfg.to_json(), 2) << '\n';
        }
        std::map<std::string, TensorSpec> specs;
        for (const auto& [k, m] : qwen3_mod.matrices) {
            specs[k] = TensorSpec::from_matrix(m, "F32");
        }
        for (const auto& [k, v] : qwen3_mod.vectors) {
            specs[k] = TensorSpec::from_vector(v, "F32");
        }
        save_safetensors_sharded(sharded_ckpt.string(), specs, 4096);

        TransformerModel loaded_sharded = TransformerModel::load_checkpoint(sharded_ckpt.string());
        expect_true(loaded_sharded.config.qk_norm, "auto-detected qk_norm");
        expect_true(loaded_sharded.config.qkv_bias, "auto-detected qkv_bias");
        expect_true(loaded_sharded.config.tie_word_embeddings, "auto-detected tie_word_embeddings");
        std::vector<int> gen_loaded = loaded_sharded.generate_greedy({2, 7}, 3);
        expect_true(gen_loaded == gen, "sharded checkpoint generate_greedy parity");
        std::filesystem::remove_all(sharded_ckpt);
    }

    const std::filesystem::path ckpt_dir =
        std::filesystem::temp_directory_path() / "aic_test_ckpt";
    std::filesystem::remove_all(ckpt_dir);

    TransformerModel orig = TransformerModel::create_synthetic(ModelConfig{}, 42);
    auto [orig_logits, orig_trace] = orig.forward({3, 7, 11}, {}, false);
    (void)orig_trace;
    orig.save_checkpoint(ckpt_dir.string());

    expect_true(std::filesystem::exists(ckpt_dir / "config.json"), "ckpt config.json exists");
    expect_true(std::filesystem::exists(ckpt_dir / "model.safetensors"), "ckpt model.safetensors exists");

    TransformerModel reloaded = TransformerModel::load_checkpoint(ckpt_dir.string());
    auto [reloaded_logits, reloaded_trace] = reloaded.forward({3, 7, 11}, {}, false);
    (void)reloaded_trace;
    expect_true(reloaded.config.arch_name == orig.config.arch_name, "ckpt arch_name");
    for (std::size_t t = 0; t < orig_logits.size(); ++t) {
        expect_near(
            norm2(vec_sub(orig_logits[t], reloaded_logits[t])),
            0.0,
            1e-5,
            "ckpt logits match");
    }

    std::filesystem::remove_all(ckpt_dir);
}

void test_kv_cache_sampling_and_divergence() {
    using namespace aic;

    ModelConfig mha_cfg;
    mha_cfg.num_layers = 3;
    mha_cfg.hidden_size = 24;
    mha_cfg.intermediate_size = 48;
    mha_cfg.num_attention_heads = 4;
    mha_cfg.num_key_value_heads = 4;
    mha_cfg.vocab_size = 32;
    mha_cfg.attention_type = "mha";

    ModelConfig gqa_rope_cfg;
    gqa_rope_cfg.num_layers = 3;
    gqa_rope_cfg.hidden_size = 24;
    gqa_rope_cfg.intermediate_size = 64;
    gqa_rope_cfg.num_attention_heads = 4;
    gqa_rope_cfg.num_key_value_heads = 2;
    gqa_rope_cfg.head_dim = 8;
    gqa_rope_cfg.vocab_size = 36;
    gqa_rope_cfg.attention_type = "gqa";
    gqa_rope_cfg.use_rope = true;
    gqa_rope_cfg.rope_theta = 100000.0;
    gqa_rope_cfg.qk_norm = true;
    gqa_rope_cfg.qkv_bias = true;
    gqa_rope_cfg.tie_word_embeddings = true;

    ModelConfig mla_cfg;
    mla_cfg.num_layers = 3;
    mla_cfg.hidden_size = 24;
    mla_cfg.intermediate_size = 48;
    mla_cfg.num_attention_heads = 4;
    mla_cfg.num_key_value_heads = 4;
    mla_cfg.vocab_size = 32;
    mla_cfg.attention_type = "mla";
    mla_cfg.kv_lora_rank = 6;
    mla_cfg.use_rope = true;

    const std::vector<int> seq{2, 5, 11, 7, 19};

    for (const auto& [name, cfg] : std::vector<std::pair<std::string, ModelConfig>>{
             {"mha", mha_cfg},
             {"gqa_rope_qknorm", gqa_rope_cfg},
             {"mla", mla_cfg}}) {
        TransformerModel model = TransformerModel::create_synthetic(cfg, 1234);
        auto [batch_logits, trace] = model.forward(seq, {}, false);
        (void)trace;
        expect_true(batch_logits.size() == seq.size(), "batched forward len");

        KVCache cache = model.init_kv_cache();
        expect_true(cache.layers.size() == static_cast<std::size_t>(cfg.num_layers), "kv cache layers");
        expect_true(cache.seq_len == 0, "kv cache init seq_len");

        for (std::size_t t = 0; t < seq.size(); ++t) {
            Vector step_logits = model.forward_step(seq[t], cache);
            expect_true(cache.seq_len == t + 1, "kv cache seq_len advances");
            if (cfg.attention_type == "mla") {
                expect_true(cache.layers[0].c_kv.size() == t + 1, "mla c_kv cached");
            }
            const double diff = norm2(vec_sub(batch_logits[t], step_logits));
            expect_near(diff, 0.0, 1e-11, name.c_str());
        }

        std::vector<int> prompt{2, 5, 11};
        std::vector<int> greedy = model.generate_greedy(prompt, 4);
        expect_true(greedy.size() == 7, "greedy length");

        std::vector<int> naive = prompt;
        for (int s = 0; s < 4; ++s) {
            auto [lg, tr] = model.forward(naive, {}, false);
            (void)tr;
            const Vector& last = lg.back();
            std::size_t best = 0;
            for (std::size_t i = 1; i < last.size(); ++i) {
                if (last[i] > last[best]) {
                    best = i;
                }
            }
            naive.push_back(static_cast<int>(best));
        }
        expect_true(greedy == naive, "kv-cached generate_greedy matches naive forward");

        std::vector<int> sampled_t0 = model.generate_sampled(prompt, 4, 0.0, 1.0, 77);
        expect_true(sampled_t0 == greedy, "generate_sampled temp=0 delegates to greedy");

        std::vector<int> sampled_a = model.generate_sampled(prompt, 4, 0.8, 0.9, 99);
        std::vector<int> sampled_b = model.generate_sampled(prompt, 4, 0.8, 0.9, 99);
        expect_true(sampled_a.size() == 7, "sampled len");
        expect_true(sampled_a == sampled_b, "sampled deterministic with seed");
        for (int tok : sampled_a) {
            expect_true(tok >= 0 && tok < cfg.vocab_size, "sampled valid token id");
        }
    }

    TransformerModel p_model = TransformerModel::create_synthetic(gqa_rope_cfg, 55);
    ModelDivergenceStats zero_div = compute_model_divergence(p_model, p_model, {{1, 2, 3}, {4, 5}});
    expect_near(zero_div.mean_kl, 0.0, 1e-13, "self kl mean");
    expect_near(zero_div.max_kl, 0.0, 1e-13, "self kl max");
    expect_near(zero_div.max_logit_diff, 0.0, 1e-13, "self logit diff");

    TransformerModel q_model = p_model;
    q_model.matrices["model.layers.1.mlp.down_proj.weight"] =
        mat_scale(q_model.matrices.at("model.layers.1.mlp.down_proj.weight"), 1.5);
    ModelDivergenceStats pert_div = compute_model_divergence(p_model, q_model, {{1, 2, 3}, {4, 5, 6}});
    expect_true(pert_div.mean_kl > 1e-6, "perturbed mean_kl > 0");
    expect_true(pert_div.max_kl >= pert_div.mean_kl, "perturbed max_kl >= mean_kl");
    expect_true(pert_div.max_logit_diff > 1e-4, "perturbed max_logit_diff > 0");
}

void test_randomized_svd_fp8_i8_and_lora() {
    using namespace aic;

    {
        const std::size_t n = 600;
        const std::size_t r_true = 8;
        std::mt19937_64 rng(2026);
        std::normal_distribution<double> dist(0.0, 1.0);
        std::vector<Vector> u_raw(r_true, Vector(n));
        std::vector<Vector> v_raw(r_true, Vector(n));
        for (std::size_t i = 0; i < r_true; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                u_raw[i][j] = dist(rng);
                v_raw[i][j] = dist(rng);
            }
        }
        const std::vector<Vector> u_basis = gram_schmidt(u_raw);
        const std::vector<Vector> v_basis = gram_schmidt(v_raw);
        Matrix A(n, n, 0.0);
        for (std::size_t k = 0; k < r_true; ++k) {
            const double sigma = 20.0 / static_cast<double>(k + 1);
            for (std::size_t r = 0; r < n; ++r) {
                const double su = sigma * u_basis[k][r];
                for (std::size_t c = 0; c < n; ++c) {
                    A(r, c) += su * v_basis[k][c];
                }
            }
        }
        const std::vector<SVDMode> modes = randomized_svd(A, r_true, 8, 2, 7);
        expect_true(modes.size() == r_true, "randomized_svd mode count");
        Matrix recon(n, n, 0.0);
        for (std::size_t i = 0; i < modes.size(); ++i) {
            expect_near(modes[i].sigma, 20.0 / static_cast<double>(i + 1), 1e-6, "randomized_svd sigma");
            expect_near(norm2(modes[i].u), 1.0, 1e-10, "randomized_svd u unit");
            expect_near(norm2(modes[i].v), 1.0, 1e-10, "randomized_svd v unit");
            for (std::size_t r = 0; r < n; ++r) {
                const double su = modes[i].sigma * modes[i].u[r];
                for (std::size_t c = 0; c < n; ++c) {
                    recon(r, c) += su * modes[i].v[c];
                }
            }
        }
        const double rel_err = frob_norm(mat_sub(A, recon)) / frob_norm(A);
        expect_true(rel_err < 1e-6, "randomized_svd relative reconstruction error < 1e-6");

        const std::vector<SVDMode> routed = svd_top_k(A, r_true, 50, 7);
        expect_true(routed.size() == r_true, "svd_top_k routed to randomized_svd");
        expect_near(routed[0].sigma, 20.0, 1e-6, "routed svd_top_k sigma[0]");
    }

    {
        const std::filesystem::path st_path =
            std::filesystem::temp_directory_path() / "aic_test_fp8_i8.safetensors";
        Vector fp8_vals{0.0, 1.0, -1.5, 448.0, 500.0, -500.0, 0.001953125, -0.0625};
        Vector i8_vals{-128.0, -200.0, -3.0, 0.0, 5.0, 127.0, 200.0};
        std::map<std::string, TensorSpec> specs;
        specs["fp8_vec"] = TensorSpec::from_vector(fp8_vals, "F8_E4M3");
        specs["i8_vec"] = TensorSpec::from_vector(i8_vals, "I8");
        save_safetensors(st_path.string(), specs, {{"quant", "phase4"}});

        auto [loaded, meta] = load_safetensors(st_path.string());
        expect_true(meta.at("quant") == "phase4", "fp8/i8 meta");
        const Vector fp8_out = loaded.at("fp8_vec").to_vector();
        expect_near(fp8_out[0], 0.0, 1e-9, "fp8 zero");
        expect_near(fp8_out[1], 1.0, 1e-9, "fp8 1.0");
        expect_near(fp8_out[2], -1.5, 1e-9, "fp8 -1.5");
        expect_near(fp8_out[3], 448.0, 1e-9, "fp8 max 448");
        expect_near(fp8_out[4], 448.0, 1e-9, "fp8 sat +500 -> 448");
        expect_near(fp8_out[5], -448.0, 1e-9, "fp8 sat -500 -> -448");
        expect_near(fp8_out[6], 0.001953125, 1e-9, "fp8 subnormal");
        expect_near(fp8_out[7], -0.0625, 1e-9, "fp8 -0.0625");

        const Vector i8_out = loaded.at("i8_vec").to_vector();
        expect_near(i8_out[0], -128.0, 1e-9, "i8 -128");
        expect_near(i8_out[1], -128.0, 1e-9, "i8 clamp -200 -> -128");
        expect_near(i8_out[2], -3.0, 1e-9, "i8 -3");
        expect_near(i8_out[3], 0.0, 1e-9, "i8 0");
        expect_near(i8_out[4], 5.0, 1e-9, "i8 5");
        expect_near(i8_out[5], 127.0, 1e-9, "i8 127");
        expect_near(i8_out[6], 127.0, 1e-9, "i8 clamp 200 -> 127");

        MmapSafetensors mm(st_path.string());
        expect_true(mm.tensor_dtype("fp8_vec") == "F8_E4M3", "mmap fp8 dtype");
        expect_true(mm.tensor_dtype("i8_vec") == "I8", "mmap i8 dtype");
        expect_true(mm.raw_tensor_bytes("fp8_vec").size() == fp8_vals.size(), "fp8 1 byte per elem");
        expect_true(mm.raw_tensor_bytes("i8_vec").size() == i8_vals.size(), "i8 1 byte per elem");
        expect_near(mm.read_tensor("fp8_vec").to_vector()[4], 448.0, 1e-9, "mmap fp8 read");
        expect_near(mm.read_tensor("i8_vec").to_vector()[1], -128.0, 1e-9, "mmap i8 read");

        std::filesystem::remove(st_path);
    }

    {
        ModelConfig cfg;
        cfg.num_layers = 2;
        cfg.hidden_size = 16;
        cfg.intermediate_size = 32;
        cfg.num_attention_heads = 2;
        cfg.num_key_value_heads = 2;
        cfg.vocab_size = 24;
        cfg.attention_type = "mha";
        cfg.torch_dtype = "fp8";

        const std::filesystem::path fp8_ckpt =
            std::filesystem::temp_directory_path() / "aic_test_fp8_ckpt";
        std::filesystem::remove_all(fp8_ckpt);
        TransformerModel m_fp8 = TransformerModel::create_synthetic(cfg, 19);
        expect_true(ModelConfig::from_json(m_fp8.config.to_json()).torch_dtype == "F8_E4M3", "fp8 config norm");
        m_fp8.save_checkpoint(fp8_ckpt.string());
        TransformerModel m_fp8_loaded = TransformerModel::load_checkpoint(fp8_ckpt.string());
        expect_true(m_fp8_loaded.config.torch_dtype == "F8_E4M3", "fp8 ckpt dtype");
        std::filesystem::remove_all(fp8_ckpt);
    }

    {
        ModelConfig cfg;
        cfg.num_layers = 2;
        cfg.hidden_size = 16;
        cfg.intermediate_size = 32;
        cfg.num_attention_heads = 2;
        cfg.num_key_value_heads = 2;
        cfg.vocab_size = 24;
        cfg.attention_type = "mha";
        TransformerModel base = TransformerModel::create_synthetic(cfg, 33);

        const std::filesystem::path tmp_lora =
            std::filesystem::temp_directory_path() / "aic_test_lora_dir";
        const std::filesystem::path base_dir = tmp_lora / "base";
        const std::filesystem::path lora_dir = tmp_lora / "adapter";
        std::filesystem::remove_all(tmp_lora);
        base.save_checkpoint(base_dir.string());
        std::filesystem::create_directories(lora_dir);

        const int r_lora = 4;
        const double alpha = 8.0;
        const double scale = alpha / static_cast<double>(r_lora);

        Matrix A0(static_cast<std::size_t>(r_lora), 32, 0.0);
        Matrix B0(16, static_cast<std::size_t>(r_lora), 0.0);
        Matrix A1(static_cast<std::size_t>(r_lora), 16, 0.0);
        Matrix B1(16, static_cast<std::size_t>(r_lora), 0.0);
        for (std::size_t i = 0; i < A0.data.size(); ++i) {
            A0.data[i] = 0.05 * static_cast<double>(static_cast<int>(i % 7) - 3);
        }
        for (std::size_t i = 0; i < B0.data.size(); ++i) {
            B0.data[i] = 0.04 * static_cast<double>(static_cast<int>(i % 5) - 2);
        }
        for (std::size_t i = 0; i < A1.data.size(); ++i) {
            A1.data[i] = 0.03 * static_cast<double>(static_cast<int>(i % 9) - 4);
        }
        for (std::size_t i = 0; i < B1.data.size(); ++i) {
            B1.data[i] = 0.06 * static_cast<double>(static_cast<int>(i % 3) - 1);
        }

        {
            JsonValue acfg = JsonValue::object();
            acfg["peft_type"] = "LORA";
            acfg["r"] = r_lora;
            acfg["lora_alpha"] = alpha;
            std::ofstream ofs(lora_dir / "adapter_config.json");
            ofs << dump_json(acfg, 2) << '\n';
        }

        std::map<std::string, TensorSpec> lora_specs;
        lora_specs["base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"] =
            TensorSpec::from_matrix(A0, "F32");
        lora_specs["base_model.model.model.layers.0.mlp.down_proj.lora_B.weight"] =
            TensorSpec::from_matrix(B0, "F32");
        lora_specs["model.layers.1.self_attn.q_proj.lora_A.weight"] =
            TensorSpec::from_matrix(A1, "F32");
        lora_specs["model.layers.1.self_attn.q_proj.lora_B.weight"] =
            TensorSpec::from_matrix(B1, "F32");
        save_safetensors((lora_dir / "adapter_model.safetensors").string(), lora_specs);

        TransformerModel expected = TransformerModel::load_checkpoint(base_dir.string());
        expected.matrices["model.layers.0.mlp.down_proj.weight"] = mat_add(
            expected.matrices["model.layers.0.mlp.down_proj.weight"],
            mat_scale(mat_mul(B0, A0), scale));
        expected.matrices["model.layers.1.self_attn.q_proj.weight"] = mat_add(
            expected.matrices["model.layers.1.self_attn.q_proj.weight"],
            mat_scale(mat_mul(B1, A1), scale));

        TransformerModel loaded = TransformerModel::load_with_lora(base_dir.string(), lora_dir.string());
        expect_near(
            frob_norm(mat_sub(
                loaded.matrices.at("model.layers.0.mlp.down_proj.weight"),
                expected.matrices.at("model.layers.0.mlp.down_proj.weight"))),
            0.0,
            1e-5,
            "lora merge mlp.down_proj");
        expect_near(
            frob_norm(mat_sub(
                loaded.matrices.at("model.layers.1.self_attn.q_proj.weight"),
                expected.matrices.at("model.layers.1.self_attn.q_proj.weight"))),
            0.0,
            1e-5,
            "lora merge self_attn.q_proj");

        ModelDivergenceStats div = compute_model_divergence(expected, loaded, {{1, 3, 5}});
        expect_near(div.max_logit_diff, 0.0, 1e-5, "lora merged forward equivalence");

        TransformerModel scaled0 = TransformerModel::load_with_lora(
            base_dir.string(), lora_dir.string(), 0.0);
        TransformerModel base_reloaded = TransformerModel::load_checkpoint(base_dir.string());
        expect_near(
            frob_norm(mat_sub(
                scaled0.matrices.at("model.layers.0.mlp.down_proj.weight"),
                base_reloaded.matrices.at("model.layers.0.mlp.down_proj.weight"))),
            0.0,
            1e-7,
            "lora scale_override=0 leaves base unchanged");

        std::filesystem::remove_all(tmp_lora);
    }
}

void test_int4_selective_step_traced_and_cov() {
    using namespace aic;

    {
        const std::filesystem::path st_path =
            std::filesystem::temp_directory_path() / "aic_test_int4.safetensors";

        Vector odd_vec{-7.0, -3.5, 0.0, 1.0, 2.5, 5.2, 7.0};
        Vector even_vec{-8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 6.0, 7.0};

        std::mt19937_64 rng(9001);
        std::normal_distribution<double> dist(0.0, 1.0);
        Matrix norm_mat(16, 24);
        for (double& x : norm_mat.data) {
            x = dist(rng);
        }

        std::map<std::string, TensorSpec> specs;
        specs["odd_i4"] = TensorSpec::from_vector(odd_vec, "INT4");
        specs["even_i4"] = TensorSpec::from_vector(even_vec, "I4");
        specs["norm_i4"] = TensorSpec::from_matrix(norm_mat, "I4");
        specs["keep_f32"] = TensorSpec::from_vector({1.25, -2.5, 3.75}, "F32");
        specs["drop_i4"] = TensorSpec::from_vector({-1.0, 2.0, -3.0, 4.0}, "I4");

        save_safetensors(st_path.string(), specs, {{"mode", "i4_test"}});

        auto [loaded, meta] = load_safetensors(st_path.string());
        expect_true(loaded.at("odd_i4").dtype == "I4", "INT4 normalizes to I4");
        expect_true(meta.contains("__scale__odd_i4"), "scale metadata present for odd_i4");
        expect_true(meta.contains("__scale__norm_i4"), "scale metadata present for norm_i4");

        const Vector odd_rt = loaded.at("odd_i4").to_vector();
        for (std::size_t i = 0; i < odd_vec.size(); ++i) {
            expect_near(odd_rt[i], odd_vec[i], 0.6, "odd i4 quant accuracy");
        }
        const Matrix norm_rt = loaded.at("norm_i4").to_matrix();
        const double rel_err = frob_norm(mat_sub(norm_rt, norm_mat)) / frob_norm(norm_mat);
        expect_true(rel_err < 0.16, "I4 relative error < 0.16 on normal weights");

        {
            MmapSafetensors mm(st_path.string());
            expect_true(mm.raw_tensor_bytes("odd_i4").size() == 4, "odd 7 elems -> 4 bytes");
            expect_true(mm.raw_tensor_bytes("even_i4").size() == 4, "even 8 elems -> 4 bytes");
            expect_true(mm.raw_tensor_bytes("norm_i4").size() == 192, "384 elems -> 192 bytes");
            const Matrix mm_norm = mm.read_tensor("norm_i4").to_matrix();
            expect_near(frob_norm(mat_sub(mm_norm, norm_rt)), 0.0, 1e-15, "mmap read_tensor matches load_safetensors");
        }

        const std::filesystem::path st_dst =
            std::filesystem::temp_directory_path() / "aic_test_int4_rewritten.safetensors";
        std::map<std::string, TensorSpec> replacements;
        replacements["odd_i4"] = TensorSpec::from_vector({10.0, 20.0, 30.0}, "F32");
        replacements["keep_f32"] = TensorSpec::from_vector({-7.0, 0.0, 7.0}, "I4");
        std::map<std::string, TensorSpec> additions;
        additions["added_i4"] = TensorSpec::from_vector({-14.0, 0.0, 14.0}, "INT4");
        std::set<std::string> removals{"drop_i4"};

        std::int64_t byte_delta = 0;
        rewrite_safetensors_selective(
            st_path.string(), st_dst.string(), replacements, additions, removals, &byte_delta);

        const std::int64_t expected_delta =
            (12 - 4) + (2 - 12) + 2 - 2;
        expect_true(byte_delta == expected_delta, "selective rewrite byte_delta");

        auto [rw_loaded, rw_meta] = load_safetensors(st_dst.string());
        expect_true(rw_loaded.contains("added_i4"), "selective added_i4 present");
        expect_true(!rw_loaded.contains("drop_i4"), "selective drop_i4 removed");
        expect_true(!rw_meta.contains("__scale__drop_i4"), "stale drop_i4 scale erased");
        expect_true(!rw_meta.contains("__scale__odd_i4"), "stale odd_i4 scale erased after F32 replace");
        expect_true(rw_meta.contains("__scale__keep_f32"), "keep_f32 gained I4 scale");
        expect_true(rw_meta.contains("__scale__added_i4"), "added_i4 has I4 scale");
        expect_true(rw_meta.contains("__scale__even_i4"), "untouched even_i4 scale kept");
        expect_near(rw_loaded.at("odd_i4").to_vector()[1], 20.0, 1e-6, "selective F32 replacement");
        expect_near(rw_loaded.at("added_i4").to_vector()[2], 14.0, 1e-6, "selective I4 addition");

        std::int64_t inplace_delta = 999;
        rewrite_safetensors_selective(
            st_dst.string(), st_dst.string(), {}, {}, {"added_i4"}, &inplace_delta);
        expect_true(inplace_delta == -2, "in-place selective rewrite byte_delta");
        auto [ip_loaded, ip_meta] = load_safetensors(st_dst.string());
        expect_true(!ip_loaded.contains("added_i4"), "in-place removal succeeded");
        expect_true(!ip_meta.contains("__scale__added_i4"), "in-place removal erased scale");

        std::filesystem::remove(st_path);
        std::filesystem::remove(st_dst);
    }

    {
        const std::filesystem::path i4_ckpt =
            std::filesystem::temp_directory_path() / "aic_test_i4_ckpt";
        std::filesystem::remove_all(i4_ckpt);

        ModelConfig cfg;
        cfg.num_layers = 2;
        cfg.hidden_size = 16;
        cfg.intermediate_size = 32;
        cfg.num_attention_heads = 2;
        cfg.num_key_value_heads = 2;
        cfg.vocab_size = 24;
        cfg.attention_type = "mha";
        cfg.torch_dtype = "INT4";

        TransformerModel m_i4 = TransformerModel::create_synthetic(cfg, 51);
        expect_true(m_i4.config.torch_dtype == "INT4", "int4 input config");
        m_i4.save_checkpoint(i4_ckpt.string());

        std::map<std::string, TensorSpec> shard_specs;
        for (const auto& [k, m] : m_i4.matrices) {
            shard_specs[k] = TensorSpec::from_matrix(m, "INT4");
        }
        for (const auto& [k, v] : m_i4.vectors) {
            shard_specs[k] = TensorSpec::from_vector(v, "INT4");
        }
        save_safetensors_sharded(i4_ckpt.string(), shard_specs, 2048);

        TransformerModel reloaded = TransformerModel::load_checkpoint(i4_ckpt.string());
        expect_true(reloaded.config.torch_dtype == "I4", "sharded I4 checkpoint torch_dtype");
        TensorSpec single = load_single_tensor_sharded(i4_ckpt.string(), "model.layers.0.mlp.down_proj.weight");
        expect_true(single.dtype == "I4", "single sharded I4 dtype");
        expect_near(
            frob_norm(mat_sub(single.to_matrix(), reloaded.matrices.at("model.layers.0.mlp.down_proj.weight"))),
            0.0,
            1e-12,
            "sharded I4 single tensor parity");

        std::filesystem::remove_all(i4_ckpt);
    }

    {
        ModelConfig cfg;
        cfg.num_layers = 3;
        cfg.hidden_size = 24;
        cfg.intermediate_size = 48;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 2;
        cfg.vocab_size = 32;
        cfg.attention_type = "gqa";
        cfg.use_rope = true;
        cfg.qk_norm = true;

        TransformerModel model = TransformerModel::create_synthetic(cfg, 88);
        const std::vector<int> seq{1, 4, 9, 13};
        auto [full_logits, full_trace] = model.forward(seq, {}, true);

        KVCache cache_traced = model.init_kv_cache();
        KVCache cache_plain = model.init_kv_cache();
        for (std::size_t t = 0; t < seq.size(); ++t) {
            ForwardStepTrace st = model.forward_step_traced(seq[t], cache_traced);
            Vector plain_logits = model.forward_step(seq[t], cache_plain);

            expect_near(norm2(vec_sub(st.logits, plain_logits)), 0.0, 1e-12, "traced vs plain step logits");
            expect_near(norm2(vec_sub(st.logits, full_logits[t])), 0.0, 1e-11, "traced step vs full logits");
            expect_true(st.residual.size() == static_cast<std::size_t>(cfg.num_layers) + 1, "step residual len");
            expect_true(st.mlp_poles.size() == static_cast<std::size_t>(cfg.num_layers), "step mlp_poles len");

            for (int l = 0; l <= cfg.num_layers; ++l) {
                const std::size_t ul = static_cast<std::size_t>(l);
                expect_near(
                    norm2(vec_sub(st.residual[ul], full_trace.residual[ul][t])),
                    0.0,
                    1e-11,
                    "step residual parity");
            }
            for (int l = 0; l < cfg.num_layers; ++l) {
                const std::size_t ul = static_cast<std::size_t>(l);
                const std::size_t d_ff = static_cast<std::size_t>(cfg.intermediate_size);
                expect_true(st.mlp_poles[ul].act.size() == d_ff, "step pole width");
                for (std::size_t i = 0; i < d_ff; ++i) {
                    expect_near(
                        st.mlp_poles[ul].gate_pre[i],
                        full_trace.mlp_poles[ul][i].gate_pre[t],
                        1e-11,
                        "step pole gate_pre");
                    expect_near(
                        st.mlp_poles[ul].up_val[i],
                        full_trace.mlp_poles[ul][i].up_val[t],
                        1e-11,
                        "step pole up_val");
                    expect_near(
                        st.mlp_poles[ul].act[i],
                        full_trace.mlp_poles[ul][i].act[t],
                        1e-11,
                        "step pole act");
                }
            }
        }

        const std::vector<Matrix> empty_covs = compute_mlp_key_covariances(model, {}, 0.25);
        expect_true(empty_covs.size() == static_cast<std::size_t>(cfg.num_layers), "empty cov count");
        expect_near(
            frob_norm(mat_sub(empty_covs[0], mat_scale(eye(48), 4.0))),
            0.0,
            1e-12,
            "empty cov inverse == (1/lambda) * I");

        const std::vector<std::vector<int>> cal{{1, 4, 9}, {2, 5}};
        const double ridge = 1e-2;
        const std::vector<Matrix> covs_inv = compute_mlp_key_covariances(model, cal, ridge);
        expect_true(covs_inv.size() == static_cast<std::size_t>(cfg.num_layers), "cov count");

        for (std::size_t l = 0; l < covs_inv.size(); ++l) {
            const Matrix& ci = covs_inv[l];
            expect_true(ci.rows == 48 && ci.cols == 48, "cov shape");
            expect_near(frob_norm(mat_sub(ci, transpose(ci))), 0.0, 1e-14, "cov symmetry");
            Vector probe(48, 0.0);
            probe[l % 48] = 1.0;
            probe[(l * 7 + 3) % 48] = -0.5;
            expect_true(dot(probe, mat_vec(ci, probe)) > 0.0, "cov positive definite");
        }

        Matrix manual_c = mat_scale(eye(48), ridge);
        std::size_t n_toks = 0;
        for (const auto& s : cal) {
            auto [lg, tr] = model.forward(s, {}, true);
            (void)lg;
            for (std::size_t t = 0; t < s.size(); ++t) {
                Vector k(48);
                for (std::size_t i = 0; i < 48; ++i) {
                    k[i] = tr.mlp_poles[0][i].act[t];
                }
                manual_c = mat_add(manual_c, mat_scale(outer(k, k), 1.0));
                ++n_toks;
            }
        }
        for (std::size_t r = 0; r < 48; ++r) {
            for (std::size_t c = 0; c < 48; ++c) {
                if (r != c) {
                    manual_c(r, c) /= static_cast<double>(n_toks);
                } else {
                    manual_c(r, c) = ((manual_c(r, c) - ridge) / static_cast<double>(n_toks)) + ridge;
                }
            }
        }
        expect_near(
            frob_norm(mat_sub(mat_mul(manual_c, covs_inv[0]), eye(48))),
            0.0,
            1e-9,
            "compute_mlp_key_covariances exact matrix inverse");
    }
}

void test_multi_rhs_fisher_kernel_and_diagonals() {
    using namespace aic;

    {
        const std::size_t n = 24;
        const std::size_t m = 5;
        std::mt19937_64 rng(6001);
        std::normal_distribution<double> dist(0.0, 1.0);
        Matrix A(n, n);
        Matrix B(n, m);
        for (double& x : A.data) {
            x = dist(rng);
        }
        for (std::size_t i = 0; i < n; ++i) {
            A(i, i) += 4.0;
        }
        for (double& x : B.data) {
            x = dist(rng);
        }
        const Matrix X = solve_linear_multi(A, B);
        expect_true(X.rows == n && X.cols == m, "solve_linear_multi shape");
        const double resid = frob_norm(mat_sub(mat_mul(A, X), B));
        expect_near(resid, 0.0, 1e-10, "solve_linear_multi residual");

        Matrix rank_def = outer({1.0, 2.0, 3.0}, {1.0, 2.0, 3.0});
        Matrix rhs_rd(3, 2, 0.0);
        rhs_rd(0, 0) = 1.0;
        rhs_rd(1, 0) = 2.0;
        rhs_rd(2, 0) = 3.0;
        rhs_rd(0, 1) = -1.0;
        rhs_rd(1, 1) = -2.0;
        rhs_rd(2, 1) = -3.0;
        const Matrix x_rd = solve_linear_multi(rank_def, rhs_rd);
        expect_true(x_rd.rows == 3 && x_rd.cols == 2, "solve_linear_multi damped shape");
        expect_near(frob_norm(mat_sub(mat_mul(rank_def, x_rd), rhs_rd)), 0.0, 1e-6, "solve_linear_multi damped fit");
    }

    {
        std::mt19937_64 rng(6002);
        std::normal_distribution<double> dist(0.0, 1.0);
        std::uniform_real_distribution<double> pos_dist(0.1, 2.5);

        for (const auto& [rows, cols] : std::vector<std::pair<std::size_t, std::size_t>>{
                 {16, 20},
                 {70, 70}}) {
            Matrix A(rows, cols);
            Matrix B(rows, cols);
            Vector g_out(rows);
            Vector g_in(cols);
            for (double& x : A.data) {
                x = dist(rng);
            }
            for (double& x : B.data) {
                x = dist(rng);
            }
            for (double& x : g_out) {
                x = pos_dist(rng);
            }
            for (double& x : g_in) {
                x = pos_dist(rng);
            }
            double ref = 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    ref += g_out[r] * g_in[c] * A(r, c) * B(r, c);
                }
            }
            const double via_ctx = MetalContext::instance().fisher_inner_f32(A, B, g_out, g_in);
            const double via_fn = metal_fisher_inner(A, B, g_out, g_in);
            const double tol = 1e-3 * std::max(1.0, std::abs(ref));
            expect_near(via_ctx, ref, tol, "MetalContext::fisher_inner_f32 parity");
            expect_near(via_fn, ref, tol, "metal_fisher_inner parity");
        }
    }

    {
        ModelConfig cfg;
        cfg.num_layers = 4;
        cfg.hidden_size = 24;
        cfg.intermediate_size = 48;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 2;
        cfg.vocab_size = 32;
        cfg.attention_type = "gqa";
        cfg.use_rope = true;
        cfg.qk_norm = true;

        TransformerModel model = TransformerModel::create_synthetic(cfg, 606);

        const std::vector<LayerFisherDiag> empty_diag = compute_fisher_diagonals(model, {});
        expect_true(empty_diag.size() == 4, "empty fisher diag count");
        for (const auto& d : empty_diag) {
            expect_true(d.g_in.size() == 24 && d.g_out.size() == 24, "empty fisher diag dim");
            for (std::size_t i = 0; i < 24; ++i) {
                expect_near(d.g_in[i], 1.0, 1e-12, "empty g_in == 1");
                expect_near(d.g_out[i], 1.0, 1e-12, "empty g_out == 1");
            }
        }

        const std::vector<std::vector<int>> cal{{1, 5, 9, 13}, {2, 7, 11}};
        const double eps_floor = 1e-4;
        const std::vector<LayerFisherDiag> diag_a = compute_fisher_diagonals(model, cal, eps_floor);
        const std::vector<LayerFisherDiag> diag_b = compute_fisher_diagonals(model, cal, eps_floor);
        expect_true(diag_a.size() == 4, "fisher diag count");

        for (std::size_t l = 0; l < diag_a.size(); ++l) {
            double sum_in = 0.0;
            double sum_out = 0.0;
            double min_in = diag_a[l].g_in[0];
            double max_in = diag_a[l].g_in[0];
            double min_out = diag_a[l].g_out[0];
            double max_out = diag_a[l].g_out[0];
            for (std::size_t i = 0; i < 24; ++i) {
                expect_near(diag_a[l].g_in[i], diag_b[l].g_in[i], 1e-14, "fisher g_in determinism");
                expect_near(diag_a[l].g_out[i], diag_b[l].g_out[i], 1e-14, "fisher g_out determinism");
                expect_true(diag_a[l].g_in[i] >= eps_floor, "fisher g_in positive");
                expect_true(diag_a[l].g_out[i] >= eps_floor, "fisher g_out positive");
                sum_in += diag_a[l].g_in[i];
                sum_out += diag_a[l].g_out[i];
                min_in = std::min(min_in, diag_a[l].g_in[i]);
                max_in = std::max(max_in, diag_a[l].g_in[i]);
                min_out = std::min(min_out, diag_a[l].g_out[i]);
                max_out = std::max(max_out, diag_a[l].g_out[i]);
            }
            expect_near(sum_in / 24.0, 1.0, 1e-3, "fisher g_in unit mean");
            expect_near(sum_out / 24.0, 1.0, 1e-3, "fisher g_out unit mean");
            expect_true(max_in - min_in > 1e-3, "fisher g_in non-uniform");
            expect_true(max_out - min_out > 1e-3, "fisher g_out non-uniform");
        }
    }
}

}

int main() {
    test_basic_tensor_ops();
    test_svd_and_procrustes();
    test_metal_accel();
    test_json_min();
    test_safetensors_and_mmap_sharded();
    test_runtime_and_checkpoints();
    test_kv_cache_sampling_and_divergence();
    test_randomized_svd_fp8_i8_and_lora();
    test_int4_selective_step_traced_and_cov();
    test_multi_rhs_fisher_kernel_and_diagonals();
    std::cout << "ALL test_tensor_runtime checks passed (10/10 test groups)\n";
    return 0;
}
