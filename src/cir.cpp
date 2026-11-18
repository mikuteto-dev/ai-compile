#include "aic/cir.hpp"

#include "aic/errors.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <thread>

namespace aic {

namespace {

constexpr double kPi = 3.14159265358979323846;

template <typename Fn>
void parallel_for_range(std::size_t count, std::size_t min_per_thread, Fn&& fn) {
    if (count == 0) {
        return;
    }
    const unsigned int hw = std::thread::hardware_concurrency();
    const std::size_t max_threads = (hw > 0u) ? static_cast<std::size_t>(hw) : 4u;
    const std::size_t eff_min = std::max<std::size_t>(1, min_per_thread);
    const std::size_t wanted = (count + eff_min - 1) / eff_min;
    const std::size_t num_threads = std::min({count, max_threads, wanted});
    if (num_threads <= 1) {
        fn(0, count);
        return;
    }

    std::vector<std::exception_ptr> errors(num_threads);
    std::vector<std::thread> workers;
    workers.reserve(num_threads - 1);
    const std::size_t chunk = (count + num_threads - 1) / num_threads;

    for (std::size_t t = 1; t < num_threads; ++t) {
        const std::size_t begin = t * chunk;
        const std::size_t end = std::min(count, begin + chunk);
        if (begin >= end) {
            continue;
        }
        workers.emplace_back([&, t, begin, end]() {
            try {
                fn(begin, end);
            } catch (...) {
                errors[t] = std::current_exception();
            }
        });
    }

    try {
        fn(0, std::min(count, chunk));
    } catch (...) {
        errors[0] = std::current_exception();
    }

    for (auto& w : workers) {
        w.join();
    }
    for (const auto& ep : errors) {
        if (ep) {
            std::rethrow_exception(ep);
        }
    }
}

std::uint64_t hash_str(const std::string& s) {
    std::uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h ^ (h >> 31);
}

Vector seeded_unit_vec(const std::string& tag, std::size_t d) {
    if (d == 0) {
        return {};
    }
    std::mt19937_64 rng(hash_str(tag));
    std::normal_distribution<double> dist(0.0, 1.0);
    Vector v(d);
    for (std::size_t i = 0; i < d; ++i) {
        v[i] = dist(rng);
    }
    Vector out = normalize(v);
    if (norm2(out) < 1e-12) {
        out.assign(d, 0.0);
        out[0] = 1.0;
    }
    return out;
}

std::string stage_to_label(const MIRStageSpec& s) {
    if (s.end.empty()) {
        return s.start;
    }
    return s.start + (s.inclusive ? "..=" : "..") + s.end;
}

std::string arg_as_string(const std::map<std::string, ArgValue>& args, const std::string& key, const std::string& def = "") {
    auto it = args.find(key);
    if (it == args.end()) {
        return def;
    }
    if (const auto* s = std::get_if<std::string>(&it->second)) {
        return *s;
    }
    if (const auto* d = std::get_if<double>(&it->second)) {
        return std::to_string(*d);
    }
    if (const auto* i = std::get_if<std::int64_t>(&it->second)) {
        return std::to_string(*i);
    }
    return def;
}

std::optional<double> extract_numeric_arg(const std::map<std::string, ArgValue>& args, const std::string& key) {
    auto it = args.find(key);
    if (it == args.end()) {
        return std::nullopt;
    }
    if (const auto* d = std::get_if<double>(&it->second)) {
        return *d;
    }
    if (const auto* i = std::get_if<std::int64_t>(&it->second)) {
        return static_cast<double>(*i);
    }
    return std::nullopt;
}

bool is_t2_topology(const std::string& topo) {
    return topo == "T2" || topo == "Torus" || topo == "T<2>";
}

Matrix modes_to_matrix(const std::vector<SVDMode>& modes, std::size_t d) {
    Matrix m = zeros(d, d);
    for (const auto& mode : modes) {
        if (mode.u.size() != d || mode.v.size() != d) {
            continue;
        }
        for (std::size_t r = 0; r < d; ++r) {
            const double su = mode.sigma * mode.u[r];
            for (std::size_t c = 0; c < d; ++c) {
                m(r, c) += su * mode.v[c];
            }
        }
    }
    return m;
}

std::vector<SVDMode> get_symbol_field_modes(
    const std::string& name,
    const NeuralAtlas& atlas,
    std::size_t d,
    const std::map<std::string, std::vector<SVDMode>>& field_env) {
    auto env_it = field_env.find(name);
    if (env_it != field_env.end()) {
        return env_it->second;
    }
    auto sym_it = atlas.symbols.find(name);
    if (sym_it != atlas.symbols.end() && !sym_it->second.field_modes.empty()) {
        return sym_it->second.field_modes;
    }
    if (d == 0) {
        return {};
    }
    Vector u = seeded_unit_vec("field_u:" + name, d);
    Vector v = seeded_unit_vec("field_v:" + name, d);
    return {SVDMode{1.0, std::move(u), std::move(v)}};
}

bool is_field_name(
    const std::string& name,
    const MIRModule& mir,
    const NeuralAtlas& atlas,
    const std::map<std::string, std::vector<SVDMode>>& field_env) {
    if (field_env.contains(name)) {
        return true;
    }
    auto mit = mir.declarations.find(name);
    if (mit != mir.declarations.end() && mit->second.kind == "field") {
        return true;
    }
    auto ait = atlas.symbols.find(name);
    if (ait != atlas.symbols.end() && ait->second.kind == "field") {
        return true;
    }
    return false;
}

std::string to_lower_str(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool resolve_field_use_fisher(
    const std::string& target,
    const std::map<std::string, ArgValue>& args,
    const MIRModule& mir,
    const std::map<std::string, bool>& field_metric_env) {
    const std::string arg_metric = to_lower_str(arg_as_string(args, "metric", ""));
    if (arg_metric == "euclidean") {
        return false;
    }
    if (arg_metric == "fisher") {
        return true;
    }
    auto env_it = field_metric_env.find(target);
    if (env_it != field_metric_env.end()) {
        return env_it->second;
    }
    auto mit = mir.declarations.find(target);
    if (mit != mir.declarations.end()) {
        if (to_lower_str(mit->second.metric) == "euclidean") {
            return false;
        }
    }
    return true;
}

std::pair<Vector, Vector> resolve_flow_fisher_metric(
    int l0,
    int l1,
    std::size_t d,
    const NeuralAtlas& atlas,
    bool use_fisher) {
    if (!use_fisher || d == 0 || atlas.fisher_metrics.empty()) {
        return {Vector(d, 1.0), Vector(d, 1.0)};
    }
    Vector g_out(d, 0.0);
    Vector g_in(d, 0.0);
    int count = 0;
    for (int l = l0; l < l1; ++l) {
        auto it = atlas.fisher_metrics.find(l);
        if (it != atlas.fisher_metrics.end() &&
            it->second.g_out.size() == d &&
            it->second.g_in.size() == d) {
            for (std::size_t i = 0; i < d; ++i) {
                g_out[i] += it->second.g_out[i];
                g_in[i] += it->second.g_in[i];
            }
            ++count;
        }
    }
    if (count == 0) {
        return {Vector(d, 1.0), Vector(d, 1.0)};
    }
    const double inv = 1.0 / static_cast<double>(count);
    for (std::size_t i = 0; i < d; ++i) {
        g_out[i] = std::max(1e-4, g_out[i] * inv);
        g_in[i] = std::max(1e-4, g_in[i] * inv);
    }
    return {std::move(g_out), std::move(g_in)};
}

SymbolBinding resolve_symbol_binding(
    const std::string& name,
    const MIRModule& mir,
    const NeuralAtlas& atlas,
    const ModelConfig& cfg) {
    auto it = atlas.symbols.find(name);
    if (it != atlas.symbols.end()) {
        return it->second;
    }
    SymbolBinding sb;
    sb.name = name;
    sb.kind = "pole";
    const std::size_t d = static_cast<std::size_t>(std::max(0, cfg.hidden_size));
    std::string topo = "Euclidean";
    auto mit = mir.declarations.find(name);
    if (mit != mir.declarations.end()) {
        sb.kind = mit->second.kind;
        topo = mit->second.topology;
        sb.layer_range = cfg.stage_to_layers(
            mit->second.frame.start,
            mit->second.frame.end,
            mit->second.frame.inclusive);
    } else {
        sb.layer_range = {0, std::max(1, cfg.num_layers)};
    }
    if (d > 0) {
        sb.direction = seeded_unit_vec("sym_dir:" + name, d);
        sb.subspace_basis.push_back(sb.direction);
        if (sb.kind == "manifold") {
            const std::size_t plane_cnt = is_t2_topology(topo) ? 5u : 3u;
            std::vector<Vector> raw;
            raw.reserve(plane_cnt);
            for (std::size_t idx = 0; idx < plane_cnt; ++idx) {
                raw.push_back(seeded_unit_vec("m" + std::to_string(idx) + ":" + name, d));
            }
            sb.manifold_plane = gram_schmidt(raw, 1e-10);
            sb.subspace_basis = sb.manifold_plane;
            if (sb.manifold_plane.size() > 1) {
                sb.direction = sb.manifold_plane[1];
            }
        }
    }
    return sb;
}

struct NullspaceOperators {
    bool active = false;
    Matrix p_in;
    Matrix p_out;
    std::vector<Vector> u_basis;
    std::vector<Vector> k_rows_norm;
};

NullspaceOperators build_nullspace_ops(
    const std::vector<std::string>& preserve_names,
    const MIRModule& mir,
    const NeuralAtlas& atlas,
    std::size_t d) {
    NullspaceOperators ops;
    if (d == 0 || preserve_names.empty()) {
        ops.p_in = eye(d);
        ops.p_out = eye(d);
        return ops;
    }

    std::vector<Vector> raw_in;
    std::vector<Vector> raw_out;
    for (const std::string& pname : preserve_names) {
        auto pit = atlas.preserve_directions.find(pname);
        if (pit != atlas.preserve_directions.end()) {
            for (const Vector& v : pit->second) {
                if (v.size() == d && norm2(v) > 1e-12) {
                    raw_in.push_back(v);
                    raw_out.push_back(v);
                }
            }
        }
        auto sit = atlas.symbols.find(pname);
        if (sit != atlas.symbols.end()) {
            const SymbolBinding& sb = sit->second;
            for (const Vector& v : sb.preserve_keys) {
                if (v.size() == d && norm2(v) > 1e-12) {
                    raw_in.push_back(v);
                }
            }
            for (const Vector& v : sb.subspace_basis) {
                if (v.size() == d && norm2(v) > 1e-12) {
                    raw_in.push_back(v);
                    raw_out.push_back(v);
                }
            }
            if (sb.subspace_basis.empty() && sb.direction.size() == d && norm2(sb.direction) > 1e-12) {
                raw_in.push_back(sb.direction);
                raw_out.push_back(sb.direction);
            }
            if (sb.kind == "field") {
                for (const auto& mode : sb.field_modes) {
                    if (mode.u.size() == d && norm2(mode.u) > 1e-12) {
                        raw_out.push_back(mode.u);
                    }
                    if (mode.v.size() == d && norm2(mode.v) > 1e-12) {
                        raw_in.push_back(mode.v);
                    }
                }
            }
        }
        (void)mir;
    }

    if (raw_in.empty() && raw_out.empty()) {
        ops.p_in = eye(d);
        ops.p_out = eye(d);
        return ops;
    }

    ops.active = true;
    ops.u_basis = gram_schmidt(raw_out, 1e-10);
    ops.p_out = eye(d);
    for (const Vector& q : ops.u_basis) {
        ops.p_out = mat_sub(ops.p_out, outer(q, q));
    }

    std::vector<Vector> k_gs = gram_schmidt(raw_in, 1e-10);
    if (k_gs.empty()) {
        ops.p_in = eye(d);
    } else {
        const std::size_t r_in = k_gs.size();
        Matrix k_mat(r_in, d, 0.0);
        for (std::size_t r = 0; r < r_in; ++r) {
            k_mat.set_row(r, k_gs[r]);
            ops.k_rows_norm.push_back(normalize(k_gs[r]));
        }
        const Matrix k_t = transpose(k_mat);
        Matrix gram = mat_mul(k_mat, k_t);
        const double lambda = 1e-11;
        for (std::size_t i = 0; i < r_in; ++i) {
            gram(i, i) += lambda;
        }
        const Matrix gram_inv = mat_inv(gram);
        ops.p_in = mat_sub(eye(d), mat_mul(k_t, mat_mul(gram_inv, k_mat)));
    }

    if (ops.u_basis.size() >= d || frob_norm(ops.p_out) < 1e-6 || frob_norm(ops.p_in) < 1e-6) {
        throw NullspaceExhaustedError("Preserved subspace covers full hidden dimension; nullspace collapsed");
    }

    return ops;
}

double compute_vec_crosstalk(const Vector& u, const Vector& v, const NullspaceOperators& ns) {
    double xt = 0.0;
    if (!u.empty() && norm2(u) > 1e-12) {
        const Vector u_n = normalize(u);
        for (const Vector& q : ns.u_basis) {
            xt = std::max(xt, std::abs(dot(u_n, q)));
        }
    }
    if (!v.empty() && norm2(v) > 1e-12) {
        const Vector v_n = normalize(v);
        for (const Vector& k : ns.k_rows_norm) {
            xt = std::max(xt, std::abs(dot(v_n, k)));
        }
    }
    return xt;
}

void project_and_check_op(
    CIROp& op,
    const NullspaceOperators& ns,
    double crosstalk_bound,
    const std::string& sym_topology,
    double sym_curvature,
    double approx_bound,
    double delta1,
    double delta2,
    const SymbolBinding* sb) {
    if (!ns.active) {
        return;
    }

    if (op.kind == "vpd_rank1" || op.kind == "swiglu_pole") {
        if (!op.output_dir.empty()) {
            const Vector u_proj = mat_vec(ns.p_out, op.output_dir);
            if (norm2(u_proj) < 1e-8) {
                throw NullspaceExhaustedError("Output direction collapsed inside preserved nullspace for " + op.symbol);
            }
            op.output_dir = normalize(u_proj);
        }
        if (!op.input_dir.empty()) {
            const Vector v_proj = mat_vec(ns.p_in, op.input_dir);
            if (norm2(v_proj) < 1e-8) {
                throw NullspaceExhaustedError("Input direction collapsed inside preserved nullspace for " + op.symbol);
            }
            op.input_dir = normalize(v_proj);
        }
        op.crosstalk = compute_vec_crosstalk(op.output_dir, op.input_dir, ns);
    } else if (op.kind == "swiglu_spline") {
        double max_xt = 0.0;
        double max_down_norm = 0.0;
        for (auto& slot : op.spline_slots) {
            slot.w_gate = mat_vec(ns.p_in, slot.w_gate);
            slot.w_up = mat_vec(ns.p_in, slot.w_up);
            slot.w_down = mat_vec(ns.p_out, slot.w_down);
            max_down_norm = std::max(max_down_norm, norm2(slot.w_down));
            max_xt = std::max(max_xt, compute_vec_crosstalk(slot.w_down, slot.w_up, ns));
        }
        if (max_down_norm < 1e-10) {
            throw NullspaceExhaustedError("Tessellated manifold spline collapsed inside preserved nullspace for " + op.symbol);
        }
        if (sb != nullptr && is_t2_topology(sym_topology) && sb->manifold_plane.size() >= 5) {
            op.approx_error = eval_t2_spline_error(
                op.spline_slots,
                sb->manifold_plane[0],
                sb->manifold_plane[1],
                sb->manifold_plane[2],
                sb->manifold_plane[3],
                sb->manifold_plane[4],
                delta1,
                delta2);
            if (op.approx_error > approx_bound + 1e-9) {
                throw ApproximationUnmetError("Projected T2 spline error exceeds approx contract bound");
            }
        } else if (sb != nullptr && sb->manifold_plane.size() >= 3 && (sym_curvature > 0.0 || sym_topology == "S1")) {
            op.approx_error = eval_s1_spline_error(
                op.spline_slots,
                sb->manifold_plane[0],
                sb->manifold_plane[1],
                sb->manifold_plane[2],
                delta1);
            if (op.approx_error > approx_bound + 1e-9) {
                throw ApproximationUnmetError("Projected S1 spline error exceeds approx contract bound");
            }
        }
        op.crosstalk = max_xt;
    } else if (op.kind == "attn_bilinear") {
        op.w_qk = mat_mul(mat_mul(ns.p_in, op.w_qk), ns.p_in);
        op.w_ov = mat_mul(mat_mul(ns.p_out, op.w_ov), ns.p_in);
        if (frob_norm(op.w_qk) < 1e-10 || frob_norm(op.w_ov) < 1e-10) {
            throw NullspaceExhaustedError("Attention bilinear projections collapsed inside preserved nullspace");
        }
        double xt = 0.0;
        const double qk_norm = std::max(1e-12, frob_norm(op.w_qk));
        const double ov_norm = std::max(1e-12, frob_norm(op.w_ov));
        for (const Vector& q : ns.u_basis) {
            xt = std::max(xt, norm2(mat_vec(transpose(op.w_ov), q)) / ov_norm);
        }
        for (const Vector& k : ns.k_rows_norm) {
            xt = std::max(xt, norm2(mat_vec(op.w_qk, k)) / qk_norm);
            xt = std::max(xt, norm2(mat_vec(op.w_ov, k)) / ov_norm);
        }
        op.crosstalk = xt;
    } else if (op.kind == "assoc_mem_row") {
        const std::size_t d_ns = ns.p_in.rows;
        if (!op.key_vec.empty() && op.key_vec.size() == d_ns) {
            const Vector k_proj = mat_vec(ns.p_in, op.key_vec);
            if (norm2(k_proj) < 1e-8) {
                throw NullspaceExhaustedError("Associative memory key collapsed inside preserved nullspace");
            }
            op.key_vec = normalize(k_proj);
        }
        if (!op.val_vec.empty() && norm2(op.val_vec) > 1e-8) {
            const double orig_norm = norm2(op.val_vec);
            const Vector v_proj = mat_vec(ns.p_out, op.val_vec);
            if (norm2(v_proj) < 1e-8) {
                throw NullspaceExhaustedError("Associative memory value collapsed inside preserved nullspace");
            }
            op.val_vec = vec_scale(normalize(v_proj), orig_norm);
        }
        const Vector k_for_xt = (op.key_vec.size() == d_ns) ? op.key_vec : Vector{};
        op.crosstalk = compute_vec_crosstalk(op.val_vec, k_for_xt, ns);
    } else if (op.kind == "frame_bridge") {
        const std::size_t d = op.bridge_matrix.rows;
        if (d > 0) {
            Matrix delta = mat_sub(op.bridge_matrix, eye(d));
            if (frob_norm(delta) > 1e-9) {
                delta = mat_mul(mat_mul(ns.p_out, delta), ns.p_in);
                op.bridge_matrix = mat_add(eye(d), delta);
                const double d_norm = std::max(1e-12, frob_norm(delta));
                double xt = 0.0;
                for (const Vector& q : ns.u_basis) {
                    xt = std::max(xt, norm2(mat_vec(transpose(delta), q)) / d_norm);
                }
                for (const Vector& k : ns.k_rows_norm) {
                    xt = std::max(xt, norm2(mat_vec(delta, k)) / d_norm);
                }
                op.crosstalk = xt;
            }
        }
    }

    if (op.crosstalk > crosstalk_bound + 1e-12) {
        throw CrosstalkExceededError(
            "Residual crosstalk " + std::to_string(op.crosstalk) +
            " exceeds bound " + std::to_string(crosstalk_bound));
    }
}
}

std::pair<std::vector<TessellatedSlot>, double> build_s1_swiglu_spline(
    const Vector& e0,
    const Vector& e1,
    const Vector& e2,
    int k_charts,
    double delta_angle,
    double beta) {
    const int k_eff = std::max(1, k_charts);
    const std::size_t s_cnt = static_cast<std::size_t>(3 * k_eff);
    const double beta_eff = (beta > 0.0) ? beta : 200.0;

    const double half_w = 1.1;
    const double tau = std::cos(half_w);
    const double inv_denom = 1.0 / (1.0 - tau);

    std::vector<double> theta_s(s_cnt);
    for (std::size_t s = 0; s < s_cnt; ++s) {
        theta_s[s] = (2.0 * kPi * static_cast<double>(s)) / static_cast<double>(s_cnt);
    }

    const std::size_t m_samples = std::max<std::size_t>(360, 8 * s_cnt);
    Matrix h_tilde(m_samples, s_cnt, 0.0);
    Vector y1(m_samples, 0.0);
    Vector y2(m_samples, 0.0);

    auto fill_samples = [&](std::size_t m_begin, std::size_t m_end) {
        for (std::size_t m = m_begin; m < m_end; ++m) {
            const double th = (2.0 * kPi * static_cast<double>(m)) / static_cast<double>(m_samples);
            y1[m] = std::cos(th + delta_angle) - std::cos(th);
            y2[m] = std::sin(th + delta_angle) - std::sin(th);
            for (std::size_t s = 0; s < s_cnt; ++s) {
                const double u = (std::cos(th - theta_s[s]) - tau) * inv_denom;
                h_tilde(m, s) = (silu(beta_eff * u) * u) / beta_eff;
            }
        }
    };

    if (m_samples * s_cnt >= 4096) {
        parallel_for_range(m_samples, 64, fill_samples);
    } else {
        fill_samples(0, m_samples);
    }

    Matrix gram(s_cnt, s_cnt, 0.0);
    Vector rhs1(s_cnt, 0.0);
    Vector rhs2(s_cnt, 0.0);

    auto build_gram_rows = [&](std::size_t r_begin, std::size_t r_end) {
        for (std::size_t r = r_begin; r < r_end; ++r) {
            long double acc_y1 = 0.0;
            long double acc_y2 = 0.0;
            for (std::size_t m = 0; m < m_samples; ++m) {
                const double hr = h_tilde(m, r);
                acc_y1 += static_cast<long double>(hr) * static_cast<long double>(y1[m]);
                acc_y2 += static_cast<long double>(hr) * static_cast<long double>(y2[m]);
            }
            rhs1[r] = static_cast<double>(acc_y1);
            rhs2[r] = static_cast<double>(acc_y2);

            for (std::size_t c = 0; c < s_cnt; ++c) {
                long double acc_g = 0.0;
                for (std::size_t m = 0; m < m_samples; ++m) {
                    acc_g += static_cast<long double>(h_tilde(m, r)) * static_cast<long double>(h_tilde(m, c));
                }
                gram(r, c) = static_cast<double>(acc_g);
            }
            gram(r, r) += 1e-11;
        }
    };

    if (s_cnt >= 24 && m_samples * s_cnt >= 4096) {
        parallel_for_range(s_cnt, 4, build_gram_rows);
    } else {
        build_gram_rows(0, s_cnt);
    }

    const Vector c1 = solve_linear(gram, std::move(rhs1));
    const Vector c2 = solve_linear(gram, std::move(rhs2));

    std::vector<TessellatedSlot> slots;
    slots.reserve(s_cnt);
    for (std::size_t s = 0; s < s_cnt; ++s) {
        const Vector c_vec = vec_add(vec_scale(e1, std::cos(theta_s[s])), vec_scale(e2, std::sin(theta_s[s])));
        const Vector z_s = vec_scale(vec_sub(c_vec, vec_scale(e0, tau)), inv_denom);
        TessellatedSlot slot;
        slot.chart_index = static_cast<int>(s / 3);
        slot.w_gate = vec_scale(z_s, beta_eff);
        slot.w_up = z_s;
        slot.w_down = vec_add(vec_scale(e1, c1[s] / beta_eff), vec_scale(e2, c2[s] / beta_eff));
        slots.push_back(std::move(slot));
    }

    const double max_err = eval_s1_spline_error(slots, e0, e1, e2, delta_angle, 360);
    return {std::move(slots), max_err};
}

double eval_s1_spline_error(
    const std::vector<TessellatedSlot>& slots,
    const Vector& e0,
    const Vector& e1,
    const Vector& e2,
    double delta_angle,
    std::size_t eval_points) {
    const std::size_t m_eval = std::max<std::size_t>(16, eval_points);
    const std::size_t d = e0.size();
    Vector per_sample_err(m_eval, 0.0);

    auto eval_range = [&](std::size_t m_begin, std::size_t m_end) {
        for (std::size_t m = m_begin; m < m_end; ++m) {
            const double th = (2.0 * kPi * (static_cast<double>(m) + 0.5)) / static_cast<double>(m_eval);
            const Vector x = vec_add(
                e0,
                vec_add(vec_scale(e1, std::cos(th)), vec_scale(e2, std::sin(th))));
            const Vector target = vec_add(
                vec_scale(e1, std::cos(th + delta_angle) - std::cos(th)),
                vec_scale(e2, std::sin(th + delta_angle) - std::sin(th)));
            Vector pred(d, 0.0);
            for (const auto& s : slots) {
                const double a = silu(dot(s.w_gate, x)) * dot(s.w_up, x);
                for (std::size_t i = 0; i < d; ++i) {
                    pred[i] += a * s.w_down[i];
                }
            }
            per_sample_err[m] = norm2(vec_sub(target, pred));
        }
    };

    if (m_eval * slots.size() >= 4096) {
        parallel_for_range(m_eval, 64, eval_range);
    } else {
        eval_range(0, m_eval);
    }

    double max_err = 0.0;
    for (double err : per_sample_err) {
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

std::pair<std::vector<TessellatedSlot>, double> build_t2_swiglu_spline(
    const Vector& e0,
    const Vector& c1,
    const Vector& s1,
    const Vector& c2,
    const Vector& s2,
    int k_per_axis,
    double dtheta1,
    double dtheta2,
    double beta) {
    const int k_eff = std::max(1, k_per_axis);
    const std::size_t s_cnt = static_cast<std::size_t>(k_eff * k_eff);
    const double beta_eff = (beta > 0.0) ? beta : 160.0;
    const double tau = 0.05;
    const double inv_denom = 1.0 / (1.0 - tau);

    std::vector<double> alpha_s(s_cnt);
    std::vector<double> beta_s(s_cnt);
    for (int i = 0; i < k_eff; ++i) {
        for (int j = 0; j < k_eff; ++j) {
            const std::size_t idx = static_cast<std::size_t>(i * k_eff + j);
            alpha_s[idx] = (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(k_eff);
            beta_s[idx] = (2.0 * kPi * static_cast<double>(j)) / static_cast<double>(k_eff);
        }
    }

    const int g = std::max(18, 3 * k_eff);
    const std::size_t m_samples = static_cast<std::size_t>(g * g);
    Matrix h_tilde(m_samples, s_cnt, 0.0);
    Vector y1c(m_samples, 0.0);
    Vector y1s(m_samples, 0.0);
    Vector y2c(m_samples, 0.0);
    Vector y2s(m_samples, 0.0);

    auto fill_samples = [&](std::size_t p_begin, std::size_t p_end) {
        for (std::size_t p = p_begin; p < p_end; ++p) {
            const double t1 = (2.0 * kPi * static_cast<double>(p)) / static_cast<double>(g);
            for (int q = 0; q < g; ++q) {
                const double t2 = (2.0 * kPi * static_cast<double>(q)) / static_cast<double>(g);
                const std::size_t m = p * static_cast<std::size_t>(g) + static_cast<std::size_t>(q);
                y1c[m] = std::cos(t1 + dtheta1) - std::cos(t1);
                y1s[m] = std::sin(t1 + dtheta1) - std::sin(t1);
                y2c[m] = std::cos(t2 + dtheta2) - std::cos(t2);
                y2s[m] = std::sin(t2 + dtheta2) - std::sin(t2);
                for (std::size_t s = 0; s < s_cnt; ++s) {
                    const double dotp = 0.5 * (std::cos(t1 - alpha_s[s]) + std::cos(t2 - beta_s[s]));
                    const double u = (dotp - tau) * inv_denom;
                    h_tilde(m, s) = (silu(beta_eff * u) * u) / beta_eff;
                }
            }
        }
    };

    if (m_samples * s_cnt >= 4096) {
        parallel_for_range(static_cast<std::size_t>(g), 4, fill_samples);
    } else {
        fill_samples(0, static_cast<std::size_t>(g));
    }

    Matrix gram(s_cnt, s_cnt, 0.0);
    Vector rhs1c(s_cnt, 0.0);
    Vector rhs1s(s_cnt, 0.0);
    Vector rhs2c(s_cnt, 0.0);
    Vector rhs2s(s_cnt, 0.0);

    auto build_gram_rows = [&](std::size_t r_begin, std::size_t r_end) {
        for (std::size_t r = r_begin; r < r_end; ++r) {
            long double a1c = 0.0, a1s = 0.0, a2c = 0.0, a2s = 0.0;
            for (std::size_t m = 0; m < m_samples; ++m) {
                const double hr = h_tilde(m, r);
                a1c += static_cast<long double>(hr) * static_cast<long double>(y1c[m]);
                a1s += static_cast<long double>(hr) * static_cast<long double>(y1s[m]);
                a2c += static_cast<long double>(hr) * static_cast<long double>(y2c[m]);
                a2s += static_cast<long double>(hr) * static_cast<long double>(y2s[m]);
            }
            rhs1c[r] = static_cast<double>(a1c);
            rhs1s[r] = static_cast<double>(a1s);
            rhs2c[r] = static_cast<double>(a2c);
            rhs2s[r] = static_cast<double>(a2s);
            for (std::size_t c = 0; c < s_cnt; ++c) {
                long double ag = 0.0;
                for (std::size_t m = 0; m < m_samples; ++m) {
                    ag += static_cast<long double>(h_tilde(m, r)) * static_cast<long double>(h_tilde(m, c));
                }
                gram(r, c) = static_cast<double>(ag);
            }
            gram(r, r) += 1e-11;
        }
    };

    if (s_cnt >= 25 && m_samples * s_cnt >= 4096) {
        parallel_for_range(s_cnt, 4, build_gram_rows);
    } else {
        build_gram_rows(0, s_cnt);
    }

    const Vector w1c = solve_linear(gram, std::move(rhs1c));
    const Vector w1s = solve_linear(gram, std::move(rhs1s));
    const Vector w2c = solve_linear(gram, std::move(rhs2c));
    const Vector w2s = solve_linear(gram, std::move(rhs2s));

    const std::size_t d = e0.size();
    std::vector<TessellatedSlot> slots;
    slots.reserve(s_cnt);
    for (std::size_t s = 0; s < s_cnt; ++s) {
        const double ca = 0.5 * std::cos(alpha_s[s]);
        const double sa = 0.5 * std::sin(alpha_s[s]);
        const double cb = 0.5 * std::cos(beta_s[s]);
        const double sb_val = 0.5 * std::sin(beta_s[s]);
        Vector z_s(d, 0.0);
        Vector w_down(d, 0.0);
        for (std::size_t i = 0; i < d; ++i) {
            const double p_i = ca * c1[i] + sa * s1[i] + cb * c2[i] + sb_val * s2[i];
            z_s[i] = (p_i - tau * e0[i]) * inv_denom;
            w_down[i] = (w1c[s] * c1[i] + w1s[s] * s1[i] + w2c[s] * c2[i] + w2s[s] * s2[i]) / beta_eff;
        }
        TessellatedSlot slot;
        slot.chart_index = static_cast<int>(s);
        slot.w_gate = vec_scale(z_s, beta_eff);
        slot.w_up = std::move(z_s);
        slot.w_down = std::move(w_down);
        slots.push_back(std::move(slot));
    }

    const double max_err = eval_t2_spline_error(slots, e0, c1, s1, c2, s2, dtheta1, dtheta2, 24);
    return {std::move(slots), max_err};
}

double eval_t2_spline_error(
    const std::vector<TessellatedSlot>& slots,
    const Vector& e0,
    const Vector& c1,
    const Vector& s1,
    const Vector& c2,
    const Vector& s2,
    double dtheta1,
    double dtheta2,
    std::size_t grid_per_axis) {
    const std::size_t g = std::max<std::size_t>(4, grid_per_axis);
    const std::size_t d = e0.size();
    Vector row_max(g, 0.0);

    auto eval_rows = [&](std::size_t p_begin, std::size_t p_end) {
        for (std::size_t p = p_begin; p < p_end; ++p) {
            const double t1 = (2.0 * kPi * (static_cast<double>(p) + 0.5)) / static_cast<double>(g);
            const double ct1 = std::cos(t1);
            const double st1 = std::sin(t1);
            const double dc1 = std::cos(t1 + dtheta1) - ct1;
            const double ds1 = std::sin(t1 + dtheta1) - st1;
            double local_max = 0.0;
            for (std::size_t q = 0; q < g; ++q) {
                const double t2 = (2.0 * kPi * (static_cast<double>(q) + 0.5)) / static_cast<double>(g);
                const double ct2 = std::cos(t2);
                const double st2 = std::sin(t2);
                const double dc2 = std::cos(t2 + dtheta2) - ct2;
                const double ds2 = std::sin(t2 + dtheta2) - st2;

                Vector x(d, 0.0);
                Vector target(d, 0.0);
                for (std::size_t i = 0; i < d; ++i) {
                    x[i] = e0[i] + ct1 * c1[i] + st1 * s1[i] + ct2 * c2[i] + st2 * s2[i];
                    target[i] = dc1 * c1[i] + ds1 * s1[i] + dc2 * c2[i] + ds2 * s2[i];
                }
                Vector pred(d, 0.0);
                for (const auto& s : slots) {
                    const double a = silu(dot(s.w_gate, x)) * dot(s.w_up, x);
                    for (std::size_t i = 0; i < d; ++i) {
                        pred[i] += a * s.w_down[i];
                    }
                }
                local_max = std::max(local_max, norm2(vec_sub(target, pred)));
            }
            row_max[p] = local_max;
        }
    };

    if (g * g * slots.size() >= 4096) {
        parallel_for_range(g, 2, eval_rows);
    } else {
        eval_rows(0, g);
    }

    double max_err = 0.0;
    for (double err : row_max) {
        max_err = std::max(max_err, err);
    }
    return max_err;
}

double fisher_inner(
    const std::vector<SVDMode>& a,
    const std::vector<SVDMode>& b,
    const Vector& g_out,
    const Vector& g_in) {
    if (a.empty() || b.empty()) {
        return 0.0;
    }
    const std::size_t d = a[0].u.size();
    Vector row_totals(a.size(), 0.0);
    auto eval_rows = [&](std::size_t ia_begin, std::size_t ia_end) {
        for (std::size_t ia = ia_begin; ia < ia_end; ++ia) {
            const auto& ma = a[ia];
            if (ma.u.size() != d || ma.v.size() != d) {
                continue;
            }
            double row_acc = 0.0;
            for (const auto& mb : b) {
                if (mb.u.size() != d || mb.v.size() != d) {
                    continue;
                }
                double u_ip = 0.0;
                double v_ip = 0.0;
                for (std::size_t i = 0; i < d; ++i) {
                    const double go = (i < g_out.size()) ? g_out[i] : 1.0;
                    const double gi = (i < g_in.size()) ? g_in[i] : 1.0;
                    u_ip += go * ma.u[i] * mb.u[i];
                    v_ip += gi * ma.v[i] * mb.v[i];
                }
                row_acc += ma.sigma * mb.sigma * u_ip * v_ip;
            }
            row_totals[ia] = row_acc;
        }
    };
    if (a.size() >= 8 && b.size() >= 4 && d >= 32) {
        parallel_for_range(a.size(), 2, eval_rows);
    } else {
        eval_rows(0, a.size());
    }
    double total = 0.0;
    for (double rt : row_totals) {
        total += rt;
    }
    return total;
}

double fisher_cosine(
    const std::vector<SVDMode>& a,
    const std::vector<SVDMode>& b,
    const Vector& g_out,
    const Vector& g_in) {
    const double na = fisher_inner(a, a, g_out, g_in);
    const double nb = fisher_inner(b, b, g_out, g_in);
    if (na <= 1e-24 || nb <= 1e-24) {
        return 0.0;
    }
    return std::abs(fisher_inner(a, b, g_out, g_in)) / std::sqrt(na * nb);
}

std::vector<SVDMode> fisher_decouple(
    const std::vector<SVDMode>& f1,
    const std::vector<SVDMode>& f2,
    const Vector& g_out,
    const Vector& g_in,
    double reg) {
    (void)reg;
    if (f1.empty()) {
        return {};
    }
    const std::size_t d = f1[0].u.size();
    if (d == 0) {
        return {};
    }

    Vector sqrt_go(d, 1.0);
    Vector sqrt_gi(d, 1.0);
    for (std::size_t i = 0; i < d; ++i) {
        const double go = (i < g_out.size() && g_out[i] > 1e-12) ? g_out[i] : 1.0;
        const double gi = (i < g_in.size() && g_in[i] > 1e-12) ? g_in[i] : 1.0;
        sqrt_go[i] = std::sqrt(go);
        sqrt_gi[i] = std::sqrt(gi);
    }

    const Matrix m1 = modes_to_matrix(f1, d);
    Vector w1_flat(d * d, 0.0);
    auto fill_w1 = [&](std::size_t r_begin, std::size_t r_end) {
        for (std::size_t r = r_begin; r < r_end; ++r) {
            for (std::size_t c = 0; c < d; ++c) {
                w1_flat[r * d + c] = sqrt_go[r] * m1(r, c) * sqrt_gi[c];
            }
        }
    };
    if (d >= 32) {
        parallel_for_range(d, 8, fill_w1);
    } else {
        fill_w1(0, d);
    }

    std::vector<Vector> raw_basis_slots(f2.size());
    auto fill_f2 = [&](std::size_t j_begin, std::size_t j_end) {
        for (std::size_t j = j_begin; j < j_end; ++j) {
            const auto& m2 = f2[j];
            if (m2.sigma <= 1e-12 || m2.u.size() != d || m2.v.size() != d) {
                continue;
            }
            Vector flat(d * d, 0.0);
            for (std::size_t r = 0; r < d; ++r) {
                const double sr = sqrt_go[r] * m2.u[r];
                for (std::size_t c = 0; c < d; ++c) {
                    flat[r * d + c] = sr * sqrt_gi[c] * m2.v[c];
                }
            }
            if (norm2(flat) > 1e-12) {
                raw_basis_slots[j] = std::move(flat);
            }
        }
    };
    if (d >= 32 && f2.size() >= 2) {
        parallel_for_range(f2.size(), 1, fill_f2);
    } else {
        fill_f2(0, f2.size());
    }

    std::vector<Vector> raw_basis;
    raw_basis.reserve(f2.size());
    for (auto& slot : raw_basis_slots) {
        if (!slot.empty()) {
            raw_basis.push_back(std::move(slot));
        }
    }

    const std::vector<Vector> orth_basis = gram_schmidt(raw_basis, 1e-12);
    Vector pure_flat = w1_flat;
    for (int pass = 0; pass < 2; ++pass) {
        for (const Vector& ej : orth_basis) {
            const double coeff = dot(pure_flat, ej);
            if (coeff != 0.0) {
                auto sub_range = [&](std::size_t i_begin, std::size_t i_end) {
                    for (std::size_t idx = i_begin; idx < i_end; ++idx) {
                        pure_flat[idx] -= coeff * ej[idx];
                    }
                };
                if (pure_flat.size() >= 4096) {
                    parallel_for_range(pure_flat.size(), 1024, sub_range);
                } else {
                    sub_range(0, pure_flat.size());
                }
            }
        }
    }

    Matrix m_pure(d, d, 0.0);
    auto fill_mpure = [&](std::size_t r_begin, std::size_t r_end) {
        for (std::size_t r = r_begin; r < r_end; ++r) {
            for (std::size_t c = 0; c < d; ++c) {
                m_pure(r, c) = pure_flat[r * d + c] / (sqrt_go[r] * sqrt_gi[c]);
            }
        }
    };
    if (d >= 32) {
        parallel_for_range(d, 8, fill_mpure);
    } else {
        fill_mpure(0, d);
    }

    std::vector<SVDMode> raw_modes = svd_top_k(m_pure, d, 50, 0);
    std::vector<SVDMode> out;
    for (auto& m : raw_modes) {
        if (m.sigma > 1e-12) {
            out.push_back(std::move(m));
        }
    }
    return out;
}

std::vector<SVDMode> temper_field_modes(
    const std::vector<SVDMode>& modes,
    const std::string& band,
    double gain) {
    std::vector<SVDMode> out = modes;
    const std::size_t n = std::max<std::size_t>(1, out.size());
    const bool is_low = (band.find("low") != std::string::npos);
    const bool is_high = (band.find("high") != std::string::npos);
    for (std::size_t k = 0; k < out.size(); ++k) {
        double weight = gain;
        if (is_low) {
            weight *= std::exp(-0.15 * static_cast<double>(k));
        } else if (is_high) {
            weight *= 0.5 + 0.5 * (static_cast<double>(k) / static_cast<double>(n));
        }
        out[k].sigma *= weight;
    }
    return out;
}

std::vector<SVDMode> blend_field_modes(
    const std::vector<SVDMode>& fa,
    const std::vector<SVDMode>& fb,
    double alpha) {
    std::size_t d = 0;
    if (!fa.empty()) {
        d = fa[0].u.size();
    } else if (!fb.empty()) {
        d = fb[0].u.size();
    }
    if (d == 0) {
        return {};
    }
    const Matrix ma = modes_to_matrix(fa, d);
    const Matrix mb = modes_to_matrix(fb, d);
    const Matrix blended = mat_add(mat_scale(ma, 1.0 - alpha), mat_scale(mb, alpha));
    std::vector<SVDMode> raw = svd_top_k(blended, d, 50, 0);
    std::vector<SVDMode> out;
    for (auto& m : raw) {
        if (m.sigma > 1e-12) {
            out.push_back(std::move(m));
        }
    }
    return out;
}

CIRModule lower_mir_to_cir(
    const MIRModule& mir,
    const ModelConfig& cfg,
    const NeuralAtlas& atlas,
    const std::string& target_mode) {
    CIRModule cir;
    cir.name = mir.name;
    cir.arch_name = cfg.arch_name;
    cir.hidden_size = cfg.hidden_size;
    cir.num_layers = cfg.num_layers;
    cir.target_mode = target_mode.empty() ? "in-place" : target_mode;

    const std::size_t d = static_cast<std::size_t>(std::max(0, cfg.hidden_size));
    int elastic_layer_cursor = cfg.num_layers;

    for (const auto& flow : mir.flows) {
        if (flow.is_helper) {
            continue;
        }
        const std::string mode = target_mode.empty()
            ? (flow.target_mode_hint.empty() ? std::string("in-place") : flow.target_mode_hint)
            : target_mode;

        CIRFlow cf;
        cf.name = flow.name;
        cf.layer_range = cfg.stage_to_layers(flow.stage.start, flow.stage.end, flow.stage.inclusive);
        cf.target_mode_hint = flow.target_mode_hint;
        cf.contract = flow.contract;

        const int l0 = cf.layer_range.first;
        const int l1 = cf.layer_range.second;
        const int avail_layers = std::max(0, l1 - l0);

        std::map<std::string, std::vector<SVDMode>> field_env;
        std::map<std::string, std::set<std::string>> field_blend_sources;
        std::map<std::string, std::set<std::string>> field_decoupled_from;
        std::map<std::string, bool> field_metric_env;
        std::map<std::string, std::size_t> spline_op_index;

        int cur_step = 0;
        bool step_has_mlp = false;
        int max_steps_needed = 0;
        bool has_micro_or_rodata = false;

        for (const auto& op : flow.ops) {
            if (op.op == "field_decouple" || op.op == "field_temper" || op.op == "field_blend") {
                max_steps_needed = std::max(max_steps_needed, 1);
            } else if ((op.op == "excite" || op.op == "annihilate") &&
                       is_field_name(op.target, mir, atlas, field_env)) {
                max_steps_needed = std::max(max_steps_needed, 1);
            } else if (op.op == "attend") {
                has_micro_or_rodata = true;
                if (step_has_mlp) {
                    ++cur_step;
                    step_has_mlp = false;
                }
                max_steps_needed = std::max(max_steps_needed, cur_step + 1);
            } else if (op.op == "rodata_insert" || op.op == "rodata_prune" || op.op == "rodata_rewrite") {
                has_micro_or_rodata = true;
                max_steps_needed = std::max(max_steps_needed, cur_step + 1);
            } else if (op.op == "frame_transport") {
                has_micro_or_rodata = true;
                max_steps_needed = std::max(max_steps_needed, cur_step + 1);
                step_has_mlp = true;
            } else if (op.op == "when") {
                has_micro_or_rodata = true;
                if (step_has_mlp) {
                    ++cur_step;
                    step_has_mlp = false;
                }
                step_has_mlp = true;
                max_steps_needed = std::max(max_steps_needed, cur_step + 1);
            } else {
                has_micro_or_rodata = true;
                max_steps_needed = std::max(max_steps_needed, cur_step + 1);
            }
        }
        (void)has_micro_or_rodata;

        cf.sequential_depth = max_steps_needed;
        int flow_elastic_base = elastic_layer_cursor;
        if (max_steps_needed > avail_layers) {
            if (mode == "in-place") {
                throw OutOfDepthError(
                    "Sequential depth " + std::to_string(max_steps_needed) +
                    " exceeds available layers (" + std::to_string(avail_layers) +
                    ") in stage for flow '" + flow.name + "'");
            }
            if (mode == "lora") {
                if (avail_layers <= 0) {
                    throw OutOfDepthError(
                        "No available layers in stage for LoRA flow '" + flow.name + "'");
                }
                cf.added_layers_needed = 0;
            } else {
                cf.added_layers_needed = max_steps_needed - avail_layers;
                elastic_layer_cursor += cf.added_layers_needed;
            }
        }

        auto step_to_layer = [&](int step) -> int {
            if (step < avail_layers) {
                return l0 + step;
            }
            if (mode == "lora" && avail_layers > 0) {
                return l0 + (step % avail_layers);
            }
            return flow_elastic_base + (step - avail_layers);
        };

        const NullspaceOperators ns = build_nullspace_ops(flow.contract.preserve, mir, atlas, d);

        struct OpProjMeta {
            std::string topo = "Euclidean";
            double curvature = 0.0;
            bool has_sb = false;
            SymbolBinding sb;
            double delta1 = 2.0 * kPi / 7.0;
            double delta2 = 2.0 * kPi / 5.0;
        };
        std::vector<OpProjMeta> proj_metas;

        auto lower_single_op = [&](auto& self, const MIROp& op, int step, const MIROp* active_gate) -> void {
            const int phys_layer = step_to_layer(step);

            if (op.op == "field_decouple") {
                const auto f1 = get_symbol_field_modes(op.target, atlas, d, field_env);
                const auto f2 = get_symbol_field_modes(op.rhs_target, atlas, d, field_env);
                const bool use_fisher = resolve_field_use_fisher(op.target, op.args, mir, field_metric_env);
                const auto [g_out_flow, g_in_flow] =
                    resolve_flow_fisher_metric(l0, l1, d, atlas, use_fisher);
                auto pure = fisher_decouple(f1, f2, g_out_flow, g_in_flow, op.scalar_arg);
                const double xt = fisher_cosine(pure, f2, g_out_flow, g_in_flow);
                if (xt > flow.contract.crosstalk_bound + 1e-12) {
                    throw CrosstalkExceededError(
                        "Fisher decouple residual crosstalk " + std::to_string(xt) +
                        " exceeds bound " + std::to_string(flow.contract.crosstalk_bound));
                }
                if (!op.dst.empty()) {
                    field_env[op.dst] = pure;
                    field_metric_env[op.dst] = use_fisher;
                    field_decoupled_from[op.dst].insert(op.rhs_target);
                    if (field_decoupled_from.contains(op.target)) {
                        field_decoupled_from[op.dst].insert(
                            field_decoupled_from[op.target].begin(),
                            field_decoupled_from[op.target].end());
                    }
                }
                CIROp cop;
                cop.kind = "field_decouple";
                cop.source_op = op.op;
                cop.symbol = op.target;
                cop.pole = op.rhs_target;
                cop.layer = (avail_layers > 0) ? l0 : flow_elastic_base;
                cop.layer_end = (avail_layers > 0) ? l1 : (flow_elastic_base + cf.added_layers_needed);
                cop.crosstalk = xt;
                cop.field_modes = std::move(pure);
                cf.ops.push_back(std::move(cop));
                proj_metas.push_back(OpProjMeta{});
                return;
            }

            if (op.op == "field_temper") {
                const auto f_in = get_symbol_field_modes(op.target, atlas, d, field_env);
                const std::string band = arg_as_string(op.args, "band", "low_curvature");
                auto tempered = temper_field_modes(f_in, band, op.scalar_arg);
                if (!op.dst.empty()) {
                    field_env[op.dst] = tempered;
                    field_metric_env[op.dst] = resolve_field_use_fisher(op.target, op.args, mir, field_metric_env);
                    if (field_decoupled_from.contains(op.target)) {
                        field_decoupled_from[op.dst] = field_decoupled_from[op.target];
                    }
                    if (field_blend_sources.contains(op.target)) {
                        field_blend_sources[op.dst] = field_blend_sources[op.target];
                    }
                    field_blend_sources[op.dst].insert(op.target);
                }
                CIROp cop;
                cop.kind = "field_temper";
                cop.source_op = op.op;
                cop.symbol = op.target;
                cop.layer = (avail_layers > 0) ? l0 : flow_elastic_base;
                cop.layer_end = (avail_layers > 0) ? l1 : (flow_elastic_base + cf.added_layers_needed);
                cop.scalar = op.scalar_arg;
                cop.field_modes = std::move(tempered);
                cf.ops.push_back(std::move(cop));
                proj_metas.push_back(OpProjMeta{});
                return;
            }

            if (op.op == "field_blend") {
                const auto fa = get_symbol_field_modes(op.target, atlas, d, field_env);
                const auto fb = get_symbol_field_modes(op.rhs_target, atlas, d, field_env);
                auto blended = blend_field_modes(fa, fb, op.scalar_arg);
                if (!op.dst.empty()) {
                    field_env[op.dst] = blended;
                    field_metric_env[op.dst] = resolve_field_use_fisher(op.target, op.args, mir, field_metric_env);
                    field_blend_sources[op.dst].insert(op.target);
                    field_blend_sources[op.dst].insert(op.rhs_target);
                    if (field_blend_sources.contains(op.target)) {
                        field_blend_sources[op.dst].insert(
                            field_blend_sources[op.target].begin(),
                            field_blend_sources[op.target].end());
                    }
                    if (field_blend_sources.contains(op.rhs_target)) {
                        field_blend_sources[op.dst].insert(
                            field_blend_sources[op.rhs_target].begin(),
                            field_blend_sources[op.rhs_target].end());
                    }
                }
                CIROp cop;
                cop.kind = "field_blend";
                cop.source_op = op.op;
                cop.symbol = op.target;
                cop.pole = op.rhs_target;
                cop.layer = (avail_layers > 0) ? l0 : flow_elastic_base;
                cop.layer_end = (avail_layers > 0) ? l1 : (flow_elastic_base + cf.added_layers_needed);
                cop.scalar = op.scalar_arg;
                cop.field_modes = std::move(blended);
                cf.ops.push_back(std::move(cop));
                proj_metas.push_back(OpProjMeta{});
                return;
            }

            if ((op.op == "excite" || op.op == "annihilate") &&
                is_field_name(op.target, mir, atlas, field_env)) {
                if (!op.scalar_gate.empty()) {
                    throw EnergyUnboundedError("Dynamic coefficients are not supported for Field edits");
                }
                std::vector<SVDMode> modes = get_symbol_field_modes(op.target, atlas, d, field_env);
                const double sign = (op.op == "annihilate") ? -op.scalar_arg : op.scalar_arg;
                const bool use_fisher = resolve_field_use_fisher(op.target, op.args, mir, field_metric_env);
                const auto [g_out_flow, g_in_flow] =
                    resolve_flow_fisher_metric(l0, l1, d, atlas, use_fisher);

                double max_xt = 0.0;
                for (const std::string& p : flow.contract.preserve) {
                    const bool is_decl_field = mir.declarations.contains(p) &&
                                               mir.declarations.at(p).kind == "field";
                    const bool is_atlas_field = atlas.symbols.contains(p) &&
                                                atlas.symbols.at(p).kind == "field";
                    if (!is_decl_field && !is_atlas_field) {
                        continue;
                    }
                    const bool allowed_blend = (op.op == "excite") &&
                        (op.target == p ||
                         (field_blend_sources.contains(op.target) &&
                          field_blend_sources[op.target].contains(p)));
                    if (allowed_blend) {
                        continue;
                    }
                    const auto f_pres = get_symbol_field_modes(p, atlas, d, field_env);
                    const bool is_decoupled = field_decoupled_from.contains(op.target) &&
                        field_decoupled_from[op.target].contains(p);
                    if (!is_decoupled) {
                        const double xt = fisher_cosine(modes, f_pres, g_out_flow, g_in_flow);
                        if (xt > flow.contract.crosstalk_bound + 1e-12) {
                            modes = fisher_decouple(modes, f_pres, g_out_flow, g_in_flow, 1e-6);
                        }
                    }
                    max_xt = std::max(max_xt, fisher_cosine(modes, f_pres, g_out_flow, g_in_flow));
                }

                if (max_xt > flow.contract.crosstalk_bound + 1e-12) {
                    throw CrosstalkExceededError(
                        "Field crosstalk " + std::to_string(max_xt) +
                        " exceeds bound " + std::to_string(flow.contract.crosstalk_bound));
                }

                for (auto& m : modes) {
                    m.sigma *= sign;
                }

                CIROp cop;
                cop.kind = "field_spectral";
                cop.source_op = op.op;
                cop.symbol = op.target;
                cop.layer = (avail_layers > 0) ? l0 : flow_elastic_base;
                cop.layer_end = (avail_layers > 0) ? l1 : (flow_elastic_base + cf.added_layers_needed);
                cop.scalar = sign;
                cop.crosstalk = max_xt;
                cop.field_modes = std::move(modes);
                cf.ops.push_back(std::move(cop));
                proj_metas.push_back(OpProjMeta{});
                return;
            }

            if (op.op == "excite" && spline_op_index.contains(op.target)) {
                const std::size_t idx = spline_op_index[op.target];
                for (auto& slot : cf.ops[idx].spline_slots) {
                    slot.w_down = vec_scale(slot.w_down, op.scalar_arg);
                }
                cf.ops[idx].scalar *= op.scalar_arg;
                return;
            }

            if (op.op == "transport_map") {
                const SymbolBinding sb = resolve_symbol_binding(op.target, mir, atlas, cfg);
                double curvature = 0.0;
                std::string topo = "Euclidean";
                auto mit = mir.declarations.find(op.target);
                if (mit != mir.declarations.end()) {
                    curvature = mit->second.curvature;
                    topo = mit->second.topology;
                }
                const bool t2_mode = is_t2_topology(topo);
                if (curvature <= 0.0 && topo != "S1" && !t2_mode) {
                    CIROp cop;
                    cop.kind = "vpd_rank1";
                    cop.source_op = op.op;
                    cop.symbol = op.target;
                    cop.matrix_id = "self_attn.o_proj";
                    cop.layer = phys_layer;
                    cop.scalar = op.scalar_arg;
                    cop.approx_error = 0.0;
                    cop.input_dir = sb.direction;
                    cop.output_dir = sb.direction;
                    cf.ops.push_back(std::move(cop));
                    proj_metas.push_back(OpProjMeta{topo, curvature, true, sb, 0.0, 0.0});
                    return;
                }

                if (flow.contract.approx_bound <= 0.0) {
                    throw ApproximationUnmetError("Curved manifold transport requires positive approx bound");
                }

                std::vector<TessellatedSlot> best_slots;
                double best_err = 1e9;
                bool met = false;
                double shift1 = 2.0 * kPi / 7.0;
                double shift2 = 2.0 * kPi / 5.0;

                if (t2_mode) {
                    if (sb.manifold_plane.size() < 5) {
                        throw ApproximationUnmetError("T2 torus manifold plane requires 5 orthonormal basis vectors");
                    }
                    const auto arg_dt1 = extract_numeric_arg(op.args, "dtheta1");
                    const auto arg_d1 = extract_numeric_arg(op.args, "delta1");
                    const auto arg_delta = extract_numeric_arg(op.args, "delta");
                    const auto arg_angle = extract_numeric_arg(op.args, "angle");
                    if (arg_dt1) {
                        shift1 = *arg_dt1;
                    } else if (arg_d1) {
                        shift1 = *arg_d1;
                    } else if (arg_delta) {
                        shift1 = *arg_delta;
                    } else if (arg_angle) {
                        shift1 = *arg_angle;
                    }

                    const auto arg_dt2 = extract_numeric_arg(op.args, "dtheta2");
                    const auto arg_d2 = extract_numeric_arg(op.args, "delta2");
                    if (arg_dt2) {
                        shift2 = *arg_dt2;
                    } else if (arg_d2) {
                        shift2 = *arg_d2;
                    } else if (arg_delta) {
                        shift2 = *arg_delta;
                    }

                    for (int k_axis : {3, 5, 7, 9}) {
                        auto [slots, err] = build_t2_swiglu_spline(
                            sb.manifold_plane[0],
                            sb.manifold_plane[1],
                            sb.manifold_plane[2],
                            sb.manifold_plane[3],
                            sb.manifold_plane[4],
                            k_axis,
                            shift1,
                            shift2);
                        best_slots = std::move(slots);
                        best_err = err;
                        if (err <= flow.contract.approx_bound) {
                            met = true;
                            break;
                        }
                    }
                    if (!met) {
                        throw ApproximationUnmetError(
                            "T2 tessellation error " + std::to_string(best_err) +
                            " exceeds contract approx <= " + std::to_string(flow.contract.approx_bound));
                    }
                } else {
                    if (sb.manifold_plane.size() < 3) {
                        throw ApproximationUnmetError("S1 manifold plane requires 3 orthonormal basis vectors");
                    }
                    if (const auto arg_delta = extract_numeric_arg(op.args, "delta")) {
                        shift1 = *arg_delta;
                    } else if (const auto arg_angle = extract_numeric_arg(op.args, "angle")) {
                        shift1 = *arg_angle;
                    }
                    for (int k_charts : {4, 8, 16, 32}) {
                        auto [slots, err] = build_s1_swiglu_spline(
                            sb.manifold_plane[0],
                            sb.manifold_plane[1],
                            sb.manifold_plane[2],
                            k_charts,
                            shift1);
                        best_slots = std::move(slots);
                        best_err = err;
                        if (err <= flow.contract.approx_bound) {
                            met = true;
                            break;
                        }
                    }
                    if (!met) {
                        throw ApproximationUnmetError(
                            "S1 tessellation error " + std::to_string(best_err) +
                            " exceeds contract approx <= " + std::to_string(flow.contract.approx_bound));
                    }
                }

                CIROp cop;
                cop.kind = "swiglu_spline";
                cop.source_op = op.op;
                cop.symbol = op.target;
                cop.layer = phys_layer;
                cop.layer_end = (avail_layers > 0) ? l1 : (flow_elastic_base + cf.added_layers_needed);
                cop.scalar = 1.0;
                cop.approx_error = best_err;
                cop.spline_slots = std::move(best_slots);
                if (!op.dst.empty()) {
                    spline_op_index[op.dst] = cf.ops.size();
                }
                cf.ops.push_back(std::move(cop));
                proj_metas.push_back(OpProjMeta{topo, curvature, true, sb, shift1, shift2});
                return;
            }

            if (op.op == "frame_transport") {
                const std::string src_lbl = stage_to_label(op.src_frame);
                const std::string dst_lbl = stage_to_label(op.dst_frame);
                const std::string k1 = src_lbl + "->" + dst_lbl;
                const std::string k2 = dst_lbl + "->" + src_lbl;
                Matrix bridge = eye(d);
                if (atlas.frame_bridges.contains(k1)) {
                    bridge = atlas.frame_bridges.at(k1);
                } else if (atlas.frame_bridges.contains(k2)) {
                    bridge = transpose(atlas.frame_bridges.at(k2));
                } else if (!atlas.frame_bridges.empty()) {
                    for (const auto& [bname, bmat] : atlas.frame_bridges) {
                        if (bname.find("->" + dst_lbl) != std::string::npos) {
                            bridge = bmat;
                            break;
                        }
                    }
                }
                CIROp cop;
                cop.kind = "frame_bridge";
                cop.source_op = op.op;
                cop.symbol = k1;
                cop.matrix_id = "self_attn.o_proj";
                cop.layer = phys_layer;
                cop.bridge_matrix = std::move(bridge);
                cf.ops.push_back(std::move(cop));
                proj_metas.push_back(OpProjMeta{});
                return;
            }

            if (op.op == "attend") {
                const SymbolBinding k_sb = resolve_symbol_binding(op.target, mir, atlas, cfg);
                const SymbolBinding v_sb = resolve_symbol_binding(op.rhs_target, mir, atlas, cfg);
                const Vector q_dir = k_sb.direction.empty() ? seeded_unit_vec("q:" + op.target, d) : k_sb.direction;
                const Vector k_dir = k_sb.direction.empty() ? seeded_unit_vec("k:" + op.target, d) : k_sb.direction;
                const Vector v_dir = v_sb.direction.empty() ? seeded_unit_vec("v:" + op.rhs_target, d) : v_sb.direction;
                const Vector o_dir = v_sb.direction.empty() ? seeded_unit_vec("o:" + op.rhs_target, d) : v_sb.direction;

                CIROp cop;
                cop.kind = "attn_bilinear";
                cop.source_op = op.op;
                cop.symbol = op.target;
                cop.pole = op.rhs_target;
                cop.layer = phys_layer;
                cop.layer_end = (avail_layers > 0) ? l1 : (flow_elastic_base + cf.added_layers_needed);
                cop.w_qk = outer(q_dir, k_dir);
                cop.w_ov = outer(o_dir, v_dir);
                cop.rope_mode = arg_as_string(op.args, "mask", "causal");
                cf.ops.push_back(std::move(cop));
                proj_metas.push_back(OpProjMeta{});
                return;
            }

            if (op.op == "rodata_insert" || op.op == "rodata_prune" || op.op == "rodata_rewrite") {
                const SymbolBinding sb = resolve_symbol_binding(op.target, mir, atlas, cfg);
                const std::string key_str = arg_as_string(op.args, "key", op.target);
                const std::string val_str = (op.op == "rodata_rewrite")
                    ? arg_as_string(op.args, "new", "rewritten")
                    : arg_as_string(op.args, "value", "inserted");

                Vector k_vec;
                const std::string layer_k = "@L" + std::to_string(phys_layer) + ":" + key_str;
                if (sb.rodata_key_vectors.contains(layer_k)) {
                    k_vec = sb.rodata_key_vectors.at(layer_k);
                } else if (sb.rodata_key_vectors.contains(key_str)) {
                    k_vec = sb.rodata_key_vectors.at(key_str);
                } else if (!sb.preserve_keys.empty()) {
                    k_vec = seeded_unit_vec("rodata_key:" + op.target + ":" + key_str, d);
                    const std::size_t idx = hash_str(key_str) % sb.preserve_keys.size();
                    if (sb.preserve_keys[idx].size() == d) {
                        k_vec = sb.preserve_keys[idx];
                    }
                } else {
                    k_vec = seeded_unit_vec("rodata_key:" + op.target + ":" + key_str, d);
                }

                Vector v_vec;
                double t_norm = (op.op == "rodata_prune") ? 0.0 : 1.0;
                if (op.op == "rodata_prune") {
                    v_vec = vec_zeros(d);
                } else if (sb.rodata_val_vectors.contains(val_str)) {
                    v_vec = sb.rodata_val_vectors.at(val_str);
                    t_norm = norm2(v_vec);
                } else {
                    v_vec = seeded_unit_vec("rodata_val:" + op.target + ":" + val_str, d);
                }

                Matrix cov_mat;
                if (atlas.mlp_cov_inv.contains(phys_layer) &&
                    (k_vec.empty() || atlas.mlp_cov_inv.at(phys_layer).rows == k_vec.size())) {
                    cov_mat = atlas.mlp_cov_inv.at(phys_layer);
                } else if (!k_vec.empty()) {
                    cov_mat = eye(k_vec.size());
                } else {
                    cov_mat = eye(d);
                }

                CIROp cop;
                cop.kind = "assoc_mem_row";
                cop.source_op = op.op;
                cop.symbol = op.target;
                cop.layer = phys_layer;
                cop.key_vec = std::move(k_vec);
                cop.val_vec = std::move(v_vec);
                cop.cov_inv = std::move(cov_mat);
                cop.target_norm = t_norm;
                cf.ops.push_back(std::move(cop));
                proj_metas.push_back(OpProjMeta{"Euclidean", 0.0, true, sb});
                return;
            }

            if (op.op == "when") {
                for (const auto& sub : op.body) {
                    self(self, sub, step, &op);
                }
                for (const auto& sub : op.else_body) {
                    self(self, sub, step, active_gate);
                }
                if (op.body.empty() && op.else_body.empty()) {
                    const SymbolBinding sb = resolve_symbol_binding(op.target, mir, atlas, cfg);
                    CIROp cop;
                    cop.kind = "swiglu_pole";
                    cop.source_op = op.op;
                    cop.symbol = op.target;
                    cop.pole = op.pole;
                    cop.layer = phys_layer;
                    cop.layer_end = (avail_layers > 0) ? l1 : (flow_elastic_base + cf.added_layers_needed);
                    cop.polarity = (op.pole == "-pole") ? -1 : 1;
                    cop.threshold = op.scalar_arg;
                    cop.scalar = 0.0;
                    cop.input_dir = sb.direction;
                    cop.output_dir = sb.direction;
                    cf.ops.push_back(std::move(cop));
                    proj_metas.push_back(OpProjMeta{"Euclidean", 0.0, true, sb});
                }
                return;
            }

            if (op.op == "excite" || op.op == "annihilate") {
                const SymbolBinding sb = resolve_symbol_binding(op.target, mir, atlas, cfg);
                const int pol = (op.pole == "-pole") ? -1 : 1;
                const Vector dir = sb.direction.empty()
                    ? seeded_unit_vec("dir:" + op.target, d)
                    : sb.direction;

                CIROp cop;
                cop.source_op = op.op;
                cop.symbol = op.target;
                cop.pole = op.pole;
                cop.layer = phys_layer;
                cop.layer_end = (avail_layers > 0) ? l1 : (flow_elastic_base + cf.added_layers_needed);
                cop.polarity = pol;
                cop.input_dir = dir;
                cop.output_dir = dir;

                if (!op.scalar_gate.empty()) {
                    if (active_gate == nullptr || active_gate->dst != op.scalar_gate) {
                        throw EnergyUnboundedError(
                            "Dynamic coefficient gate '" + op.scalar_gate + "' has no active when binding");
                    }
                    const SymbolBinding gate_sb = resolve_symbol_binding(
                        active_gate->target, mir, atlas, cfg);
                    cop.kind = "thresholded_route";
                    cop.symbol = active_gate->target;
                    cop.pole = active_gate->pole;
                    cop.input_dir = gate_sb.direction.empty()
                        ? seeded_unit_vec("gate:" + active_gate->target, d)
                        : gate_sb.direction;
                    cop.output_dir = dir;
                    cop.threshold = active_gate->scalar_arg;
                    cop.scalar = (op.op == "annihilate" ? -1.0 : 1.0) *
                        op.scalar_arg * static_cast<double>(pol);
                } else if (op.op == "annihilate") {
                    cop.kind = "vpd_rank1";
                    cop.matrix_id = "mlp.down_proj";
                    cop.scalar = -op.scalar_arg;
                } else {
                    cop.kind = "swiglu_pole";
                    cop.scalar = op.scalar_arg;
                    cop.threshold = 0.5;
                }

                cf.ops.push_back(std::move(cop));
                proj_metas.push_back(OpProjMeta{"Euclidean", 0.0, true, sb});
            }
        };

        cur_step = 0;
        step_has_mlp = false;
        for (const auto& op : flow.ops) {
            if (op.op == "field_decouple" || op.op == "field_temper" || op.op == "field_blend") {
                lower_single_op(lower_single_op, op, cur_step, nullptr);
            } else if ((op.op == "excite" || op.op == "annihilate") &&
                       is_field_name(op.target, mir, atlas, field_env)) {
                lower_single_op(lower_single_op, op, cur_step, nullptr);
            } else if (op.op == "attend") {
                if (step_has_mlp) {
                    ++cur_step;
                    step_has_mlp = false;
                }
                lower_single_op(lower_single_op, op, cur_step, nullptr);
            } else if (op.op == "rodata_insert" || op.op == "rodata_prune" || op.op == "rodata_rewrite") {
                lower_single_op(lower_single_op, op, cur_step, nullptr);
            } else if (op.op == "frame_transport") {
                lower_single_op(lower_single_op, op, cur_step, nullptr);
                step_has_mlp = true;
            } else if (op.op == "when") {
                if (step_has_mlp) {
                    ++cur_step;
                    step_has_mlp = false;
                }
                lower_single_op(lower_single_op, op, cur_step, nullptr);
                step_has_mlp = true;
            } else {
                lower_single_op(lower_single_op, op, cur_step, nullptr);
            }
        }

        if (ns.active && !cf.ops.empty()) {
            auto run_proj = [&](std::size_t idx_begin, std::size_t idx_end) {
                for (std::size_t i = idx_begin; i < idx_end; ++i) {
                    const auto& pm = proj_metas[i];
                    project_and_check_op(
                        cf.ops[i],
                        ns,
                        flow.contract.crosstalk_bound,
                        pm.topo,
                        pm.curvature,
                        flow.contract.approx_bound,
                        pm.delta1,
                        pm.delta2,
                        pm.has_sb ? &pm.sb : nullptr);
                }
            };
            if (cf.ops.size() >= 2 && d >= 16) {
                parallel_for_range(cf.ops.size(), 1, run_proj);
            } else {
                run_proj(0, cf.ops.size());
            }
        }

        for (const auto& cop : cf.ops) {
            cf.max_approx_error = std::max(cf.max_approx_error, cop.approx_error);
            cf.max_crosstalk = std::max(cf.max_crosstalk, cop.crosstalk);
        }

        cir.flows.push_back(std::move(cf));
    }

    return cir;
}

CIRModule optimize_cir(const CIRModule& cir, int opt_level) {
    if (opt_level <= 0) {
        return cir;
    }

    CIRModule out = cir;
    for (auto& flow : out.flows) {
        std::vector<CIROp> opt_ops;
        opt_ops.reserve(flow.ops.size());

        for (const auto& op : flow.ops) {
            if ((op.kind == "vpd_rank1" || op.kind == "swiglu_pole") && std::abs(op.scalar) < 1e-12) {
                continue;
            }
            if (op.kind == "field_spectral" || op.kind == "field_spectral_op") {
                bool any_nz = false;
                for (const auto& m : op.field_modes) {
                    if (std::abs(m.sigma) >= 1e-12) {
                        any_nz = true;
                        break;
                    }
                }
                if (!any_nz) {
                    continue;
                }
            }
            if (op.kind == "frame_bridge" && !opt_ops.empty() &&
                opt_ops.back().kind == "frame_bridge" &&
                opt_ops.back().layer == op.layer &&
                !opt_ops.back().bridge_matrix.data.empty() &&
                opt_ops.back().bridge_matrix.cols == op.bridge_matrix.rows) {
                opt_ops.back().bridge_matrix = mat_mul(op.bridge_matrix, opt_ops.back().bridge_matrix);
                opt_ops.back().crosstalk = std::max(opt_ops.back().crosstalk, op.crosstalk);
                opt_ops.back().symbol += "+" + op.symbol;
                continue;
            }
            opt_ops.push_back(op);
        }

        flow.ops = std::move(opt_ops);
        flow.max_approx_error = 0.0;
        flow.max_crosstalk = 0.0;
        for (const auto& cop : flow.ops) {
            flow.max_approx_error = std::max(flow.max_approx_error, cop.approx_error);
            flow.max_crosstalk = std::max(flow.max_crosstalk, cop.crosstalk);
        }
    }
    return out;
}

std::string format_cir(const CIRModule& cir) {
    std::ostringstream oss;
    oss << "cir.module @" << cir.name << " arch=\"" << cir.arch_name
        << "\" d=" << cir.hidden_size << " L=" << cir.num_layers
        << " target=\"" << cir.target_mode << "\" {\n";
    for (const auto& flow : cir.flows) {
        oss << "  cir.flow @" << flow.name << " layers=[" << flow.layer_range.first
            << ", " << flow.layer_range.second << ")"
            << " seq_depth=" << flow.sequential_depth
            << " added_layers=" << flow.added_layers_needed
            << " max_approx=" << flow.max_approx_error
            << " max_crosstalk=" << flow.max_crosstalk << " {\n";
        for (const auto& op : flow.ops) {
            oss << "    cir." << op.kind << " @" << op.symbol;
            if (!op.pole.empty()) {
                oss << "." << op.pole;
            }
            oss << " layer=" << op.layer;
            if (op.kind == "swiglu_spline") {
                oss << " slots=" << op.spline_slots.size()
                    << " approx=" << op.approx_error;
            } else if (op.kind == "field_spectral" || op.kind == "field_decouple" ||
                       op.kind == "field_temper" || op.kind == "field_blend") {
                oss << " layers=[" << op.layer << ", " << op.layer_end << ")"
                    << " modes=" << op.field_modes.size();
            } else {
                oss << " scalar=" << op.scalar;
            }
            oss << " crosstalk=" << op.crosstalk << "\n";
        }
        oss << "  }\n";
    }
    oss << "}\n";
    return oss.str();
}

}
