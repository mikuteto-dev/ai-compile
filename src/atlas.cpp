#include "aic/atlas.hpp"

#include "aic/errors.hpp"
#include "aic/json_min.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>

namespace aic {

namespace {

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

std::uint64_t hash_name(std::uint64_t seed, const std::string& s) {
    std::uint64_t h = 14695981039346656037ULL ^ seed;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return splitmix64(h);
}

Vector random_unit_vec(std::mt19937_64& rng, std::size_t dim) {
    if (dim == 0) {
        return {};
    }
    std::normal_distribution<double> dist(0.0, 1.0);
    Vector v(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        v[i] = dist(rng);
    }
    Vector out = normalize(v);
    if (norm2(out) < 1e-12) {
        out.assign(dim, 0.0);
        out[0] = 1.0;
    }
    return out;
}

std::vector<Vector> orthonormal_in_complement(
    std::mt19937_64& rng,
    std::size_t dim,
    std::size_t count,
    const std::vector<Vector>& forbidden) {
    std::vector<Vector> basis = gram_schmidt(forbidden, 1e-10);
    std::vector<Vector> picked;
    picked.reserve(count);

    for (std::size_t attempt = 0; attempt < count * 8 + 16 && picked.size() < count; ++attempt) {
        Vector cand = random_unit_vec(rng, dim);
        for (int pass = 0; pass < 2; ++pass) {
            for (const Vector& b : basis) {
                const double proj = dot(cand, b);
                for (std::size_t i = 0; i < dim; ++i) {
                    cand[i] -= proj * b[i];
                }
            }
            for (const Vector& p : picked) {
                const double proj = dot(cand, p);
                for (std::size_t i = 0; i < dim; ++i) {
                    cand[i] -= proj * p[i];
                }
            }
        }
        const double n = norm2(cand);
        if (n > 1e-6) {
            cand = vec_scale(cand, 1.0 / n);
            picked.push_back(std::move(cand));
        }
    }

    while (picked.size() < count && dim > 0) {
        Vector cand = random_unit_vec(rng, dim);
        for (const Vector& p : picked) {
            const double proj = dot(cand, p);
            for (std::size_t i = 0; i < dim; ++i) {
                cand[i] -= proj * p[i];
            }
        }
        if (norm2(cand) > 1e-8) {
            picked.push_back(normalize(cand));
        } else {
            Vector fallback(dim, 0.0);
            fallback[picked.size() % dim] = 1.0;
            picked.push_back(std::move(fallback));
        }
    }
    return picked;
}

Vector project_orthogonal(const Vector& v, const std::vector<Vector>& basis) {
    Vector out = v;
    const std::size_t dim = out.size();
    for (int pass = 0; pass < 2; ++pass) {
        for (const Vector& b : basis) {
            if (b.size() != dim) {
                continue;
            }
            const double p = dot(out, b);
            for (std::size_t i = 0; i < dim; ++i) {
                out[i] -= p * b[i];
            }
        }
    }
    return out;
}

JsonValue vec_to_json(const Vector& v) {
    JsonValue::Array arr;
    arr.reserve(v.size());
    for (double x : v) {
        arr.emplace_back(x);
    }
    return JsonValue(std::move(arr));
}

Vector json_to_vec(const JsonValue& j) {
    const auto& arr = j.as_array();
    Vector v;
    v.reserve(arr.size());
    for (const auto& elem : arr) {
        v.push_back(elem.as_number());
    }
    return v;
}

JsonValue vecs_to_json(const std::vector<Vector>& vs) {
    JsonValue::Array arr;
    arr.reserve(vs.size());
    for (const auto& v : vs) {
        arr.push_back(vec_to_json(v));
    }
    return JsonValue(std::move(arr));
}

std::vector<Vector> json_to_vecs(const JsonValue& j) {
    const auto& arr = j.as_array();
    std::vector<Vector> vs;
    vs.reserve(arr.size());
    for (const auto& elem : arr) {
        vs.push_back(json_to_vec(elem));
    }
    return vs;
}

JsonValue mat_to_json(const Matrix& m) {
    JsonValue::Array rows;
    rows.reserve(m.rows);
    for (std::size_t r = 0; r < m.rows; ++r) {
        rows.push_back(vec_to_json(m.row(r)));
    }
    return JsonValue(std::move(rows));
}

Matrix json_to_mat(const JsonValue& j) {
    const auto& rows = j.as_array();
    if (rows.empty()) {
        return Matrix(0, 0);
    }
    const std::size_t r_cnt = rows.size();
    const std::size_t c_cnt = rows[0].as_array().size();
    Matrix m(r_cnt, c_cnt, 0.0);
    for (std::size_t r = 0; r < r_cnt; ++r) {
        Vector row_v = json_to_vec(rows[r]);
        if (row_v.size() != c_cnt) {
            throw IOError("Inconsistent matrix row width in NeuralAtlas JSON");
        }
        m.set_row(r, row_v);
    }
    return m;
}

std::vector<SVDMode> build_field_modes(
    const TransformerModel& model,
    int layer_idx,
    std::size_t rank,
    std::uint64_t sym_seed,
    std::uint64_t global_seed) {
    const std::size_t d = static_cast<std::size_t>(model.config.hidden_size);
    if (d == 0 || rank == 0) {
        return {};
    }
    const std::size_t r_eff = std::min(rank, d);

    Matrix base(d, d, 0.0);
    if (model.config.num_layers > 0) {
        const int clamped_l = std::clamp(layer_idx, 0, model.config.num_layers - 1);
        const std::string key = "model.layers." + std::to_string(clamped_l) + ".self_attn.o_proj.weight";
        auto it = model.matrices.find(key);
        if (it != model.matrices.end() && it->second.rows == d && it->second.cols == d) {
            base = it->second;
        }
    }

    std::mt19937_64 shared_rng(splitmix64(global_seed ^ 0x55aa33ccULL));
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(static_cast<double>(d)));
    Matrix shared(d, d, 0.0);
    for (double& x : shared.data) {
        x = dist(shared_rng);
    }

    std::mt19937_64 sym_rng(sym_seed);
    Matrix specific(d, d, 0.0);
    for (double& x : specific.data) {
        x = dist(sym_rng);
    }

    Matrix combined(d, d, 0.0);
    const bool has_base = (frob_norm(base) > 1e-8);
    const double w_base = has_base ? 0.70 : 0.0;
    const double w_shared = has_base ? 0.15 : 0.35;
    const double w_specific = has_base ? 0.15 : 0.65;
    for (std::size_t i = 0; i < combined.data.size(); ++i) {
        combined.data[i] = w_base * base.data[i] + w_shared * shared.data[i] + w_specific * specific.data[i];
    }
    Matrix sym(d, d, 0.0);
    for (std::size_t r = 0; r < d; ++r) {
        for (std::size_t c = 0; c < d; ++c) {
            sym(r, c) = 0.5 * (combined(r, c) + combined(c, r));
        }
    }

    std::vector<SVDMode> modes = svd_top_k(sym, r_eff, 40, sym_seed);
    for (std::size_t i = 0; i < modes.size(); ++i) {
        if (modes[i].sigma < 1e-6) {
            modes[i].sigma = 1.0 / static_cast<double>(i + 1);
        }
    }
    return modes;
}

Matrix make_orthogonal_bridge(std::size_t d, std::uint64_t seed) {
    if (d == 0) {
        return Matrix(0, 0);
    }
    std::mt19937_64 rng(seed);
    std::vector<Vector> raw;
    raw.reserve(d);
    for (std::size_t i = 0; i < d; ++i) {
        Vector v(d, 0.0);
        v[i] = 1.0;
        std::normal_distribution<double> eps_dist(0.0, 0.15);
        for (std::size_t j = 0; j < d; ++j) {
            v[j] += eps_dist(rng);
        }
        raw.push_back(std::move(v));
    }
    std::vector<Vector> orth = gram_schmidt(raw, 1e-10);
    while (orth.size() < d) {
        Vector e = random_unit_vec(rng, d);
        for (const Vector& q : orth) {
            const double p = dot(e, q);
            for (std::size_t j = 0; j < d; ++j) {
                e[j] -= p * q[j];
            }
        }
        if (norm2(e) > 1e-8) {
            orth.push_back(normalize(e));
        }
    }
    Matrix m(d, d, 0.0);
    for (std::size_t r = 0; r < d; ++r) {
        m.set_row(r, orth[r]);
    }
    return m;
}

}

void NeuralAtlas::save(const std::string& path) const {
    const std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }

    JsonValue root = JsonValue::object();
    root["arch_name"] = arch_name;
    root["hidden_size"] = hidden_size;
    root["num_layers"] = num_layers;

    JsonValue syms_obj = JsonValue::object();
    for (const auto& [name, sb] : symbols) {
        JsonValue s = JsonValue::object();
        s["name"] = sb.name;
        s["kind"] = sb.kind;
        JsonValue lr = JsonValue::array();
        lr.as_array().emplace_back(sb.layer_range.first);
        lr.as_array().emplace_back(sb.layer_range.second);
        s["layer_range"] = std::move(lr);
        s["direction"] = vec_to_json(sb.direction);
        s["subspace_basis"] = vecs_to_json(sb.subspace_basis);

        JsonValue slots_obj = JsonValue::object();
        for (const auto& [pole_name, tup] : sb.pole_slots) {
            JsonValue arr = JsonValue::array();
            arr.as_array().emplace_back(std::get<0>(tup));
            arr.as_array().emplace_back(std::get<1>(tup));
            arr.as_array().emplace_back(std::get<2>(tup));
            slots_obj[pole_name] = std::move(arr);
        }
        s["pole_slots"] = std::move(slots_obj);

        JsonValue fmodes = JsonValue::array();
        for (const auto& mode : sb.field_modes) {
            JsonValue fm = JsonValue::object();
            fm["sigma"] = mode.sigma;
            fm["u"] = vec_to_json(mode.u);
            fm["v"] = vec_to_json(mode.v);
            fmodes.as_array().push_back(std::move(fm));
        }
        s["field_modes"] = std::move(fmodes);
        s["preserve_keys"] = vecs_to_json(sb.preserve_keys);
        s["manifold_plane"] = vecs_to_json(sb.manifold_plane);
        if (!sb.rodata_key_vectors.empty()) {
            JsonValue rk = JsonValue::object();
            for (const auto& [k, v] : sb.rodata_key_vectors) {
                rk[k] = vec_to_json(v);
            }
            s["rodata_key_vectors"] = std::move(rk);
        }
        if (!sb.rodata_val_vectors.empty()) {
            JsonValue rv = JsonValue::object();
            for (const auto& [k, v] : sb.rodata_val_vectors) {
                rv[k] = vec_to_json(v);
            }
            s["rodata_val_vectors"] = std::move(rv);
        }
        syms_obj[name] = std::move(s);
    }
    root["symbols"] = std::move(syms_obj);

    JsonValue pres_obj = JsonValue::object();
    for (const auto& [k, vecs] : preserve_directions) {
        pres_obj[k] = vecs_to_json(vecs);
    }
    root["preserve_directions"] = std::move(pres_obj);

    JsonValue dormant_arr = JsonValue::array();
    for (const auto& [l, slot] : dormant_poles) {
        JsonValue pair_arr = JsonValue::array();
        pair_arr.as_array().emplace_back(l);
        pair_arr.as_array().emplace_back(slot);
        dormant_arr.as_array().push_back(std::move(pair_arr));
    }
    root["dormant_poles"] = std::move(dormant_arr);

    JsonValue kv_arr = JsonValue::array();
    for (const auto& [l, g] : occupied_kv_groups) {
        JsonValue pair_arr = JsonValue::array();
        pair_arr.as_array().emplace_back(l);
        pair_arr.as_array().emplace_back(g);
        kv_arr.as_array().push_back(std::move(pair_arr));
    }
    root["occupied_kv_groups"] = std::move(kv_arr);

    JsonValue bridges_obj = JsonValue::object();
    for (const auto& [k, m] : frame_bridges) {
        bridges_obj[k] = mat_to_json(m);
    }
    root["frame_bridges"] = std::move(bridges_obj);

    if (!mlp_cov_inv.empty()) {
        JsonValue cov_obj = JsonValue::object();
        for (const auto& [lay, m] : mlp_cov_inv) {
            cov_obj[std::to_string(lay)] = mat_to_json(m);
        }
        root["mlp_cov_inv"] = std::move(cov_obj);
    }

    if (!fisher_metrics.empty()) {
        JsonValue fm_obj = JsonValue::object();
        for (const auto& [lay, diag] : fisher_metrics) {
            JsonValue entry = JsonValue::object();
            entry["g_in"] = vec_to_json(diag.g_in);
            entry["g_out"] = vec_to_json(diag.g_out);
            fm_obj[std::to_string(lay)] = std::move(entry);
        }
        root["fisher_metrics"] = std::move(fm_obj);
    }

    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs) {
        throw IOError("Failed to open NeuralAtlas file for writing: " + path);
    }
    ofs << dump_json(root, 2) << '\n';
    if (!ofs.good()) {
        throw IOError("Failed writing NeuralAtlas file: " + path);
    }
}

NeuralAtlas NeuralAtlas::load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw IOError("Failed to open NeuralAtlas file for reading: " + path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    JsonValue root = parse_json(oss.str());
    if (!root.is_object()) {
        throw IOError("NeuralAtlas JSON root must be an object: " + path);
    }

    NeuralAtlas atlas;
    if (root.has("arch_name")) {
        atlas.arch_name = root["arch_name"].as_string();
    }
    if (root.has("hidden_size")) {
        atlas.hidden_size = static_cast<int>(root["hidden_size"].as_int());
    }
    if (root.has("num_layers")) {
        atlas.num_layers = static_cast<int>(root["num_layers"].as_int());
    }

    if (root.has("symbols") && root["symbols"].is_object()) {
        for (const auto& [name, sj] : root["symbols"].as_object()) {
            SymbolBinding sb;
            sb.name = sj.has("name") ? sj["name"].as_string() : name;
            if (sj.has("kind")) {
                sb.kind = sj["kind"].as_string();
            }
            if (sj.has("layer_range")) {
                const auto& lr = sj["layer_range"].as_array();
                if (lr.size() >= 2) {
                    sb.layer_range = {static_cast<int>(lr[0].as_int()), static_cast<int>(lr[1].as_int())};
                }
            }
            if (sj.has("direction")) {
                sb.direction = json_to_vec(sj["direction"]);
            }
            if (sj.has("subspace_basis")) {
                sb.subspace_basis = json_to_vecs(sj["subspace_basis"]);
            }
            if (sj.has("pole_slots") && sj["pole_slots"].is_object()) {
                for (const auto& [p_name, p_arr_j] : sj["pole_slots"].as_object()) {
                    const auto& p_arr = p_arr_j.as_array();
                    if (p_arr.size() >= 3) {
                        sb.pole_slots[p_name] = {
                            static_cast<int>(p_arr[0].as_int()),
                            static_cast<int>(p_arr[1].as_int()),
                            static_cast<int>(p_arr[2].as_int())};
                    }
                }
            }
            if (sj.has("field_modes") && sj["field_modes"].is_array()) {
                for (const auto& fm_j : sj["field_modes"].as_array()) {
                    SVDMode mode;
                    mode.sigma = fm_j["sigma"].as_number();
                    mode.u = json_to_vec(fm_j["u"]);
                    mode.v = json_to_vec(fm_j["v"]);
                    sb.field_modes.push_back(std::move(mode));
                }
            }
            if (sj.has("preserve_keys")) {
                sb.preserve_keys = json_to_vecs(sj["preserve_keys"]);
            }
            if (sj.has("manifold_plane")) {
                sb.manifold_plane = json_to_vecs(sj["manifold_plane"]);
            }
            if (sj.has("rodata_key_vectors") && sj["rodata_key_vectors"].is_object()) {
                for (const auto& [k, v_j] : sj["rodata_key_vectors"].as_object()) {
                    sb.rodata_key_vectors[k] = json_to_vec(v_j);
                }
            }
            if (sj.has("rodata_val_vectors") && sj["rodata_val_vectors"].is_object()) {
                for (const auto& [k, v_j] : sj["rodata_val_vectors"].as_object()) {
                    sb.rodata_val_vectors[k] = json_to_vec(v_j);
                }
            }
            atlas.symbols[name] = std::move(sb);
        }
    }

    if (root.has("preserve_directions") && root["preserve_directions"].is_object()) {
        for (const auto& [k, v_j] : root["preserve_directions"].as_object()) {
            atlas.preserve_directions[k] = json_to_vecs(v_j);
        }
    }

    if (root.has("dormant_poles") && root["dormant_poles"].is_array()) {
        for (const auto& p_j : root["dormant_poles"].as_array()) {
            const auto& arr = p_j.as_array();
            if (arr.size() >= 2) {
                atlas.dormant_poles.emplace_back(
                    static_cast<int>(arr[0].as_int()),
                    static_cast<int>(arr[1].as_int()));
            }
        }
    }

    if (root.has("occupied_kv_groups") && root["occupied_kv_groups"].is_array()) {
        for (const auto& g_j : root["occupied_kv_groups"].as_array()) {
            const auto& arr = g_j.as_array();
            if (arr.size() >= 2) {
                atlas.occupied_kv_groups.emplace(
                    static_cast<int>(arr[0].as_int()),
                    static_cast<int>(arr[1].as_int()));
            }
        }
    }

    if (root.has("frame_bridges") && root["frame_bridges"].is_object()) {
        for (const auto& [k, m_j] : root["frame_bridges"].as_object()) {
            atlas.frame_bridges[k] = json_to_mat(m_j);
        }
    }

    if (root.has("mlp_cov_inv") && root["mlp_cov_inv"].is_object()) {
        for (const auto& [k, m_j] : root["mlp_cov_inv"].as_object()) {
            atlas.mlp_cov_inv[std::stoi(k)] = json_to_mat(m_j);
        }
    }

    if (root.has("fisher_metrics") && root["fisher_metrics"].is_object()) {
        for (const auto& [k, f_j] : root["fisher_metrics"].as_object()) {
            if (f_j.is_object()) {
                LayerFisherDiag diag;
                if (f_j.has("g_in")) {
                    diag.g_in = json_to_vec(f_j["g_in"]);
                }
                if (f_j.has("g_out")) {
                    diag.g_out = json_to_vec(f_j["g_out"]);
                }
                atlas.fisher_metrics[std::stoi(k)] = std::move(diag);
            }
        }
    }

    return atlas;
}

NeuralAtlas build_default_atlas(
    const TransformerModel& model,
    const MIRModule* mir,
    std::uint64_t seed) {
    NeuralAtlas atlas;
    atlas.arch_name = model.config.arch_name;
    atlas.hidden_size = model.config.hidden_size;
    atlas.num_layers = model.config.num_layers;

    const std::size_t d = static_cast<std::size_t>(std::max(0, model.config.hidden_size));
    const int n_layers = std::max(0, model.config.num_layers);

    std::vector<int> layer_active_slots(static_cast<std::size_t>(n_layers), 0);
    std::set<std::pair<int, int>> dormant_set;

    for (int l = 0; l < n_layers; ++l) {
        const std::string prefix = "model.layers." + std::to_string(l);
        const auto gate_it = model.matrices.find(prefix + ".mlp.gate_proj.weight");
        const auto up_it = model.matrices.find(prefix + ".mlp.up_proj.weight");
        const auto down_it = model.matrices.find(prefix + ".mlp.down_proj.weight");

        std::size_t d_ff = static_cast<std::size_t>(std::max(0, model.config.intermediate_size));
        if (gate_it != model.matrices.end()) {
            d_ff = gate_it->second.rows;
        }
        if (d_ff == 0) {
            continue;
        }

        std::vector<std::pair<double, int>> slot_energies;
        slot_energies.reserve(d_ff);
        for (std::size_t s = 0; s < d_ff; ++s) {
            double e = 0.0;
            if (down_it != model.matrices.end() && s < down_it->second.cols) {
                const double dn = norm2(down_it->second.col(s));
                double gn = 1.0;
                if (gate_it != model.matrices.end() && s < gate_it->second.rows) {
                    gn = norm2(gate_it->second.row(s));
                }
                double un = 1.0;
                if (up_it != model.matrices.end() && s < up_it->second.rows) {
                    un = norm2(up_it->second.row(s));
                }
                e = dn * (gn + un);
            } else {
                e = static_cast<double>(s);
            }
            slot_energies.emplace_back(e, static_cast<int>(s));
        }
        std::stable_sort(slot_energies.begin(), slot_energies.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        const std::size_t dormant_count = std::max<std::size_t>(1, d_ff / 2);
        for (std::size_t i = 0; i < dormant_count && i < slot_energies.size(); ++i) {
            const int slot_id = slot_energies[i].second;
            atlas.dormant_poles.emplace_back(l, slot_id);
            dormant_set.emplace(l, slot_id);
        }
        layer_active_slots[static_cast<std::size_t>(l)] = slot_energies.back().second;
    }

    const std::vector<std::string> default_preserve = {
        "CoreFluency",
        "CoreReasoning",
        "Residual",
        "Fluency"
    };
    std::vector<Vector> all_preserved;
    std::size_t max_default_preserve = std::max<std::size_t>(1, d / 4);
    for (const std::string& pname : default_preserve) {
        std::mt19937_64 prng(hash_name(seed, "preserve:" + pname));
        const std::size_t vec_cnt = (d >= 16 && all_preserved.size() + 2 <= max_default_preserve) ? 2 : 1;
        if (all_preserved.size() >= d) {
            atlas.preserve_directions[pname] = {};
            continue;
        }
        const std::size_t actual_cnt = std::min(vec_cnt, std::max<std::size_t>(1, d - all_preserved.size()));
        std::vector<Vector> dirs = orthonormal_in_complement(prng, d, actual_cnt, all_preserved);
        for (const auto& v : dirs) {
            all_preserved.push_back(v);
        }
        atlas.preserve_directions[pname] = std::move(dirs);
    }

    if (d > 0) {
        atlas.frame_bridges["Early->Mid"] = make_orthogonal_bridge(d, hash_name(seed, "bridge:Early->Mid"));
        atlas.frame_bridges["Mid->Late"] = make_orthogonal_bridge(d, hash_name(seed, "bridge:Mid->Late"));
        atlas.frame_bridges["Late->Logit"] = make_orthogonal_bridge(d, hash_name(seed, "bridge:Late->Logit"));
        atlas.frame_bridges["Mid..Late->Late"] = make_orthogonal_bridge(d, hash_name(seed, "bridge:Mid..Late->Late"));
    }

    if (mir == nullptr) {
        return atlas;
    }

    std::set<std::string> preserved_names;
    for (const auto& flow : mir->flows) {
        for (const std::string& p : flow.contract.preserve) {
            preserved_names.insert(p);
            if (!atlas.preserve_directions.contains(p) && !mir->declarations.contains(p)) {
                std::mt19937_64 prng(hash_name(seed, "preserve:" + p));
                if (all_preserved.size() < d) {
                    std::vector<Vector> dirs = orthonormal_in_complement(prng, d, 1, all_preserved);
                    for (const auto& v : dirs) {
                        all_preserved.push_back(v);
                    }
                    atlas.preserve_directions[p] = std::move(dirs);
                } else {
                    atlas.preserve_directions[p] = {random_unit_vec(prng, d)};
                }
            }
        }
    }
    if (!all_preserved.empty()) {
        all_preserved = gram_schmidt(all_preserved, 1e-10);
    }

    int slot_cursor = 0;
    auto build_decl_binding = [&](const std::string& name, const MIRSymbolDecl& decl) {
        SymbolBinding sb;
        sb.name = decl.name;
        sb.kind = decl.kind;
        sb.layer_range = model.config.stage_to_layers(decl.frame.start, decl.frame.end, decl.frame.inclusive);

        const int safe_layer = (n_layers > 0)
            ? std::clamp(sb.layer_range.first, 0, n_layers - 1)
            : 0;
        std::mt19937_64 srng(hash_name(seed, "sym:" + name));

        if (decl.kind == "frame") {
            for (const auto& [dst_name, via_name] : decl.connections) {
                const std::string key1 = decl.name + "->" + dst_name;
                if (!atlas.frame_bridges.contains(key1) && d > 0) {
                    atlas.frame_bridges[key1] = make_orthogonal_bridge(d, hash_name(seed, "bridge:" + key1));
                }
                if (!via_name.empty() && !atlas.frame_bridges.contains(via_name) && d > 0) {
                    atlas.frame_bridges[via_name] = atlas.frame_bridges[key1];
                }
            }
            if (d > 0) {
                sb.direction = orthonormal_in_complement(srng, d, 1, all_preserved).front();
            }
        } else if (decl.kind == "pole" || decl.kind == "dipole") {
            int chosen_slot = -1;
            if (n_layers > 0 && model.config.intermediate_size > 0) {
                chosen_slot = layer_active_slots[static_cast<std::size_t>(safe_layer)];
                const int d_ff = model.config.intermediate_size;
                for (int probe = 0; probe < d_ff; ++probe) {
                    const int cand = (d_ff - 1 - ((slot_cursor + probe) % d_ff));
                    if (!dormant_set.contains({safe_layer, cand})) {
                        chosen_slot = cand;
                        break;
                    }
                }
                ++slot_cursor;
                sb.pole_slots["+pole"] = {safe_layer, chosen_slot, 1};
                if (decl.kind == "dipole") {
                    sb.pole_slots["-pole"] = {safe_layer, chosen_slot, -1};
                }
            }
            if (d > 0) {
                bool extracted = false;
                if (n_layers > 0 && chosen_slot >= 0) {
                    const std::string down_key =
                        "model.layers." + std::to_string(safe_layer) + ".mlp.down_proj.weight";
                    const auto dit = model.matrices.find(down_key);
                    if (dit != model.matrices.end() && dit->second.rows == d &&
                        static_cast<std::size_t>(chosen_slot) < dit->second.cols) {
                        const Vector raw_col = dit->second.col(static_cast<std::size_t>(chosen_slot));
                        const std::vector<Vector> pres_basis = gram_schmidt(all_preserved, 1e-10);
                        const Vector proj = project_orthogonal(raw_col, pres_basis);
                        if (norm2(proj) > 1e-8) {
                            sb.direction = normalize(proj);
                            extracted = true;
                        }
                    }
                }
                if (!extracted) {
                    sb.direction = orthonormal_in_complement(srng, d, 1, all_preserved).front();
                }
                sb.subspace_basis.push_back(sb.direction);
            }
        } else if (decl.kind == "subspace") {
            const std::size_t sub_dim = std::min(
                static_cast<std::size_t>(std::max(1, decl.dim)),
                std::max<std::size_t>(1, d));
            if (d > 0) {
                std::vector<Vector> basis;
                const std::vector<Vector> pres_basis = gram_schmidt(all_preserved, 1e-10);
                if (n_layers > 0) {
                    const std::string down_key =
                        "model.layers." + std::to_string(safe_layer) + ".mlp.down_proj.weight";
                    const auto dit = model.matrices.find(down_key);
                    if (dit != model.matrices.end() && dit->second.rows == d && dit->second.cols > 0) {
                        const std::size_t k_svd = std::min({sub_dim, d, dit->second.cols});
                        const std::vector<SVDMode> modes = svd_top_k(
                            dit->second,
                            k_svd,
                            40,
                            hash_name(seed, "sub:" + name));
                        std::vector<Vector> proj_vecs;
                        for (const auto& m : modes) {
                            if (m.sigma > 1e-12 && m.u.size() == d) {
                                Vector p = project_orthogonal(m.u, pres_basis);
                                if (norm2(p) > 1e-8) {
                                    proj_vecs.push_back(std::move(p));
                                }
                            }
                        }
                        basis = gram_schmidt(proj_vecs, 1e-10);
                        if (basis.size() > sub_dim) {
                            basis.resize(sub_dim);
                        }
                    }
                }
                if (basis.size() < sub_dim) {
                    std::vector<Vector> forbidden = all_preserved;
                    forbidden.insert(forbidden.end(), basis.begin(), basis.end());
                    std::vector<Vector> pad = orthonormal_in_complement(
                        srng,
                        d,
                        sub_dim - basis.size(),
                        forbidden);
                    for (auto& v : pad) {
                        if (basis.size() < sub_dim) {
                            basis.push_back(std::move(v));
                        }
                    }
                }
                sb.subspace_basis = std::move(basis);
                sb.direction = sb.subspace_basis[0];
            }
        } else if (decl.kind == "manifold") {
            const bool is_t2 = (decl.topology == "T2" || decl.topology == "Torus" || decl.topology == "T<2>");
            const std::size_t plane_dim = is_t2
                ? std::min<std::size_t>(5, d)
                : ((decl.topology == "S1" || decl.curvature > 0.0)
                    ? std::min<std::size_t>(3, d)
                    : std::min<std::size_t>(static_cast<std::size_t>(std::max(1, decl.dim)), d));
            if (d > 0) {
                std::vector<Vector> plane = orthonormal_in_complement(srng, d, plane_dim, all_preserved);
                sb.manifold_plane = plane;
                sb.subspace_basis = std::move(plane);
                sb.direction = (sb.manifold_plane.size() > 1) ? sb.manifold_plane[1] : sb.manifold_plane[0];
            }
        } else if (decl.kind == "field") {
            const std::size_t r_eff = std::min(
                static_cast<std::size_t>(std::max(1, decl.rank)),
                std::max<std::size_t>(1, d));
            sb.field_modes = build_field_modes(model, safe_layer, r_eff, hash_name(seed, "field:" + name), seed);
            if (!sb.field_modes.empty()) {
                sb.direction = sb.field_modes[0].u;
            } else if (d > 0) {
                sb.direction = orthonormal_in_complement(srng, d, 1, all_preserved).front();
            }
        } else if (decl.kind == "rodata") {
            const std::size_t key_count = std::max<std::size_t>(1, decl.rodata_entries.size());
            if (d > 0) {
                sb.preserve_keys = orthonormal_in_complement(srng, d, std::min(key_count, d), all_preserved);
                sb.direction = orthonormal_in_complement(srng, d, 1, all_preserved).front();
            }
        } else {
            if (d > 0) {
                sb.direction = orthonormal_in_complement(srng, d, 1, all_preserved).front();
            }
        }

        return sb;
    };

    auto append_binding_to_preserved = [&](const SymbolBinding& sb) {
        auto add_vec = [&](const Vector& v) {
            if (v.size() == d && norm2(v) > 1e-12) {
                all_preserved.push_back(v);
            }
        };
        add_vec(sb.direction);
        for (const auto& v : sb.subspace_basis) {
            add_vec(v);
        }
        for (const auto& v : sb.manifold_plane) {
            add_vec(v);
        }
        for (const auto& v : sb.preserve_keys) {
            add_vec(v);
        }
        for (const auto& m : sb.field_modes) {
            add_vec(m.u);
            add_vec(m.v);
        }
        if (!all_preserved.empty()) {
            all_preserved = gram_schmidt(all_preserved, 1e-10);
        }
    };

    for (const auto& [name, decl] : mir->declarations) {
        if (!preserved_names.contains(name)) {
            continue;
        }
        SymbolBinding sb = build_decl_binding(name, decl);
        append_binding_to_preserved(sb);
        atlas.symbols[name] = std::move(sb);
    }

    for (const auto& [name, decl] : mir->declarations) {
        if (preserved_names.contains(name)) {
            continue;
        }
        atlas.symbols[name] = build_decl_binding(name, decl);
    }

    return atlas;
}

void calibrate_atlas_empirical(
    NeuralAtlas& atlas,
    const TransformerModel& model,
    const MIRModule* mir,
    const Tokenizer* tokenizer) {
    const int n_layers = std::max(0, model.config.num_layers);
    const std::size_t d = static_cast<std::size_t>(std::max(0, model.config.hidden_size));
    if (n_layers <= 0 || d == 0) {
        return;
    }

    int vocab = std::max(1, model.config.vocab_size);
    const auto emb_it = model.matrices.find("model.embed_tokens.weight");
    if (emb_it != model.matrices.end() && emb_it->second.rows > 0) {
        vocab = static_cast<int>(emb_it->second.rows);
    }

    auto tokenize_text = [&](const std::string& s) -> std::vector<int> {
        if (tokenizer != nullptr) {
            const std::vector<int> encoded = tokenizer->encode(s);
            std::vector<int> clamped;
            clamped.reserve(encoded.size());
            for (int id : encoded) {
                clamped.push_back(((id % vocab) + vocab) % vocab);
            }
            if (!clamped.empty()) {
                return clamped;
            }
        }
        std::vector<int> bytes;
        bytes.reserve(s.size());
        for (unsigned char c : s) {
            bytes.push_back(static_cast<int>(c) % vocab);
        }
        if (bytes.empty()) {
            bytes.push_back(0);
        }
        return bytes;
    };

    std::vector<std::vector<int>> cal_seqs;
    cal_seqs.reserve(16);
    for (int s = 0; s < 8; ++s) {
        std::vector<int> seq;
        seq.reserve(8);
        for (int t = 0; t < 8; ++t) {
            seq.push_back((s * 13 + t * 7 + t * t) % vocab);
        }
        cal_seqs.push_back(std::move(seq));
    }

    auto arg_string = [](const std::map<std::string, ArgValue>& args,
                         const std::string& key,
                         const std::string& fallback) -> std::string {
        auto it = args.find(key);
        if (it == args.end()) {
            return fallback;
        }
        if (std::holds_alternative<std::string>(it->second)) {
            return std::get<std::string>(it->second);
        }
        return fallback;
    };

    auto push_cal_str = [&](const std::string& s) {
        if (s.empty() || cal_seqs.size() >= 16) {
            return;
        }
        std::vector<int> ids = tokenize_text(s);
        for (const auto& existing : cal_seqs) {
            if (existing == ids) {
                return;
            }
        }
        cal_seqs.push_back(std::move(ids));
    };

    if (mir != nullptr) {
        for (const auto& [sym_name, decl] : mir->declarations) {
            (void)sym_name;
            for (const auto& anchor : decl.anchors) {
                for (const auto& av : anchor.pos_args) {
                    if (std::holds_alternative<std::string>(av)) {
                        push_cal_str(std::get<std::string>(av));
                    } else if (std::holds_alternative<std::vector<std::string>>(av)) {
                        for (const auto& item : std::get<std::vector<std::string>>(av)) {
                            push_cal_str(item);
                        }
                    }
                }
                for (const auto& [ak, av] : anchor.named_args) {
                    (void)ak;
                    if (std::holds_alternative<std::string>(av)) {
                        push_cal_str(std::get<std::string>(av));
                    } else if (std::holds_alternative<std::vector<std::string>>(av)) {
                        for (const auto& item : std::get<std::vector<std::string>>(av)) {
                            push_cal_str(item);
                        }
                    }
                }
            }
            for (const auto& entry : decl.rodata_entries) {
                push_cal_str(entry.key);
                push_cal_str(entry.value);
            }
        }
        for (const auto& flow : mir->flows) {
            auto collect_op_strs = [&](auto& self, const MIROp& op) -> void {
                if (op.op == "rodata_insert" || op.op == "rodata_prune" || op.op == "rodata_rewrite") {
                    push_cal_str(arg_string(op.args, "key", ""));
                    if (op.op == "rodata_rewrite") {
                        push_cal_str(arg_string(op.args, "new", ""));
                    } else if (op.op == "rodata_insert") {
                        push_cal_str(arg_string(op.args, "value", ""));
                    }
                }
                for (const auto& sub : op.body) {
                    self(self, sub);
                }
                for (const auto& sub : op.else_body) {
                    self(self, sub);
                }
            };
            for (const auto& op : flow.ops) {
                collect_op_strs(collect_op_strs, op);
            }
        }
    }

    const std::vector<Matrix> cov_invs = compute_mlp_key_covariances(model, cal_seqs, 1e-2);
    atlas.mlp_cov_inv.clear();
    for (int l = 0; l < static_cast<int>(cov_invs.size()); ++l) {
        atlas.mlp_cov_inv[l] = cov_invs[static_cast<std::size_t>(l)];
    }

    const std::vector<LayerFisherDiag> f_diags = compute_fisher_diagonals(model, cal_seqs);
    atlas.fisher_metrics.clear();
    for (int l = 0; l < static_cast<int>(f_diags.size()); ++l) {
        atlas.fisher_metrics[l] = f_diags[static_cast<std::size_t>(l)];
    }

    if (mir == nullptr) {
        return;
    }

    std::vector<Vector> raw_preserve;
    for (const auto& [pname, pvecs] : atlas.preserve_directions) {
        (void)pname;
        for (const auto& pv : pvecs) {
            if (pv.size() == d) {
                raw_preserve.push_back(pv);
            }
        }
    }
    std::set<std::string> preserved_names;
    for (const auto& flow : mir->flows) {
        for (const std::string& pn : flow.contract.preserve) {
            preserved_names.insert(pn);
        }
    }
    for (const std::string& pn : preserved_names) {
        auto sit = atlas.symbols.find(pn);
        if (sit == atlas.symbols.end()) {
            continue;
        }
        const SymbolBinding& psb = sit->second;
        if (psb.direction.size() == d) {
            raw_preserve.push_back(psb.direction);
        }
        for (const auto& v : psb.subspace_basis) {
            if (v.size() == d) {
                raw_preserve.push_back(v);
            }
        }
        for (const auto& v : psb.manifold_plane) {
            if (v.size() == d) {
                raw_preserve.push_back(v);
            }
        }
        for (const auto& v : psb.preserve_keys) {
            if (v.size() == d) {
                raw_preserve.push_back(v);
            }
        }
        for (const auto& m : psb.field_modes) {
            if (m.u.size() == d) {
                raw_preserve.push_back(m.u);
            }
            if (m.v.size() == d) {
                raw_preserve.push_back(m.v);
            }
        }
    }
    const std::vector<Vector> preserve_basis = gram_schmidt(raw_preserve, 1e-10);

    const Matrix* out_mat = nullptr;
    const auto lm_it = model.matrices.find("lm_head.weight");
    if (lm_it != model.matrices.end() && lm_it->second.cols == d && lm_it->second.rows > 0) {
        out_mat = &lm_it->second;
    } else if (emb_it != model.matrices.end() && emb_it->second.cols == d && emb_it->second.rows > 0) {
        out_mat = &emb_it->second;
    }

    struct CachedTrace {
        ForwardTrace trace;
        std::size_t seq_len = 1;
    };
    std::map<std::string, CachedTrace> trace_cache;

    auto get_cached_trace = [&](const std::string& key_str) -> const CachedTrace& {
        auto it = trace_cache.find(key_str);
        if (it != trace_cache.end()) {
            return it->second;
        }
        const std::vector<int> key_ids = tokenize_text(key_str);
        auto [logits, tr] = model.forward(key_ids, {}, true);
        (void)logits;
        CachedTrace ct{std::move(tr), key_ids.size()};
        return trace_cache.emplace(key_str, std::move(ct)).first->second;
    };

    auto extract_layer_key = [&](const CachedTrace& ct, int layer_idx) -> Vector {
        const int l = std::clamp(layer_idx, 0, n_layers - 1);
        const std::size_t ul = static_cast<std::size_t>(l);
        if (ul >= ct.trace.mlp_poles.size() || ct.trace.mlp_poles[ul].empty()) {
            return {};
        }
        const std::size_t d_ff = ct.trace.mlp_poles[ul].size();
        const std::size_t last_t = (ct.seq_len > 0) ? (ct.seq_len - 1) : 0;
        Vector k_star(d_ff, 0.0);
        for (std::size_t i = 0; i < d_ff; ++i) {
            if (last_t < ct.trace.mlp_poles[ul][i].act.size()) {
                k_star[i] = ct.trace.mlp_poles[ul][i].act[last_t];
            }
        }
        return k_star;
    };

    auto store_rodata_key = [&](SymbolBinding& sb,
                                const std::string& key_str,
                                int primary_layer,
                                int range_lo,
                                int range_hi) {
        const CachedTrace& ct = get_cached_trace(key_str);
        const int l_prim = std::clamp(primary_layer, 0, n_layers - 1);
        Vector k_prim = extract_layer_key(ct, l_prim);
        if (!k_prim.empty()) {
            sb.rodata_key_vectors[key_str] = k_prim;
            sb.rodata_key_vectors["@L" + std::to_string(l_prim) + ":" + key_str] = std::move(k_prim);
        }
        const int lo = std::clamp(range_lo, 0, n_layers - 1);
        const int hi = std::clamp(std::max(range_lo + 1, range_hi), lo + 1, n_layers);
        for (int l = lo; l < hi; ++l) {
            Vector k_l = extract_layer_key(ct, l);
            if (!k_l.empty()) {
                sb.rodata_key_vectors["@L" + std::to_string(l) + ":" + key_str] = std::move(k_l);
            }
        }
    };

    auto store_rodata_val = [&](SymbolBinding& sb, const std::string& val_str) {
        if (out_mat == nullptr || out_mat->rows == 0) {
            return;
        }
        const std::vector<int> val_ids = tokenize_text(val_str);
        const int tok_val = val_ids.empty()
            ? 0
            : (val_ids[0] % static_cast<int>(out_mat->rows));
        Vector row = out_mat->row(static_cast<std::size_t>(tok_val));
        Vector v_dir = (norm2(row) > 1e-12)
            ? normalize(row)
            : Vector(d, 1.0 / std::sqrt(static_cast<double>(d)));
        const Vector v_proj = project_orthogonal(v_dir, preserve_basis);
        if (norm2(v_proj) > 1e-8) {
            v_dir = normalize(v_proj);
        }
        Vector v_star = vec_scale(v_dir, 8.0);
        if (norm2(v_star) <= 1e-8) {
            v_star.assign(d, 0.0);
            v_star[0] = 8.0;
        }
        sb.rodata_val_vectors[val_str] = std::move(v_star);
    };

    for (const auto& [name, decl] : mir->declarations) {
        if (decl.kind != "rodata") {
            continue;
        }
        SymbolBinding& sb = atlas.symbols[name];
        if (sb.name.empty()) {
            sb.name = name;
            sb.kind = decl.kind;
            sb.layer_range = model.config.stage_to_layers(
                decl.frame.start,
                decl.frame.end,
                decl.frame.inclusive);
        }
        const int l_rodata = std::clamp(sb.layer_range.first, 0, n_layers - 1);
        for (const auto& entry : decl.rodata_entries) {
            if (!entry.key.empty()) {
                store_rodata_key(
                    sb,
                    entry.key,
                    l_rodata,
                    sb.layer_range.first,
                    sb.layer_range.second);
            }
            if (!entry.value.empty()) {
                store_rodata_val(sb, entry.value);
            }
        }
    }

    for (const auto& flow : mir->flows) {
        const auto lr = model.config.stage_to_layers(
            flow.stage.start,
            flow.stage.end,
            flow.stage.inclusive);
        const int l0 = std::clamp(lr.first, 0, n_layers - 1);
        const int l1 = std::clamp(std::max(lr.first + 1, lr.second), l0 + 1, n_layers);
        const int avail = std::max(1, l1 - l0);

        int cur_step = 0;
        bool step_has_mlp = false;
        auto step_to_layer = [&](int step) -> int {
            if (step < avail) {
                return l0 + step;
            }
            return std::clamp(l0 + (step % avail), 0, n_layers - 1);
        };

        auto visit_op = [&](auto& self, const MIROp& op, int step) -> void {
            const int phys_layer = step_to_layer(step);
            if (op.op == "rodata_insert" || op.op == "rodata_prune" || op.op == "rodata_rewrite") {
                SymbolBinding& sb = atlas.symbols[op.target];
                if (sb.name.empty()) {
                    sb.name = op.target;
                    sb.kind = "rodata";
                    sb.layer_range = lr;
                }
                const std::string key_str = arg_string(op.args, "key", op.target);
                store_rodata_key(sb, key_str, phys_layer, l0, l1);
                if (op.op != "rodata_prune") {
                    const std::string val_str = (op.op == "rodata_rewrite")
                        ? arg_string(op.args, "new", "rewritten")
                        : arg_string(op.args, "value", "inserted");
                    store_rodata_val(sb, val_str);
                }
                return;
            }
            if (op.op == "when") {
                for (const auto& sub : op.body) {
                    self(self, sub, step);
                }
                for (const auto& sub : op.else_body) {
                    self(self, sub, step);
                }
            }
        };

        for (const auto& op : flow.ops) {
            if (op.op == "attend") {
                if (step_has_mlp) {
                    ++cur_step;
                    step_has_mlp = false;
                }
                visit_op(visit_op, op, cur_step);
            } else if (op.op == "frame_transport") {
                visit_op(visit_op, op, cur_step);
                step_has_mlp = true;
            } else if (op.op == "when") {
                if (step_has_mlp) {
                    ++cur_step;
                    step_has_mlp = false;
                }
                visit_op(visit_op, op, cur_step);
                step_has_mlp = true;
            } else {
                visit_op(visit_op, op, cur_step);
            }
        }
    }
}

}
