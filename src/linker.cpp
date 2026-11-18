#include "aic/linker.hpp"

#include "aic/errors.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace aic {

namespace {

constexpr double kPi = 3.14159265358979323846;

std::uint64_t splitmix64_link(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

std::uint64_t fnv1a_64(const std::string& s, std::uint64_t seed = 0) {
    std::uint64_t h = 14695981039346656037ULL ^ seed;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return splitmix64_link(h);
}

std::vector<int> tokenize_prompt(const std::string& text, int vocab_size) {
    const int vocab = std::max(1, vocab_size);
    std::vector<std::string> pieces;
    std::string cur;
    for (unsigned char c : text) {
        if (std::isalnum(c)) {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else if (!cur.empty()) {
            pieces.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) {
        pieces.push_back(cur);
    }
    if (pieces.empty()) {
        pieces.push_back(text.empty() ? std::string("empty") : text);
    }
    if (pieces.size() < 4 && !text.empty()) {
        for (std::size_t i = 0; i < text.size() && pieces.size() < 6; ++i) {
            const std::size_t len = std::min<std::size_t>(3, text.size() - i);
            pieces.push_back(text.substr(i, len) + "#" + std::to_string(i));
        }
    }

    std::vector<int> ids;
    ids.reserve(std::min<std::size_t>(24, std::max<std::size_t>(4, pieces.size())));
    for (std::size_t i = 0; i < pieces.size() && ids.size() < 24; ++i) {
        const std::uint64_t h = fnv1a_64(pieces[i], static_cast<std::uint64_t>(i) * 1315423911ULL);
        ids.push_back(static_cast<int>(h % static_cast<std::uint64_t>(vocab)));
    }
    while (ids.size() < 4) {
        const std::uint64_t h = fnv1a_64(text, static_cast<std::uint64_t>(ids.size()) + 17ULL);
        ids.push_back(static_cast<int>(h % static_cast<std::uint64_t>(vocab)));
    }
    return ids;
}

Vector seeded_vec(std::size_t d, std::uint64_t seed) {
    if (d == 0) {
        return {};
    }
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Vector v(d);
    for (std::size_t i = 0; i < d; ++i) {
        v[i] = dist(rng);
    }
    Vector n = normalize(v);
    if (norm2(n) < 1e-12) {
        n.assign(d, 0.0);
        n[0] = 1.0;
    }
    return n;
}

Vector subtract_basis(Vector v, const std::vector<Vector>& basis) {
    const std::size_t d = v.size();
    for (int pass = 0; pass < 2; ++pass) {
        for (const Vector& b : basis) {
            if (b.size() != d) {
                continue;
            }
            const double p = dot(v, b);
            for (std::size_t i = 0; i < d; ++i) {
                v[i] -= p * b[i];
            }
        }
    }
    return v;
}

Vector project_to_complement(const Vector& v, const std::vector<Vector>& basis, std::uint64_t fallback_seed) {
    const std::size_t d = v.size();
    if (d == 0) {
        return {};
    }
    Vector proj = subtract_basis(v, basis);
    if (norm2(proj) > 1e-8) {
        return normalize(proj);
    }
    for (std::size_t attempt = 0; attempt < 32; ++attempt) {
        Vector cand = subtract_basis(seeded_vec(d, fallback_seed + attempt * 7919ULL), basis);
        if (norm2(cand) > 1e-8) {
            return normalize(cand);
        }
    }
    Vector e(d, 0.0);
    e[fallback_seed % d] = 1.0;
    return e;
}

std::vector<Vector> orthonormalize_complement(
    const std::vector<Vector>& cands,
    std::size_t count,
    std::size_t d,
    const std::vector<Vector>& forbidden,
    std::uint64_t seed) {
    if (d == 0 || count == 0) {
        return {};
    }
    std::vector<Vector> out;
    out.reserve(count);
    std::vector<Vector> full_basis = gram_schmidt(forbidden, 1e-10);

    for (const Vector& raw : cands) {
        if (out.size() >= count) {
            break;
        }
        if (raw.size() != d) {
            continue;
        }
        Vector w = subtract_basis(raw, full_basis);
        if (norm2(w) > 1e-7) {
            w = normalize(w);
            out.push_back(w);
            full_basis.push_back(w);
        }
    }

    for (std::size_t attempt = 0; attempt < count * 16 + 32 && out.size() < count; ++attempt) {
        Vector w = subtract_basis(seeded_vec(d, seed + attempt * 104729ULL), full_basis);
        if (norm2(w) > 1e-7) {
            w = normalize(w);
            out.push_back(w);
            full_basis.push_back(w);
        }
    }

    while (out.size() < count) {
        Vector w = seeded_vec(d, seed + out.size() * 65537ULL);
        for (const Vector& prev : out) {
            const double p = dot(w, prev);
            for (std::size_t i = 0; i < d; ++i) {
                w[i] -= p * prev[i];
            }
        }
        if (norm2(w) > 1e-9) {
            out.push_back(normalize(w));
        } else {
            Vector e(d, 0.0);
            e[out.size() % d] = 1.0;
            out.push_back(std::move(e));
        }
    }
    return out;
}

std::vector<std::string> arg_to_strings(const ArgValue& val) {
    if (const auto* sv = std::get_if<std::vector<std::string>>(&val)) {
        return *sv;
    }
    if (const auto* s = std::get_if<std::string>(&val)) {
        if (!s->empty()) {
            return {*s};
        }
    }
    return {};
}

void collect_anchor_prompts(
    const MIRSymbolDecl& decl,
    std::vector<std::string>& pos_prompts,
    std::vector<std::string>& neg_prompts,
    std::vector<std::string>& general_prompts) {
    for (const auto& anc : decl.anchors) {
        if (anc.named_args.contains("pos")) {
            auto ps = arg_to_strings(anc.named_args.at("pos"));
            pos_prompts.insert(pos_prompts.end(), ps.begin(), ps.end());
        }
        if (anc.named_args.contains("neg")) {
            auto ns = arg_to_strings(anc.named_args.at("neg"));
            neg_prompts.insert(neg_prompts.end(), ns.begin(), ns.end());
        }
        if (anc.named_args.contains("prompts")) {
            auto gs = arg_to_strings(anc.named_args.at("prompts"));
            general_prompts.insert(general_prompts.end(), gs.begin(), gs.end());
        }
        for (const auto& pval : anc.pos_args) {
            auto gs = arg_to_strings(pval);
            general_prompts.insert(general_prompts.end(), gs.begin(), gs.end());
        }
        for (const auto& [k, v] : anc.named_args) {
            if (k != "pos" && k != "neg" && k != "prompts" && k != "curvature_band") {
                auto gs = arg_to_strings(v);
                general_prompts.insert(general_prompts.end(), gs.begin(), gs.end());
            }
        }
    }
}

std::string replace_template_slots(std::string tmpl, const std::string& key, const std::string& val) {
    auto rep = [](std::string& s, const std::string& pat, const std::string& to) {
        std::size_t pos = 0;
        while ((pos = s.find(pat, pos)) != std::string::npos) {
            s.replace(pos, pat.size(), to);
            pos += to.size();
        }
    };
    rep(tmpl, "{subject}", key);
    rep(tmpl, "{object}", val);
    rep(tmpl, "{x}", key);
    rep(tmpl, "{y}", val);
    return tmpl;
}

std::vector<int> encode_probe_tokens(
    const std::string& prompt,
    int vocab_size,
    const Tokenizer* tokenizer) {
    const int vocab = std::max(1, vocab_size);
    if (tokenizer != nullptr) {
        std::vector<int> ids = tokenizer->encode(prompt);
        if (!ids.empty()) {
            for (int& tid : ids) {
                if (tid < 0 || tid >= vocab) {
                    tid = ((tid % vocab) + vocab) % vocab;
                }
            }
            return ids;
        }
    }
    return tokenize_prompt(prompt, vocab);
}

class ProbeCache {
public:
    ProbeCache(const TransformerModel& m, const Tokenizer* tok)
        : model_(&m), config_(m.config), tokenizer_(tok) {}

    ProbeCache(std::string model_dir, ModelConfig config, const Tokenizer* tok)
        : model_dir_(std::move(model_dir)), config_(std::move(config)), tokenizer_(tok) {}

    const ForwardTrace& trace_for(const std::string& prompt) {
        auto it = cache_.find(prompt);
        if (it != cache_.end()) {
            return it->second;
        }
        const std::vector<int> toks = encode_probe_tokens(prompt, config_.vocab_size, tokenizer_);
        ForwardTrace tr;
        if (model_ != nullptr) {
            auto [logits, full_trace] = model_->forward(toks, {}, true);
            (void)logits;
            tr = std::move(full_trace);
        } else {
            tr = trace_checkpoint_streaming(model_dir_, toks);
        }
        auto [ins_it, ok] = cache_.emplace(prompt, std::move(tr));
        (void)ok;
        return ins_it->second;
    }

    Vector layer_repr(const std::string& prompt, int layer_idx) {
        const ForwardTrace& tr = trace_for(prompt);
        const std::size_t d = static_cast<std::size_t>(std::max(0, config_.hidden_size));
        if (tr.residual.empty() || d == 0) {
            return vec_zeros(d);
        }
        const std::size_t idx = static_cast<std::size_t>(
            std::clamp(layer_idx + 1, 0, static_cast<int>(tr.residual.size()) - 1));
        const auto& seq_vecs = tr.residual[idx];
        if (seq_vecs.empty()) {
            return vec_zeros(d);
        }
        Vector mean_v(d, 0.0);
        for (const Vector& v : seq_vecs) {
            for (std::size_t i = 0; i < d; ++i) {
                mean_v[i] += v[i];
            }
        }
        const double inv_n = 1.0 / static_cast<double>(seq_vecs.size());
        for (double& x : mean_v) {
            x *= inv_n;
        }
        const Vector& last_v = seq_vecs.back();
        Vector out(d);
        for (std::size_t i = 0; i < d; ++i) {
            out[i] = 0.5 * mean_v[i] + 0.5 * last_v[i];
        }
        return out;
    }

private:
    const TransformerModel* model_ = nullptr;
    std::string model_dir_;
    ModelConfig config_;
    const Tokenizer* tokenizer_ = nullptr;
    std::map<std::string, ForwardTrace> cache_;
};

Vector mean_vec(const std::vector<Vector>& vs, std::size_t d) {
    Vector out(d, 0.0);
    if (vs.empty()) {
        return out;
    }
    for (const auto& v : vs) {
        for (std::size_t i = 0; i < d && i < v.size(); ++i) {
            out[i] += v[i];
        }
    }
    const double inv = 1.0 / static_cast<double>(vs.size());
    for (double& x : out) {
        x *= inv;
    }
    return out;
}

double variance_sum(const std::vector<Vector>& vs, const Vector& mu) {
    double acc = 0.0;
    for (const auto& v : vs) {
        const double dn = norm2(vec_sub(v, mu));
        acc += dn * dn;
    }
    return vs.empty() ? 0.0 : (acc / static_cast<double>(vs.size()));
}

int pick_best_contrastive_layer(
    ProbeCache& cache,
    const std::vector<std::string>& pos_prompts,
    const std::vector<std::string>& neg_prompts,
    int l0,
    int l1,
    std::size_t d) {
    int best_l = l0;
    double best_score = -1.0;
    for (int l = l0; l < l1; ++l) {
        std::vector<Vector> p_vecs;
        std::vector<Vector> n_vecs;
        for (const auto& p : pos_prompts) {
            p_vecs.push_back(cache.layer_repr(p, l));
        }
        for (const auto& n : neg_prompts) {
            n_vecs.push_back(cache.layer_repr(n, l));
        }
        const Vector mu_p = mean_vec(p_vecs, d);
        const Vector mu_n = mean_vec(n_vecs, d);
        const double diff_n = norm2(vec_sub(mu_p, mu_n));
        const double var_sum = variance_sum(p_vecs, mu_p) + variance_sum(n_vecs, mu_n);
        const double score = (diff_n * diff_n) / (var_sum + 1e-6);
        if (score > best_score) {
            best_score = score;
            best_l = l;
        }
    }
    return best_l;
}

std::pair<int, int> pick_best_mlp_slots(
    ProbeCache& cache,
    const std::vector<std::string>& pos_prompts,
    const std::vector<std::string>& neg_prompts,
    int layer_idx,
    int d_ff) {
    if (d_ff <= 0) {
        return {0, 0};
    }
    int best_pos_slot = 0;
    int best_neg_slot = 0;
    double best_pos_score = -1e18;
    double best_neg_score = -1e18;

    for (int slot = 0; slot < d_ff; ++slot) {
        double p_sum = 0.0;
        double n_sum = 0.0;
        double p_up = 0.0;
        double n_up = 0.0;
        for (const auto& pr : pos_prompts) {
            const ForwardTrace& tr = cache.trace_for(pr);
            if (static_cast<std::size_t>(layer_idx) < tr.mlp_poles.size() &&
                static_cast<std::size_t>(slot) < tr.mlp_poles[static_cast<std::size_t>(layer_idx)].size()) {
                const auto& pt = tr.mlp_poles[static_cast<std::size_t>(layer_idx)][static_cast<std::size_t>(slot)];
                if (!pt.act.empty()) {
                    p_sum += pt.act.back();
                    p_up += pt.up_val.back();
                }
            }
        }
        for (const auto& pr : neg_prompts) {
            const ForwardTrace& tr = cache.trace_for(pr);
            if (static_cast<std::size_t>(layer_idx) < tr.mlp_poles.size() &&
                static_cast<std::size_t>(slot) < tr.mlp_poles[static_cast<std::size_t>(layer_idx)].size()) {
                const auto& pt = tr.mlp_poles[static_cast<std::size_t>(layer_idx)][static_cast<std::size_t>(slot)];
                if (!pt.act.empty()) {
                    n_sum += pt.act.back();
                    n_up += pt.up_val.back();
                }
            }
        }
        const double p_mean = pos_prompts.empty() ? 0.0 : p_sum / static_cast<double>(pos_prompts.size());
        const double n_mean = neg_prompts.empty() ? 0.0 : n_sum / static_cast<double>(neg_prompts.size());
        const double sep_pos = (p_mean - n_mean) + 0.05 * std::max(0.0, p_up);
        const double sep_neg = (n_mean - p_mean) + 0.05 * std::max(0.0, -n_up);
        if (sep_pos > best_pos_score) {
            best_pos_score = sep_pos;
            best_pos_slot = slot;
        }
        if (sep_neg > best_neg_score) {
            best_neg_score = sep_neg;
            best_neg_slot = slot;
        }
    }
    return {best_pos_slot, best_neg_slot};
}

}

NeuralAtlas resolve_symbols_for_target_impl(
    const MIRModule& mir,
    const TransformerModel& target_model,
    const NeuralAtlas* target_atlas,
    std::uint64_t seed,
    const Tokenizer* tokenizer,
    const std::string& streaming_model_dir) {
    const NeuralAtlas def = build_default_atlas(target_model, &mir, seed == 0 ? 42 : seed);
    NeuralAtlas atlas = def;

    const bool atlas_matches = (target_atlas != nullptr) &&
        (target_atlas->hidden_size == target_model.config.hidden_size) &&
        (target_atlas->num_layers == target_model.config.num_layers);

    if (atlas_matches) {
        atlas = *target_atlas;
        if (atlas.dormant_poles.empty()) {
            atlas.dormant_poles = def.dormant_poles;
        }
        for (const auto& [k, v] : def.frame_bridges) {
            if (!atlas.frame_bridges.contains(k)) {
                atlas.frame_bridges[k] = v;
            }
        }
        for (const auto& [k, v] : def.preserve_directions) {
            if (!atlas.preserve_directions.contains(k)) {
                atlas.preserve_directions[k] = v;
            }
        }
    } else if (target_atlas != nullptr) {
        atlas.arch_name = target_model.config.arch_name;
    }

    const int n_layers = std::max(1, target_model.config.num_layers);
    const std::size_t d = static_cast<std::size_t>(std::max(0, target_model.config.hidden_size));
    if (d == 0 || target_model.config.num_layers <= 0) {
        return atlas;
    }

    std::vector<Vector> raw_protected;
    std::set<std::string> preserve_names = {"Residual", "CoreFluency", "CoreReasoning", "Fluency"};
    for (const auto& flow : mir.flows) {
        for (const auto& pname : flow.contract.preserve) {
            preserve_names.insert(pname);
        }
    }
    for (const auto& pname : preserve_names) {
        auto pit = atlas.preserve_directions.find(pname);
        if (pit != atlas.preserve_directions.end()) {
            for (const Vector& v : pit->second) {
                if (v.size() == d) {
                    raw_protected.push_back(v);
                }
            }
        }
    }
    const std::vector<Vector> protected_basis = gram_schmidt(raw_protected, 1e-10);

    ProbeCache cache = streaming_model_dir.empty()
        ? ProbeCache(target_model, tokenizer)
        : ProbeCache(streaming_model_dir, target_model.config, tokenizer);

    for (const auto& [sym_name, decl] : mir.declarations) {
        if (atlas_matches && target_atlas->symbols.contains(sym_name)) {
            continue;
        }

        SymbolBinding sb;
        sb.name = decl.name;
        sb.kind = decl.kind;

        auto [l0, l1] = target_model.config.stage_to_layers(
            decl.frame.start, decl.frame.end, decl.frame.inclusive);
        if (l0 >= l1 || l0 >= n_layers) {
            l0 = 0;
            l1 = n_layers;
        }
        l0 = std::clamp(l0, 0, n_layers - 1);
        l1 = std::clamp(std::max(l0 + 1, l1), l0 + 1, n_layers);
        sb.layer_range = {l0, l1};

        std::vector<std::string> pos_prompts;
        std::vector<std::string> neg_prompts;
        std::vector<std::string> gen_prompts;
        collect_anchor_prompts(decl, pos_prompts, neg_prompts, gen_prompts);

        const std::uint64_t sym_seed = fnv1a_64(sym_name, seed + 13ULL);

        if (decl.kind == "frame") {
            for (const auto& [dst_name, via_name] : decl.connections) {
                const std::string k1 = decl.name + "->" + dst_name;
                if (!atlas.frame_bridges.contains(k1)) {
                    atlas.frame_bridges[k1] = def.frame_bridges.contains("Mid->Late")
                        ? def.frame_bridges.at("Mid->Late")
                        : eye(d);
                }
                if (!via_name.empty() && !atlas.frame_bridges.contains(via_name)) {
                    atlas.frame_bridges[via_name] = atlas.frame_bridges[k1];
                }
            }
            sb.direction = project_to_complement(seeded_vec(d, sym_seed), protected_basis, sym_seed);
        } else if (decl.kind == "pole" || decl.kind == "dipole") {
            std::vector<std::string> p_set = pos_prompts;
            std::vector<std::string> n_set = neg_prompts;
            if (p_set.empty()) {
                if (!gen_prompts.empty()) {
                    p_set.push_back(gen_prompts[0]);
                } else {
                    p_set.push_back(sym_name + " positive probe");
                }
            }
            if (n_set.empty()) {
                if (gen_prompts.size() > 1) {
                    n_set.push_back(gen_prompts[1]);
                } else {
                    n_set.push_back(sym_name + " neutral baseline");
                }
            }

            const int best_l = pick_best_contrastive_layer(cache, p_set, n_set, l0, l1, d);
            std::vector<Vector> p_vecs;
            std::vector<Vector> n_vecs;
            for (const auto& p : p_set) {
                p_vecs.push_back(cache.layer_repr(p, best_l));
            }
            for (const auto& n : n_set) {
                n_vecs.push_back(cache.layer_repr(n, best_l));
            }
            const Vector raw_dir = vec_sub(mean_vec(p_vecs, d), mean_vec(n_vecs, d));
            sb.direction = project_to_complement(raw_dir, protected_basis, sym_seed);
            sb.subspace_basis = {sb.direction};

            auto [pos_slot, neg_slot] = pick_best_mlp_slots(
                cache, p_set, n_set, best_l, target_model.config.intermediate_size);
            sb.pole_slots["+pole"] = {best_l, pos_slot, 1};
            if (decl.kind == "dipole") {
                sb.pole_slots["-pole"] = {best_l, neg_slot, -1};
            }
        } else if (decl.kind == "subspace") {
            const std::size_t k = std::min(
                static_cast<std::size_t>(std::max(1, decl.dim)),
                std::max<std::size_t>(1, d - std::min(d - 1, protected_basis.size())));

            std::vector<std::string> probes;
            std::vector<double> signs;
            for (const auto& p : pos_prompts) {
                probes.push_back(p);
                signs.push_back(1.0);
            }
            for (const auto& n : neg_prompts) {
                probes.push_back(n);
                signs.push_back(-1.0);
            }
            for (const auto& g : gen_prompts) {
                probes.push_back(g);
                signs.push_back(1.0);
            }
            if (probes.empty()) {
                probes.push_back(sym_name);
                signs.push_back(1.0);
            }
            const std::size_t target_n = std::max<std::size_t>(k + 2, 4);
            std::size_t base_idx = 0;
            while (probes.size() < target_n) {
                const std::string base = probes[base_idx % probes.size()];
                const double sgn = signs[base_idx % signs.size()];
                probes.push_back(base + " variant_" + std::to_string(probes.size()));
                signs.push_back(sgn);
                ++base_idx;
            }

            int best_l = l0;
            if (!pos_prompts.empty() && !neg_prompts.empty()) {
                best_l = pick_best_contrastive_layer(cache, pos_prompts, neg_prompts, l0, l1, d);
            } else {
                best_l = (l0 + l1 - 1) / 2;
            }

            const std::size_t n_pts = probes.size();
            std::vector<Vector> h_list;
            h_list.reserve(n_pts);
            for (const auto& pr : probes) {
                h_list.push_back(cache.layer_repr(pr, best_l));
            }
            const Vector mu = mean_vec(h_list, d);
            Matrix h_centered(n_pts, d, 0.0);
            for (std::size_t i = 0; i < n_pts; ++i) {
                h_centered.set_row(i, subtract_basis(vec_sub(h_list[i], mu), protected_basis));
            }
            const std::vector<SVDMode> svd_u = svd_top_k(h_centered, k, 40, sym_seed);
            std::vector<Vector> raw_basis;
            for (const auto& m : svd_u) {
                raw_basis.push_back(m.v);
            }
            std::vector<Vector> u_tgt = orthonormalize_complement(
                raw_basis, k, d, protected_basis, sym_seed);

            Matrix tgt_points(n_pts, k, 0.0);
            Matrix src_points(n_pts, k, 0.0);
            for (std::size_t i = 0; i < n_pts; ++i) {
                const Vector h_p = h_centered.row(i);
                for (std::size_t j = 0; j < k; ++j) {
                    tgt_points(i, j) = dot(h_p, u_tgt[j]);
                }
                src_points(i, 0) = signs[i];
                for (std::size_t j = 1; j < k; ++j) {
                    const double phase = 2.0 * kPi * static_cast<double>((i + 1) * j) / static_cast<double>(n_pts);
                    src_points(i, j) = std::cos(phase);
                }
            }

            const Matrix r_align = orthogonal_procrustes(src_points, tgt_points);
            std::vector<Vector> aligned_cands(k, vec_zeros(d));
            for (std::size_t c = 0; c < k; ++c) {
                for (std::size_t j = 0; j < k; ++j) {
                    const double w = r_align(j, c);
                    for (std::size_t idx = 0; idx < d; ++idx) {
                        aligned_cands[c][idx] += w * u_tgt[j][idx];
                    }
                }
            }
            sb.subspace_basis = orthonormalize_complement(
                aligned_cands, k, d, protected_basis, sym_seed + 101ULL);
            sb.direction = sb.subspace_basis[0];
        } else if (decl.kind == "manifold") {
            std::vector<std::string> cyc_prompts = gen_prompts;
            if (cyc_prompts.size() < 3) {
                const std::string base = cyc_prompts.empty() ? sym_name : cyc_prompts[0];
                while (cyc_prompts.size() < 7) {
                    cyc_prompts.push_back(base + "_" + std::to_string(cyc_prompts.size()));
                }
            }
            const std::size_t n_pts = cyc_prompts.size();
            const std::size_t plane_dim = (decl.topology == "S1" || decl.curvature > 0.0)
                ? std::min<std::size_t>(3, d)
                : std::min<std::size_t>(static_cast<std::size_t>(std::max(1, decl.dim)), d);

            double best_err = 1e18;
            std::vector<Vector> best_plane;
            for (int l = l0; l < l1; ++l) {
                std::vector<Vector> h_list;
                h_list.reserve(n_pts);
                for (const auto& pr : cyc_prompts) {
                    h_list.push_back(cache.layer_repr(pr, l));
                }
                const Vector mu = mean_vec(h_list, d);
                const Vector e0 = project_to_complement(mu, protected_basis, sym_seed + static_cast<std::uint64_t>(l));

                std::vector<Vector> prot_plus_e0 = protected_basis;
                prot_plus_e0.push_back(e0);

                Matrix h_cent(n_pts, d, 0.0);
                for (std::size_t i = 0; i < n_pts; ++i) {
                    h_cent.set_row(i, subtract_basis(vec_sub(h_list[i], mu), prot_plus_e0));
                }
                const std::vector<SVDMode> svd_c = svd_top_k(h_cent, 2, 40, sym_seed);
                std::vector<Vector> v_cands;
                for (const auto& m : svd_c) {
                    v_cands.push_back(m.v);
                }
                const std::vector<Vector> v2 = orthonormalize_complement(
                    v_cands, 2, d, prot_plus_e0, sym_seed + 303ULL);

                if (plane_dim >= 3) {
                    Matrix tgt_2d(n_pts, 2, 0.0);
                    Matrix src_2d(n_pts, 2, 0.0);
                    for (std::size_t i = 0; i < n_pts; ++i) {
                        const Vector row_v = h_cent.row(i);
                        tgt_2d(i, 0) = dot(row_v, v2[0]);
                        tgt_2d(i, 1) = dot(row_v, v2[1]);
                        const double th = (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(n_pts);
                        src_2d(i, 0) = std::cos(th);
                        src_2d(i, 1) = std::sin(th);
                    }
                    const Matrix r2 = orthogonal_procrustes(src_2d, tgt_2d);
                    const double tgt_n = std::max(1e-9, frob_norm(tgt_2d));
                    const double rel_err = frob_norm(mat_sub(mat_mul(src_2d, r2), tgt_2d)) / tgt_n;

                    if (rel_err < best_err) {
                        best_err = rel_err;
                        Vector e1_raw = vec_add(vec_scale(v2[0], r2(0, 0)), vec_scale(v2[1], r2(1, 0)));
                        Vector e2_raw = vec_add(vec_scale(v2[0], r2(0, 1)), vec_scale(v2[1], r2(1, 1)));
                        std::vector<Vector> plane_2 = orthonormalize_complement(
                            {e1_raw, e2_raw}, 2, d, prot_plus_e0, sym_seed + 509ULL);
                        best_plane = {e0, plane_2[0], plane_2[1]};
                    }
                } else {
                    best_plane = orthonormalize_complement(v_cands, plane_dim, d, protected_basis, sym_seed);
                    break;
                }
            }
            if (best_plane.size() < plane_dim) {
                best_plane = orthonormalize_complement({}, plane_dim, d, protected_basis, sym_seed);
            }
            sb.manifold_plane = best_plane;
            sb.subspace_basis = best_plane;
            sb.direction = (best_plane.size() > 1) ? best_plane[1] : best_plane[0];
        } else if (decl.kind == "rodata") {
            std::string tmpl = gen_prompts.empty() ? std::string("{subject} is {object}") : gen_prompts[0];
            std::vector<Vector> key_cands;
            for (const auto& re : decl.rodata_entries) {
                const std::string prompt = replace_template_slots(tmpl, re.key, re.value);
                key_cands.push_back(cache.layer_repr(prompt, (l0 + l1 - 1) / 2));
            }
            if (key_cands.empty()) {
                key_cands.push_back(cache.layer_repr(tmpl, (l0 + l1 - 1) / 2));
            }
            const std::size_t key_cnt = std::min(key_cands.size(), std::max<std::size_t>(1, d / 4));
            sb.preserve_keys = orthonormalize_complement(
                key_cands, key_cnt, d, protected_basis, sym_seed);
            sb.direction = sb.preserve_keys[0];
        } else if (decl.kind == "field") {
            const std::size_t r_eff = std::min(
                static_cast<std::size_t>(std::max(1, decl.rank)),
                std::max<std::size_t>(1, d));

            std::vector<std::string> prompts;
            prompts.insert(prompts.end(), gen_prompts.begin(), gen_prompts.end());
            prompts.insert(prompts.end(), pos_prompts.begin(), pos_prompts.end());
            prompts.insert(prompts.end(), neg_prompts.begin(), neg_prompts.end());
            if (prompts.empty()) {
                prompts.push_back(sym_name + " spectral anchor");
            }

            Matrix accum = zeros(d, d);
            for (std::size_t p_idx = 0; p_idx < prompts.size(); ++p_idx) {
                const ForwardTrace& tr = cache.trace_for(prompts[p_idx]);
                for (int l = l0; l < l1; ++l) {
                    const std::size_t in_idx = static_cast<std::size_t>(l);
                    const std::size_t out_idx = static_cast<std::size_t>(l + 1);
                    if (out_idx < tr.residual.size() && !tr.residual[in_idx].empty()) {
                        const Vector& x_in = tr.residual[in_idx].back();
                        const Vector& x_out = tr.residual[out_idx].back();
                        const double w = (!neg_prompts.empty() && p_idx >= gen_prompts.size() + pos_prompts.size())
                            ? -1.0
                            : 1.0;
                        accum = mat_add(accum, mat_scale(outer(x_out, x_in), w));
                    }
                }
            }
            const std::string o_key = "model.layers." + std::to_string(l0) + ".self_attn.o_proj.weight";
            if (target_model.matrices.contains(o_key) &&
                target_model.matrices.at(o_key).rows == d &&
                target_model.matrices.at(o_key).cols == d) {
                const double a_norm = std::max(1e-9, frob_norm(accum));
                accum = mat_add(mat_scale(accum, 1.0 / a_norm), mat_scale(target_model.matrices.at(o_key), 0.25));
            }
            if (frob_norm(accum) < 1e-12) {
                accum = eye(d);
            }
            std::vector<SVDMode> modes = svd_top_k(accum, r_eff, 40, sym_seed);
            for (std::size_t i = 0; i < modes.size(); ++i) {
                if (modes[i].sigma < 1e-6) {
                    modes[i].sigma = 1.0 / static_cast<double>(i + 1);
                }
            }
            sb.field_modes = std::move(modes);
            if (!sb.field_modes.empty()) {
                sb.direction = sb.field_modes[0].u;
            } else {
                sb.direction = project_to_complement(seeded_vec(d, sym_seed), protected_basis, sym_seed);
            }
        } else {
            sb.direction = project_to_complement(seeded_vec(d, sym_seed), protected_basis, sym_seed);
        }

        atlas.symbols[sym_name] = std::move(sb);
    }

    return atlas;
}

NeuralAtlas resolve_symbols_for_target(
    const MIRModule& mir,
    const TransformerModel& target_model,
    const NeuralAtlas* target_atlas,
    std::uint64_t seed,
    const Tokenizer* tokenizer) {
    return resolve_symbols_for_target_impl(
        mir, target_model, target_atlas, seed, tokenizer, "");
}

NeuralAtlas resolve_symbols_for_target_streaming(
    const MIRModule& mir,
    const std::string& model_dir,
    const NeuralAtlas* target_atlas,
    std::uint64_t seed,
    const Tokenizer* tokenizer) {
    TransformerModel config_only;
    config_only.config = load_checkpoint_config(model_dir);
    return resolve_symbols_for_target_impl(
        mir, config_only, target_atlas, seed, tokenizer, model_dir);
}

CompileOutput cross_compile(
    const std::string& strata_source,
    const TransformerModel& target_model,
    const NeuralAtlas* target_atlas,
    const std::string& target_mode,
    const Tokenizer* tokenizer) {
    const MIRModule mir = compile_source_to_mir(strata_source);
    const NeuralAtlas resolved = resolve_symbols_for_target(mir, target_model, target_atlas, 0, tokenizer);
    return compile_mir_to_model(mir, target_model, &resolved, target_mode);
}

}
