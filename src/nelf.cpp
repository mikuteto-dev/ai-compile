#include "aic/nelf.hpp"

#include "aic/errors.hpp"
#include "aic/json_min.hpp"
#include "aic/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>

namespace aic {

namespace {

double matrix_energy(const std::map<std::string, Matrix>& mats) {
    long double total = 0.0;
    for (const auto& [name, m] : mats) {
        (void)name;
        for (double x : m.data) {
            const long double lx = static_cast<long double>(x);
            total += lx * lx;
        }
    }
    return static_cast<double>(total);
}

void save_matrix_section(
    const std::filesystem::path& path,
    const std::map<std::string, Matrix>& mats,
    const std::string& section_name,
    const std::string& arch_name) {
    std::map<std::string, TensorSpec> specs;
    for (const auto& [name, mat] : mats) {
        specs[name] = TensorSpec::from_matrix(mat, "F32");
    }
    save_safetensors(
        path.string(),
        specs,
        {{"section", section_name}, {"arch_name", arch_name}});
}

std::map<std::string, Matrix> load_matrix_section(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw IOError("Missing NeuralELF section file: " + path.string());
    }
    auto [specs, metadata] = load_safetensors(path.string());
    (void)metadata;
    std::map<std::string, Matrix> out;
    for (const auto& [name, spec] : specs) {
        if (spec.shape.size() != 2) {
            throw IOError("Expected 2D matrix tensor in section for: " + name);
        }
        out[name] = spec.to_matrix();
    }
    return out;
}

void write_text_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream ofs(path.string(), std::ios::trunc);
    if (!ofs) {
        throw IOError("Failed to open file for writing: " + path.string());
    }
    ofs << content << '\n';
    if (!ofs.good()) {
        throw IOError("Failed writing file: " + path.string());
    }
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream ifs(path.string());
    if (!ifs) {
        throw IOError("Failed to open file for reading: " + path.string());
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

void add_section_if_present(
    const std::map<std::string, Matrix>& sec,
    const std::string& key,
    Matrix& acc,
    std::size_t rows,
    std::size_t cols) {
    const auto it = sec.find(key);
    if (it == sec.end()) {
        return;
    }
    const Matrix& m = it->second;
    if (m.rows != rows || m.cols != cols) {
        throw IOError("Matrix shape mismatch across NeuralELF sections for: " + key);
    }
    for (std::size_t i = 0; i < acc.data.size(); ++i) {
        acc.data[i] += m.data[i];
    }
}

}

TransformerModel NeuralELF::reconstruct_model() const {
    TransformerModel out;
    out.config = config;
    out.matrices = passthrough_matrices;
    out.vectors = passthrough_vectors;

    std::map<std::string, std::pair<std::size_t, std::size_t>> dims;
    for (const auto* sec : {&text_weights, &rodata_weights, &field_weights, &residual_weights}) {
        for (const auto& [key, mat] : *sec) {
            if (!dims.contains(key)) {
                dims[key] = {mat.rows, mat.cols};
            }
        }
    }

    for (const auto& [key, shape] : dims) {
        Matrix acc = zeros(shape.first, shape.second);
        add_section_if_present(text_weights, key, acc, shape.first, shape.second);
        add_section_if_present(rodata_weights, key, acc, shape.first, shape.second);
        add_section_if_present(field_weights, key, acc, shape.first, shape.second);
        add_section_if_present(residual_weights, key, acc, shape.first, shape.second);
        out.matrices[key] = std::move(acc);
    }

    return out;
}

std::map<std::string, double> NeuralELF::section_energy_ratios() const {
    const double e_text = matrix_energy(text_weights);
    const double e_rodata = matrix_energy(rodata_weights);
    const double e_field = matrix_energy(field_weights);
    const double e_residual = matrix_energy(residual_weights);
    const double total = e_text + e_rodata + e_field + e_residual;

    if (total <= 0.0) {
        return {
            {".text", 0.0},
            {".rodata", 0.0},
            {".field", 0.0},
            {".residual", 0.0}
        };
    }

    return {
        {".text", e_text / total},
        {".rodata", e_rodata / total},
        {".field", e_field / total},
        {".residual", e_residual / total}
    };
}

void NeuralELF::save(const std::string& dir) const {
    const std::filesystem::path out_dir(dir);
    std::filesystem::create_directories(out_dir);

    save_matrix_section(out_dir / ".text.safetensors", text_weights, ".text", config.arch_name);
    save_matrix_section(out_dir / ".rodata.safetensors", rodata_weights, ".rodata", config.arch_name);
    save_matrix_section(out_dir / ".field.safetensors", field_weights, ".field", config.arch_name);
    save_matrix_section(out_dir / ".residual.safetensors", residual_weights, ".residual", config.arch_name);

    std::map<std::string, TensorSpec> pass_specs;
    for (const auto& [name, mat] : passthrough_matrices) {
        pass_specs[name] = TensorSpec::from_matrix(mat, "F32");
    }
    for (const auto& [name, vec] : passthrough_vectors) {
        pass_specs[name] = TensorSpec::from_vector(vec, "F32");
    }
    save_safetensors(
        (out_dir / "passthrough.safetensors").string(),
        pass_specs,
        {{"section", "passthrough"}, {"arch_name", config.arch_name}});

    atlas.save((out_dir / "atlas.json").string());
    write_text_file(out_dir / "config.json", dump_json(config.to_json(), 2));

    const auto ratios = section_energy_ratios();
    JsonValue ratio_obj = JsonValue::object();
    for (const auto& [sec_name, val] : ratios) {
        ratio_obj[sec_name] = val;
    }

    JsonValue sec_arr = JsonValue::array();
    sec_arr.as_array().emplace_back(".text");
    sec_arr.as_array().emplace_back(".rodata");
    sec_arr.as_array().emplace_back(".field");
    sec_arr.as_array().emplace_back(".residual");
    sec_arr.as_array().emplace_back("passthrough");

    JsonValue manifest = JsonValue::object();
    manifest["format"] = "neural-elf-v1";
    manifest["arch_name"] = config.arch_name;
    manifest["sections"] = std::move(sec_arr);
    manifest["energy_ratios"] = std::move(ratio_obj);
    manifest["partitioned_tensors"] = static_cast<std::int64_t>(text_weights.size());
    manifest["passthrough_matrices"] = static_cast<std::int64_t>(passthrough_matrices.size());
    manifest["passthrough_vectors"] = static_cast<std::int64_t>(passthrough_vectors.size());
    write_text_file(out_dir / "manifest.json", dump_json(manifest, 2));
}

NeuralELF NeuralELF::load(const std::string& dir) {
    const std::filesystem::path in_dir(dir);
    if (!std::filesystem::exists(in_dir)) {
        throw IOError("NeuralELF directory does not exist: " + dir);
    }

    NeuralELF nelf;
    nelf.config = ModelConfig::from_json(parse_json(read_text_file(in_dir / "config.json")));
    nelf.atlas = NeuralAtlas::load((in_dir / "atlas.json").string());
    nelf.text_weights = load_matrix_section(in_dir / ".text.safetensors");
    nelf.rodata_weights = load_matrix_section(in_dir / ".rodata.safetensors");
    nelf.field_weights = load_matrix_section(in_dir / ".field.safetensors");
    nelf.residual_weights = load_matrix_section(in_dir / ".residual.safetensors");

    const std::filesystem::path pass_path = in_dir / "passthrough.safetensors";
    if (!std::filesystem::exists(pass_path)) {
        throw IOError("Missing passthrough.safetensors in NeuralELF directory: " + dir);
    }
    auto [pass_specs, pass_meta] = load_safetensors(pass_path.string());
    (void)pass_meta;
    for (const auto& [name, spec] : pass_specs) {
        if (spec.shape.size() == 2) {
            nelf.passthrough_matrices[name] = spec.to_matrix();
        } else if (spec.shape.size() == 1) {
            nelf.passthrough_vectors[name] = spec.to_vector();
        } else {
            throw IOError("Unsupported tensor rank in passthrough.safetensors for: " + name);
        }
    }

    return nelf;
}

namespace {

double section_frob_diff(
    const std::map<std::string, Matrix>& lhs,
    const std::map<std::string, Matrix>& rhs) {
    long double sum_sq = 0.0;
    std::set<std::string> keys;
    for (const auto& [k, m] : lhs) {
        (void)m;
        keys.insert(k);
    }
    for (const auto& [k, m] : rhs) {
        (void)m;
        keys.insert(k);
    }
    for (const std::string& k : keys) {
        const auto ita = lhs.find(k);
        const auto itb = rhs.find(k);
        if (ita != lhs.end() && itb != rhs.end()) {
            const Matrix& ma = ita->second;
            const Matrix& mb = itb->second;
            const std::size_t max_r = std::max(ma.rows, mb.rows);
            const std::size_t max_c = std::max(ma.cols, mb.cols);
            for (std::size_t r = 0; r < max_r; ++r) {
                for (std::size_t c = 0; c < max_c; ++c) {
                    const double va = (r < ma.rows && c < ma.cols) ? ma(r, c) : 0.0;
                    const double vb = (r < mb.rows && c < mb.cols) ? mb(r, c) : 0.0;
                    const long double d = static_cast<long double>(va - vb);
                    sum_sq += d * d;
                }
            }
        } else if (ita != lhs.end()) {
            const double fn = frob_norm(ita->second);
            sum_sq += static_cast<long double>(fn) * static_cast<long double>(fn);
        } else if (itb != rhs.end()) {
            const double fn = frob_norm(itb->second);
            sum_sq += static_cast<long double>(fn) * static_cast<long double>(fn);
        }
    }
    return std::sqrt(static_cast<double>(sum_sq));
}

bool vecs_differ(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        return true;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > 1e-9) {
            return true;
        }
    }
    return false;
}

bool vec_lists_differ(const std::vector<Vector>& a, const std::vector<Vector>& b) {
    if (a.size() != b.size()) {
        return true;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (vecs_differ(a[i], b[i])) {
            return true;
        }
    }
    return false;
}

bool symbol_bindings_differ(const SymbolBinding& a, const SymbolBinding& b) {
    if (a.kind != b.kind || a.layer_range != b.layer_range || a.pole_slots != b.pole_slots) {
        return true;
    }
    if (vecs_differ(a.direction, b.direction) ||
        vec_lists_differ(a.subspace_basis, b.subspace_basis) ||
        vec_lists_differ(a.preserve_keys, b.preserve_keys) ||
        vec_lists_differ(a.manifold_plane, b.manifold_plane)) {
        return true;
    }
    if (a.field_modes.size() != b.field_modes.size()) {
        return true;
    }
    for (std::size_t i = 0; i < a.field_modes.size(); ++i) {
        if (std::abs(a.field_modes[i].sigma - b.field_modes[i].sigma) > 1e-9 ||
            vecs_differ(a.field_modes[i].u, b.field_modes[i].u) ||
            vecs_differ(a.field_modes[i].v, b.field_modes[i].v)) {
            return true;
        }
    }
    return false;
}

}

NelfDiffReport diff_nelf(const NeuralELF& lhs, const NeuralELF& rhs) {
    NelfDiffReport rep;
    rep.text_frob_diff = section_frob_diff(lhs.text_weights, rhs.text_weights);
    rep.rodata_frob_diff = section_frob_diff(lhs.rodata_weights, rhs.rodata_weights);
    rep.field_frob_diff = section_frob_diff(lhs.field_weights, rhs.field_weights);
    rep.residual_frob_diff = section_frob_diff(lhs.residual_weights, rhs.residual_weights);

    std::set<std::string> sym_names;
    for (const auto& [name, sb] : lhs.atlas.symbols) {
        (void)sb;
        sym_names.insert(name);
    }
    for (const auto& [name, sb] : rhs.atlas.symbols) {
        (void)sb;
        sym_names.insert(name);
    }
    for (const std::string& name : sym_names) {
        const auto ita = lhs.atlas.symbols.find(name);
        const auto itb = rhs.atlas.symbols.find(name);
        if (ita == lhs.atlas.symbols.end() || itb == rhs.atlas.symbols.end()) {
            ++rep.changed_symbols;
        } else if (symbol_bindings_differ(ita->second, itb->second)) {
            ++rep.changed_symbols;
        }
    }

    return rep;
}

std::string format_nelf_diff(const NelfDiffReport& rep) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "NelfDiff:\n"
        << "  text_frob_diff: " << rep.text_frob_diff << "\n"
        << "  rodata_frob_diff: " << rep.rodata_frob_diff << "\n"
        << "  field_frob_diff: " << rep.field_frob_diff << "\n"
        << "  residual_frob_diff: " << rep.residual_frob_diff << "\n"
        << "  changed_symbols: " << rep.changed_symbols << "\n";
    return oss.str();
}

}
