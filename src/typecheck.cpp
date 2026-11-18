#include "aic/typecheck.hpp"
#include "aic/errors.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>

namespace aic {

namespace {

std::string join_path(const std::vector<std::string>& parts) {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            out += "::";
        }
        out += parts[i];
    }
    return out;
}

std::string to_lower_copy(const std::string& s) {
    std::string out = s;
    for (char& ch : out) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return out;
}

bool is_digits(const std::string& s) {
    if (s.empty()) {
        return false;
    }
    for (unsigned char c : s) {
        if (!std::isdigit(c)) {
            return false;
        }
    }
    return true;
}

struct ParsedType {
    std::string base;
    std::vector<std::string> args;
};

std::string trim_ws(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

ParsedType parse_type_str(const std::string& s) {
    ParsedType pt;
    auto lt = s.find('<');
    if (lt == std::string::npos) {
        pt.base = trim_ws(s);
        return pt;
    }
    auto gt = s.rfind('>');
    if (gt == std::string::npos || gt < lt) {
        pt.base = trim_ws(s);
        return pt;
    }
    pt.base = trim_ws(s.substr(0, lt));
    std::string inner = s.substr(lt + 1, gt - lt - 1);
    int depth = 0;
    std::size_t start = 0;
    for (std::size_t i = 0; i < inner.size(); ++i) {
        if (inner[i] == '<') {
            ++depth;
        } else if (inner[i] == '>') {
            --depth;
        } else if (inner[i] == ',' && depth == 0) {
            pt.args.push_back(trim_ws(inner.substr(start, i - start)));
            start = i + 1;
        }
    }
    if (start < inner.size()) {
        pt.args.push_back(trim_ws(inner.substr(start)));
    }
    return pt;
}

std::string format_type_ref(const TypeRef& tr) {
    std::string out = tr.name;
    if (!tr.args.empty()) {
        out += "<";
        for (std::size_t i = 0; i < tr.args.size(); ++i) {
            if (i > 0) {
                out += ", ";
            }
            const auto& a = tr.args[i];
            if (a.kind == TypeArg::Kind::Int) {
                out += std::to_string(a.int_val);
            } else if (a.kind == TypeArg::Kind::Float) {
                std::ostringstream oss;
                oss << a.float_val;
                out += oss.str();
            } else if (a.type_ref) {
                out += format_type_ref(*a.type_ref);
            }
        }
        out += ">";
    }
    return out;
}

std::optional<double> eval_const_number(const ExprPtr& e,
                                        const std::map<std::string, double>* consts = nullptr) {
    if (!e) {
        return std::nullopt;
    }
    if (const auto* f = std::get_if<FloatLitExpr>(&e->node)) {
        return f->value;
    }
    if (const auto* i = std::get_if<IntLitExpr>(&e->node)) {
        return static_cast<double>(i->value);
    }
    if (const auto* p = std::get_if<PathExpr>(&e->node)) {
        if (consts) {
            auto it = consts->find(join_path(p->parts));
            if (it != consts->end()) {
                return it->second;
            }
        }
        return std::nullopt;
    }
    if (const auto* u = std::get_if<UnaryExpr>(&e->node)) {
        auto sub = eval_const_number(u->operand, consts);
        if (!sub) {
            return std::nullopt;
        }
        if (u->op == "-") {
            return -(*sub);
        }
        if (u->op == "+") {
            return *sub;
        }
        return std::nullopt;
    }
    if (const auto* b = std::get_if<BinaryExpr>(&e->node)) {
        auto lv = eval_const_number(b->lhs, consts);
        auto rv = eval_const_number(b->rhs, consts);
        if (!lv || !rv) {
            return std::nullopt;
        }
        if (b->op == "+") return *lv + *rv;
        if (b->op == "-") return *lv - *rv;
        if (b->op == "*") return *lv * *rv;
        if (b->op == "/" && *rv != 0.0) return *lv / *rv;
    }
    return std::nullopt;
}

struct Interval {
    bool valid = false;
    bool is_normalized = false;
    double lo = 0.0;
    double hi = 1.0;
    bool lo_open = false;
    bool hi_open = false;
};

std::optional<std::tuple<double, double, bool>> named_stage_bounds(const std::string& name) {
    if (name == "Early") return std::make_tuple(0.0, 0.25, true);
    if (name == "Mid") return std::make_tuple(0.25, 0.75, true);
    if (name == "Late") return std::make_tuple(0.75, 0.95, true);
    if (name == "Logit") return std::make_tuple(0.95, 1.0, false);
    if (name == "All" || name.empty()) return std::make_tuple(0.0, 1.0, false);
    return std::nullopt;
}

Interval to_interval(const StageSpec& s) {
    Interval iv;
    if (s.start.empty() || s.start == "All") {
        if (s.end.empty()) {
            iv.valid = true;
            iv.is_normalized = true;
            iv.lo = 0.0;
            iv.hi = 1.0;
            iv.lo_open = false;
            iv.hi_open = false;
            return iv;
        }
    }
    if (is_digits(s.start) && (s.end.empty() || is_digits(s.end))) {
        iv.valid = true;
        iv.is_normalized = false;
        iv.lo = std::stod(s.start);
        iv.hi = s.end.empty() ? iv.lo : std::stod(s.end);
        iv.lo_open = false;
        iv.hi_open = s.end.empty() ? false : !s.inclusive;
        return iv;
    }
    auto b1 = named_stage_bounds(s.start);
    if (b1) {
        iv.valid = true;
        iv.is_normalized = true;
        iv.lo = std::get<0>(*b1);
        if (s.end.empty()) {
            iv.hi = std::get<1>(*b1);
            iv.hi_open = std::get<2>(*b1);
        } else {
            auto b2 = named_stage_bounds(s.end);
            if (!b2) {
                iv.valid = false;
                return iv;
            }
            iv.hi = std::get<1>(*b2);
            iv.hi_open = s.inclusive ? false : std::get<2>(*b2);
        }
        return iv;
    }
    return iv;
}

bool intervals_overlap(const Interval& a, const Interval& b) {
    if (!a.valid || !b.valid) {
        return true;
    }
    if (a.is_normalized && a.lo == 0.0 && a.hi == 1.0 && !a.lo_open && !a.hi_open) {
        return true;
    }
    if (b.is_normalized && b.lo == 0.0 && b.hi == 1.0 && !b.lo_open && !b.hi_open) {
        return true;
    }
    if (a.is_normalized != b.is_normalized) {
        return true;
    }
    double lo = std::max(a.lo, b.lo);
    double hi = std::min(a.hi, b.hi);
    if (lo < hi - 1e-12) {
        return true;
    }
    if (std::abs(lo - hi) <= 1e-12) {
        bool a_has = (lo > a.lo + 1e-12 || !a.lo_open) && (lo < a.hi - 1e-12 || !a.hi_open);
        bool b_has = (lo > b.lo + 1e-12 || !b.lo_open) && (lo < b.hi - 1e-12 || !b.hi_open);
        return a_has && b_has;
    }
    return false;
}

std::string stage_key(const StageSpec& s) {
    return s.start + (s.inclusive ? "..=" : "..") + s.end;
}

StageSpec intersect_stages(const StageSpec& a, const StageSpec& b) {
    if (a.start == "All" && a.end.empty()) return b;
    if (b.start == "All" && b.end.empty()) return a;
    Interval ia = to_interval(a);
    Interval ib = to_interval(b);
    if (!ia.valid || !ib.valid || ia.is_normalized != ib.is_normalized) {
        return a;
    }
    double a_len = ia.hi - ia.lo;
    double b_len = ib.hi - ib.lo;
    return (b_len <= a_len) ? b : a;
}

int infer_manifold_dim(const TypeRef& topo) {
    for (const auto& arg : topo.args) {
        if (arg.kind == TypeArg::Kind::Int && arg.int_val > 0) {
            return static_cast<int>(arg.int_val);
        }
    }
    if (topo.name.size() >= 2 && (topo.name[0] == 'S' || topo.name[0] == 'T') &&
        is_digits(topo.name.substr(1))) {
        return std::max(1, std::stoi(topo.name.substr(1)));
    }
    return 1;
}

struct ExprType {
    std::string geom_type = "Unknown";
    StageSpec frame;
    std::string frame_name;
    double curvature = 0.0;
    int dim = 1;
    std::string symbol_name;
    std::set<std::string> decoupled_from;
    std::set<std::string> blend_sources;
    bool is_negated_pole = false;
};

struct FlowEffectSummary {
    double actual_energy = 0.0;
    double declared_xt = 0.0;
    double declared_approx = 0.0;
    double declared_energy = 0.0;
    bool has_xt = false;
    bool has_approx = false;
    bool has_energy = false;
    std::string return_geom_type = "Residual";
    StageSpec return_stage;
    std::string return_frame_name;
};

class TypeChecker {
public:
    explicit TypeChecker(const ProgramAST& ast) : ast_(ast) {}

    TypedProgram run() {
        TypedProgram out;
        out.ast = ast_;
        out.module_name = ast_.module_path.empty() ? "Module" : join_path(ast_.module_path);

        for (const auto& imp : ast_.imports) {
            std::string full = join_path(imp.path);
            out.imports.push_back(full);
            std::string exposed = imp.alias.empty() ? imp.path.back() : imp.alias;
            imported_names_.insert(exposed);
        }

        collect_consts(out);
        collect_frames(out);
        collect_decls(out);
        check_flows(out);
        check_tests(out);

        return out;
    }

private:
    const ProgramAST& ast_;
    std::set<std::string> imported_names_;
    std::map<std::string, StageSpec> frame_stages_;
    std::set<std::pair<std::string, std::string>> connected_frame_names_;
    std::set<std::pair<std::string, std::string>> connected_stage_keys_;

    struct FlowEntry {
        const TopItem* item = nullptr;
        const FlowDecl* decl = nullptr;
        ResolvedFlow base;
        bool has_xt = false;
        bool has_approx = false;
        bool has_energy = false;
        std::set<std::string> preserved_declared_fields;
        std::set<std::string> preserved_declared_symbols;
        StageSpec return_stage;
        std::string return_frame_name;
        std::string return_geom_type = "Residual";
    };

    std::map<std::string, FlowEntry> flow_registry_;
    std::set<std::string> called_flows_;

    void validate_const_expr_refs(const ExprPtr& e, const TypedProgram& out) const {
        if (!e) return;
        if (const auto* p = std::get_if<PathExpr>(&e->node)) {
            std::string name = join_path(p->parts);
            if (!out.const_numbers.count(name) && !out.const_strings.count(name) &&
                !out.const_bools.count(name) && !out.const_lists.count(name)) {
                throw UndefinedSymbolError("Undefined constant '" + name + "' in const expression",
                                           e->loc.line, e->loc.col);
            }
            return;
        }
        if (const auto* u = std::get_if<UnaryExpr>(&e->node)) {
            validate_const_expr_refs(u->operand, out);
        } else if (const auto* b = std::get_if<BinaryExpr>(&e->node)) {
            validate_const_expr_refs(b->lhs, out);
            validate_const_expr_refs(b->rhs, out);
        } else if (const auto* le = std::get_if<ListExpr>(&e->node)) {
            for (const auto& elem : le->elements) {
                validate_const_expr_refs(elem, out);
            }
        }
    }

    void collect_consts(TypedProgram& out) const {
        for (const auto& item : ast_.items) {
            const auto* cd = std::get_if<ConstDecl>(&item.node);
            if (!cd) continue;
            validate_const_expr_refs(cd->value, out);
            if (auto num = eval_const_number(cd->value, &out.const_numbers)) {
                out.const_numbers[cd->name] = *num;
            } else if (const auto* sl = std::get_if<StringLitExpr>(&cd->value->node)) {
                out.const_strings[cd->name] = sl->value;
            } else if (const auto* bl = std::get_if<BoolLitExpr>(&cd->value->node)) {
                out.const_bools[cd->name] = bl->value;
            } else if (const auto* le = std::get_if<ListExpr>(&cd->value->node)) {
                std::vector<std::string> vals;
                for (const auto& elem : le->elements) {
                    if (const auto* es = std::get_if<StringLitExpr>(&elem->node)) {
                        vals.push_back(es->value);
                    }
                }
                out.const_lists[cd->name] = std::move(vals);
            } else if (const auto* pe = std::get_if<PathExpr>(&cd->value->node)) {
                std::string ref = join_path(pe->parts);
                if (out.const_strings.count(ref)) {
                    out.const_strings[cd->name] = out.const_strings.at(ref);
                } else if (out.const_bools.count(ref)) {
                    out.const_bools[cd->name] = out.const_bools.at(ref);
                } else if (out.const_lists.count(ref)) {
                    out.const_lists[cd->name] = out.const_lists.at(ref);
                }
            }
        }
    }

    StageSpec stage_from_annotations(const std::vector<Annotation>& anns) const {
        for (const auto& ann : anns) {
            if (ann.name == "stage" && !ann.args.empty() && ann.args[0].value) {
                const auto& e = ann.args[0].value;
                if (const auto* bin = std::get_if<BinaryExpr>(&e->node)) {
                    if (bin->op == ".." || bin->op == "..=") {
                        StageSpec s;
                        if (const auto* li = std::get_if<IntLitExpr>(&bin->lhs->node)) {
                            s.start = std::to_string(li->value);
                        } else if (const auto* lp = std::get_if<PathExpr>(&bin->lhs->node)) {
                            s.start = join_path(lp->parts);
                        }
                        if (const auto* ri = std::get_if<IntLitExpr>(&bin->rhs->node)) {
                            s.end = std::to_string(ri->value);
                        } else if (const auto* rp = std::get_if<PathExpr>(&bin->rhs->node)) {
                            s.end = join_path(rp->parts);
                        }
                        s.inclusive = (bin->op == "..=");
                        return s;
                    }
                } else if (const auto* p = std::get_if<PathExpr>(&e->node)) {
                    StageSpec s;
                    s.start = join_path(p->parts);
                    return s;
                }
            }
        }
        return StageSpec{};
    }

    std::string target_mode_from_annotations(const std::vector<Annotation>& anns) const {
        for (const auto& ann : anns) {
            if (ann.name == "target" && !ann.args.empty() && ann.args[0].value) {
                if (const auto* s = std::get_if<StringLitExpr>(&ann.args[0].value->node)) {
                    return s->value;
                }
            }
        }
        return "";
    }

    bool has_annotation(const std::vector<Annotation>& anns, const std::string& name) const {
        for (const auto& ann : anns) {
            if (ann.name == name) {
                return true;
            }
        }
        return false;
    }

    std::pair<StageSpec, std::string> resolve_frame_ref(const FrameRef& ref) const {
        if (ref.is_explicit_frame) {
            return {ref.stage, ""};
        }
        std::string name = join_path(ref.path);
        auto it = frame_stages_.find(name);
        if (it == frame_stages_.end()) {
            throw UndefinedSymbolError("Undefined frame '" + name + "'", ref.loc.line, ref.loc.col);
        }
        return {it->second, name};
    }

    bool frames_compatible(const StageSpec& sa, const std::string& fa,
                           const StageSpec& sb, const std::string& fb) const {
        if (stages_overlap(sa, sb)) {
            return true;
        }
        if (!fa.empty() && !fb.empty()) {
            if (connected_frame_names_.count({fa, fb}) || connected_frame_names_.count({fb, fa})) {
                return true;
            }
        }
        std::string ka = stage_key(sa);
        std::string kb = stage_key(sb);
        if (connected_stage_keys_.count({ka, kb}) || connected_stage_keys_.count({kb, ka})) {
            return true;
        }
        return false;
    }

    void collect_frames(TypedProgram& out) {
        for (const auto& item : ast_.items) {
            if (const auto* fd = std::get_if<FrameDecl>(&item.node)) {
                ResolvedSymbol sym;
                sym.name = fd->name;
                sym.kind = "frame";
                sym.frame = fd->stage.value_or(StageSpec{"All", "", false, item.loc});
                sym.frame_name = fd->name;
                sym.anchors = fd->anchors;
                for (const auto& c : fd->connections) {
                    sym.connections.emplace_back(c.target, c.via);
                    connected_frame_names_.insert({fd->name, c.target});
                    connected_frame_names_.insert({c.target, fd->name});
                }
                frame_stages_[fd->name] = sym.frame;
                out.symbols[fd->name] = std::move(sym);
            }
        }
        for (const auto& [fname, sym] : out.symbols) {
            if (sym.kind == "frame") {
                for (const auto& [tgt, _] : sym.connections) {
                    auto it = frame_stages_.find(tgt);
                    if (it != frame_stages_.end()) {
                        std::string k1 = stage_key(sym.frame);
                        std::string k2 = stage_key(it->second);
                        connected_stage_keys_.insert({k1, k2});
                        connected_stage_keys_.insert({k2, k1});
                    }
                }
            }
        }
    }

    void collect_decls(TypedProgram& out) {
        for (const auto& item : ast_.items) {
            if (const auto* pd = std::get_if<PoleDecl>(&item.node)) {
                auto [st, fn] = resolve_frame_ref(pd->frame);
                ResolvedSymbol sym;
                sym.name = pd->name;
                sym.kind = "pole";
                sym.frame = st;
                sym.frame_name = fn;
                sym.dim = 1;
                sym.curvature = 0.0;
                sym.anchors = pd->meta.anchors;
                if (pd->meta.curvature) {
                    if (auto c = eval_const_number(pd->meta.curvature, &out.const_numbers)) {
                        sym.curvature = *c;
                    }
                }
                out.symbols[pd->name] = std::move(sym);
            } else if (const auto* dd = std::get_if<DipoleDecl>(&item.node)) {
                auto [st, fn] = resolve_frame_ref(dd->frame);
                ResolvedSymbol sym;
                sym.name = dd->name;
                sym.kind = "dipole";
                sym.frame = st;
                sym.frame_name = fn;
                sym.dim = 1;
                sym.curvature = 0.0;
                sym.anchors = dd->meta.anchors;
                for (const auto& side : dd->poles) {
                    sym.poles[side.polarity] = side.name;
                    for (const auto& a : side.anchors) {
                        sym.anchors.push_back(a);
                    }
                }
                out.symbols[dd->name] = std::move(sym);
            } else if (const auto* sd = std::get_if<SubspaceDecl>(&item.node)) {
                auto [st, fn] = resolve_frame_ref(sd->frame);
                ResolvedSymbol sym;
                sym.name = sd->name;
                sym.kind = "subspace";
                sym.frame = st;
                sym.frame_name = fn;
                sym.dim = sd->dim;
                sym.rank = sd->dim;
                sym.topology = "Euclidean";
                sym.curvature = 0.0;
                sym.anchors = sd->meta.anchors;
                out.symbols[sd->name] = std::move(sym);
            } else if (const auto* md = std::get_if<ManifoldDecl>(&item.node)) {
                auto [st, fn] = resolve_frame_ref(md->frame);
                ResolvedSymbol sym;
                sym.name = md->name;
                sym.kind = "manifold";
                sym.frame = st;
                sym.frame_name = fn;
                sym.topology = format_type_ref(md->topo);
                sym.dim = infer_manifold_dim(md->topo);
                sym.rank = sym.dim;
                if (md->meta.curvature) {
                    if (auto c = eval_const_number(md->meta.curvature, &out.const_numbers)) {
                        sym.curvature = *c;
                    }
                } else {
                    sym.curvature = (md->topo.name == "Euclidean" || md->topo.name == "Flat") ? 0.0 : 1.0;
                }
                sym.anchors = md->meta.anchors;
                out.symbols[md->name] = std::move(sym);
            } else if (const auto* fd = std::get_if<FieldDecl>(&item.node)) {
                auto [st, fn] = resolve_frame_ref(fd->frame);
                ResolvedSymbol sym;
                sym.name = fd->name;
                sym.kind = "field";
                sym.frame = st;
                sym.frame_name = fn;
                sym.topology = fd->topo ? format_type_ref(*fd->topo) : "All";
                sym.rank = fd->meta.rank.value_or(1);
                sym.metric = fd->meta.metric.empty() ? "fisher" : to_lower_copy(fd->meta.metric);
                sym.curvature = 0.0;
                if (fd->meta.curvature) {
                    if (auto c = eval_const_number(fd->meta.curvature, &out.const_numbers)) {
                        sym.curvature = *c;
                    }
                }
                sym.anchors = fd->meta.anchors;
                out.symbols[fd->name] = std::move(sym);
            } else if (const auto* rd = std::get_if<RodataDecl>(&item.node)) {
                auto [st, fn] = resolve_frame_ref(rd->frame);
                ResolvedSymbol sym;
                sym.name = rd->name;
                sym.kind = "rodata";
                sym.frame = st;
                sym.frame_name = fn;
                sym.topology = format_type_ref(rd->key_type) + "->" + format_type_ref(rd->val_type);
                sym.rodata_entries = rd->entries;
                sym.anchors = rd->meta.anchors;
                out.symbols[rd->name] = std::move(sym);
            }
        }
    }

    ExprType symbol_to_expr_type(const ResolvedSymbol& sym) const {
        ExprType et;
        et.frame = sym.frame;
        et.frame_name = sym.frame_name;
        et.curvature = sym.curvature;
        et.dim = sym.dim;
        et.symbol_name = sym.name;
        if (sym.kind == "pole") {
            et.geom_type = "Pole";
        } else if (sym.kind == "dipole") {
            et.geom_type = "Dipole";
        } else if (sym.kind == "subspace") {
            et.geom_type = "Subspace<" + std::to_string(sym.dim) + ">";
        } else if (sym.kind == "manifold") {
            et.geom_type = "Manifold<" + sym.topology + ", " + std::to_string(sym.dim) + ">";
        } else if (sym.kind == "field") {
            et.geom_type = "Field";
        } else if (sym.kind == "rodata") {
            et.geom_type = "Rodata";
        } else {
            et.geom_type = "Residual";
        }
        return et;
    }

    struct FlowCheckCtx {
        const TypedProgram& prog;
        std::map<std::string, ExprType> env;
        std::map<std::string, double> scalar_consts;
        std::map<std::string, double> scalar_gate_thresholds;
        std::set<std::string> preserved_declared_fields;
        std::set<std::string> preserved_declared_symbols;
        double crosstalk_bound = 1e-2;
        double approx_bound = 1e-1;
        double energy_bound = 1.0;
        bool has_xt_contract = false;
        bool has_approx_contract = false;
        bool has_energy_contract = false;
        double sub_xt_max = 0.0;
        double sub_approx_sum = 0.0;
        double sub_energy_extra = 0.0;
        SourceLoc last_action_loc{};
        std::vector<std::string>* call_stack = nullptr;
    };

    ExprType check_flow_call(const CallExpr& ce, const SourceLoc& call_loc, FlowCheckCtx& ctx) {
        const auto* pe = std::get_if<PathExpr>(&ce.callee->node);
        if (!pe) {
            return ExprType{};
        }
        std::string callee_name = join_path(pe->parts);
        auto fit = flow_registry_.find(callee_name);
        if (fit == flow_registry_.end()) {
            return ExprType{};
        }

        if (ctx.call_stack) {
            auto cit = std::find(ctx.call_stack->begin(), ctx.call_stack->end(), callee_name);
            if (cit != ctx.call_stack->end()) {
                std::string chain;
                for (auto it = cit; it != ctx.call_stack->end(); ++it) {
                    chain += *it + " -> ";
                }
                chain += callee_name;
                throw ParseError("Recursive flow call cycle detected: " + chain,
                                 call_loc.line, call_loc.col);
            }
        }

        called_flows_.insert(callee_name);
        const FlowEntry& target = fit->second;
        const FlowDecl& fdecl = *target.decl;

        std::vector<ExprType> arg_types;
        arg_types.reserve(ce.args.size());
        for (const auto& a : ce.args) {
            arg_types.push_back(check_expr(a.value, ctx));
        }

        if (!fdecl.params.empty()) {
            if (ce.args.empty() && fdecl.params.size() == 1) {
                auto sit = ctx.env.find("state");
                if (sit != ctx.env.end()) {
                    StageSpec p_stage = target.base.stage;
                    std::string p_fn;
                    if (fdecl.params[0].frame) {
                        auto pr = resolve_frame_ref(*fdecl.params[0].frame);
                        p_stage = pr.first;
                        p_fn = pr.second;
                    }
                    if (!frames_compatible(sit->second.frame, sit->second.frame_name, p_stage, p_fn)) {
                        throw FrameMismatchError(
                            "Incompatible frame in implicit state argument to flow '" + callee_name + "'",
                            call_loc.line, call_loc.col);
                    }
                }
            } else {
                for (std::size_t i = 0; i < ce.args.size(); ++i) {
                    const ParamDecl* matched_param = nullptr;
                    if (!ce.args[i].name.empty()) {
                        for (const auto& pd : fdecl.params) {
                            if (pd.name == ce.args[i].name) {
                                matched_param = &pd;
                                break;
                            }
                        }
                    } else if (i < fdecl.params.size()) {
                        matched_param = &fdecl.params[i];
                    }
                    if (matched_param) {
                        StageSpec p_stage = target.base.stage;
                        std::string p_fn;
                        if (matched_param->frame) {
                            auto pr = resolve_frame_ref(*matched_param->frame);
                            p_stage = pr.first;
                            p_fn = pr.second;
                        }
                        if (!frames_compatible(arg_types[i].frame, arg_types[i].frame_name, p_stage, p_fn)) {
                            throw FrameMismatchError(
                                "Incompatible frame for parameter '" + matched_param->name +
                                "' in call to flow '" + callee_name + "'",
                                ce.args[i].loc.line, ce.args[i].loc.col);
                        }
                    }
                }
            }
        }

        double min_approx = ctx.approx_bound;
        if (target.has_approx && target.base.approx_bound < min_approx) {
            min_approx = target.base.approx_bound;
        }
        FlowEffectSummary sub = run_flow_check(
            callee_name,
            *ctx.call_stack,
            ctx.preserved_declared_fields,
            ctx.preserved_declared_symbols,
            min_approx);

        if (sub.has_xt) {
            ctx.sub_xt_max = std::max(ctx.sub_xt_max, sub.declared_xt);
            if (ctx.has_xt_contract && ctx.sub_xt_max > ctx.crosstalk_bound + 1e-12) {
                throw CrosstalkExceededError(
                    "Sub-flow '" + callee_name + "' crosstalk bound " + std::to_string(sub.declared_xt) +
                    " exceeds caller contract <= " + std::to_string(ctx.crosstalk_bound),
                    call_loc.line, call_loc.col);
            }
        }
        if (sub.has_approx) {
            ctx.sub_approx_sum += sub.declared_approx;
            if (ctx.has_approx_contract && ctx.sub_approx_sum > ctx.approx_bound + 1e-12) {
                throw ApproximationUnmetError(
                    "Sub-flow '" + callee_name + "' accumulated approx bound " +
                    std::to_string(ctx.sub_approx_sum) +
                    " exceeds caller contract <= " + std::to_string(ctx.approx_bound),
                    call_loc.line, call_loc.col);
            }
        }
        double sub_e = (sub.actual_energy > 0.0)
                           ? sub.actual_energy
                           : (sub.has_energy ? sub.declared_energy : 0.0);
        ctx.sub_energy_extra += sub_e;
        if (sub.has_energy && ctx.has_energy_contract &&
            sub.declared_energy > ctx.energy_bound + 1e-12) {
            throw EnergyBudgetExceededError(
                "Sub-flow '" + callee_name + "' energy bound " + std::to_string(sub.declared_energy) +
                " exceeds caller contract <= " + std::to_string(ctx.energy_bound),
                call_loc.line, call_loc.col);
        }

        ctx.last_action_loc = call_loc;
        ExprType ret;
        ret.geom_type = sub.return_geom_type;
        ret.frame = sub.return_stage;
        ret.frame_name = sub.return_frame_name;
        ret.symbol_name = callee_name;
        return ret;
    }

    ExprType check_expr(const ExprPtr& e, FlowCheckCtx& ctx) {
        if (!e) {
            return ExprType{};
        }
        if (std::holds_alternative<IntLitExpr>(e->node) ||
            std::holds_alternative<FloatLitExpr>(e->node) ||
            std::holds_alternative<StringLitExpr>(e->node) ||
            std::holds_alternative<BoolLitExpr>(e->node)) {
            return ExprType{"Scalar", StageSpec{}, "", 0.0, 1, "", {}, {}, false};
        }
        if (const auto* le = std::get_if<ListExpr>(&e->node)) {
            for (const auto& elem : le->elements) {
                check_expr(elem, ctx);
            }
            return ExprType{"Scalar", StageSpec{}, "", 0.0, 1, "", {}, {}, false};
        }
        if (const auto* pe = std::get_if<PathExpr>(&e->node)) {
            std::string name = join_path(pe->parts);
            auto lit = ctx.env.find(name);
            if (lit != ctx.env.end()) {
                return lit->second;
            }
            auto sit = ctx.prog.symbols.find(name);
            if (sit != ctx.prog.symbols.end()) {
                return symbol_to_expr_type(sit->second);
            }
            if (ctx.prog.const_numbers.count(name) || ctx.prog.const_strings.count(name) ||
                ctx.prog.const_bools.count(name) || ctx.prog.const_lists.count(name)) {
                return ExprType{"Scalar", StageSpec{}, "", 0.0, 1, name, {}, {}, false};
            }
            if (flow_registry_.count(name)) {
                const auto& fe = flow_registry_.at(name);
                return ExprType{"Flow", fe.base.stage, "", 0.0, 1, name, {}, {}, false};
            }
            if (imported_names_.count(name) || (!pe->parts.empty() && imported_names_.count(pe->parts.back()))) {
                return ExprType{"Residual", StageSpec{}, "", 0.0, 1, name, {}, {}, false};
            }
            throw UndefinedSymbolError("Undefined symbol '" + name + "'", e->loc.line, e->loc.col);
        }
        if (const auto* ue = std::get_if<UnaryExpr>(&e->node)) {
            ExprType sub = check_expr(ue->operand, ctx);
            if (ue->op == "-" && sub.geom_type == "Pole") {
                throw UnipolarViolationError("Cannot negate unipolar Pole '" + sub.symbol_name + "'",
                                             e->loc.line, e->loc.col);
            }
            return sub;
        }
        if (const auto* me = std::get_if<MemberExpr>(&e->node)) {
            ExprType obj = check_expr(me->object, ctx);
            if (me->member == "+pole" || me->member == "-pole") {
                if (obj.geom_type != "Dipole") {
                    throw ParseError("Pole access '" + me->member + "' requires a Dipole",
                                     e->loc.line, e->loc.col);
                }
                ExprType res;
                res.geom_type = "Pole";
                res.frame = obj.frame;
                res.frame_name = obj.frame_name;
                res.curvature = 0.0;
                res.dim = 1;
                res.symbol_name = obj.symbol_name + "." + me->member;
                return res;
            }
            return obj;
        }
        if (const auto* ce = std::get_if<CallExpr>(&e->node)) {
            if (const auto* pe = std::get_if<PathExpr>(&ce->callee->node)) {
                std::string name = join_path(pe->parts);
                if (flow_registry_.count(name)) {
                    return check_flow_call(*ce, e->loc, ctx);
                }
            }
            ExprType callee_t = check_expr(ce->callee, ctx);
            for (const auto& a : ce->args) {
                check_expr(a.value, ctx);
            }
            return callee_t;
        }
        if (const auto* ae = std::get_if<AttendExpr>(&e->node)) {
            std::vector<ExprType> geom_args;
            ExprType val_type{"Residual", StageSpec{}, "", 0.0, 1, "", {}, {}, false};
            for (std::size_t i = 0; i < ae->args.size(); ++i) {
                ExprType at = check_expr(ae->args[i].value, ctx);
                if (at.geom_type != "Scalar") {
                    geom_args.push_back(at);
                }
                if (ae->args[i].name == "value" || (ae->args[i].name.empty() && i == 2)) {
                    val_type = at;
                }
            }
            StageSpec common;
            std::string common_fn;
            if (!geom_args.empty()) {
                common = geom_args[0].frame;
                common_fn = geom_args[0].frame_name;
                for (std::size_t i = 1; i < geom_args.size(); ++i) {
                    if (!frames_compatible(common, common_fn, geom_args[i].frame, geom_args[i].frame_name)) {
                        throw FrameMismatchError("Incompatible frames in @attend arguments",
                                                 e->loc.line, e->loc.col);
                    }
                    common = intersect_stages(common, geom_args[i].frame);
                }
            }
            val_type.frame = common;
            val_type.frame_name = common_fn;
            return val_type;
        }
        if (const auto* te = std::get_if<TransportExpr>(&e->node)) {
            ExprType inner = check_expr(te->expr, ctx);
            auto [src_s, src_fn] = resolve_frame_ref(te->src_frame);
            auto [dst_s, dst_fn] = resolve_frame_ref(te->dst_frame);
            if (!frames_compatible(inner.frame, inner.frame_name, src_s, src_fn)) {
                throw FrameMismatchError("Source expression frame does not match transport source frame",
                                         e->loc.line, e->loc.col);
            }
            inner.frame = dst_s;
            inner.frame_name = dst_fn;
            return inner;
        }
        if (const auto* foe = std::get_if<FieldOpExpr>(&e->node)) {
            ExprType lt = check_expr(foe->lhs, ctx);
            ExprPtr rhs_target_expr = foe->rhs;
            if (const auto* rcall = std::get_if<CallExpr>(&foe->rhs->node)) {
                for (const auto& a : rcall->args) {
                    check_expr(a.value, ctx);
                }
                rhs_target_expr = rcall->callee;
            }
            ExprType rt = check_expr(rhs_target_expr, ctx);
            for (const auto& a : foe->args) {
                check_expr(a.value, ctx);
            }
            if (!frames_compatible(lt.frame, lt.frame_name, rt.frame, rt.frame_name)) {
                throw FrameMismatchError("Incompatible frames in field operator '" + foe->op + "'",
                                         e->loc.line, e->loc.col);
            }
            ExprType res;
            res.geom_type = "Field";
            res.frame = intersect_stages(lt.frame, rt.frame);
            res.frame_name = lt.frame_name.empty() ? rt.frame_name : lt.frame_name;
            if (foe->op == "decouple") {
                res.symbol_name = lt.symbol_name;
                res.decoupled_from = lt.decoupled_from;
                if (!rt.symbol_name.empty()) {
                    res.decoupled_from.insert(rt.symbol_name);
                }
            } else if (foe->op == "blend") {
                for (const auto& d : lt.decoupled_from) {
                    if (rt.decoupled_from.count(d)) {
                        res.decoupled_from.insert(d);
                    }
                }
                res.blend_sources = lt.blend_sources;
                res.blend_sources.insert(rt.blend_sources.begin(), rt.blend_sources.end());
                if (!lt.symbol_name.empty()) res.blend_sources.insert(lt.symbol_name);
                if (!rt.symbol_name.empty()) res.blend_sources.insert(rt.symbol_name);
            } else {
                res = lt;
                res.geom_type = "Field";
            }
            return res;
        }
        if (const auto* fce = std::get_if<FieldCallExpr>(&e->node)) {
            std::vector<ExprType> pos_types;
            for (const auto& a : fce->args) {
                ExprType at = check_expr(a.value, ctx);
                if (a.name.empty() && at.geom_type != "Scalar") {
                    pos_types.push_back(at);
                }
            }
            if (fce->op == "temper") {
                if (pos_types.empty()) {
                    return ExprType{"Field", StageSpec{}, "", 0.0, 1, "", {}, {}, false};
                }
                ExprType res = pos_types[0];
                res.geom_type = "Field";
                return res;
            }
            if (pos_types.size() >= 2) {
                const auto& lt = pos_types[0];
                const auto& rt = pos_types[1];
                if (!frames_compatible(lt.frame, lt.frame_name, rt.frame, rt.frame_name)) {
                    throw FrameMismatchError("Incompatible frames in '" + fce->op + "'",
                                             e->loc.line, e->loc.col);
                }
                ExprType res;
                res.geom_type = "Field";
                res.frame = intersect_stages(lt.frame, rt.frame);
                res.frame_name = lt.frame_name.empty() ? rt.frame_name : lt.frame_name;
                if (fce->op == "decouple") {
                    res.symbol_name = lt.symbol_name;
                    res.decoupled_from = lt.decoupled_from;
                    if (!rt.symbol_name.empty()) {
                        res.decoupled_from.insert(rt.symbol_name);
                    }
                } else {
                    for (const auto& d : lt.decoupled_from) {
                        if (rt.decoupled_from.count(d)) {
                            res.decoupled_from.insert(d);
                        }
                    }
                    res.blend_sources = lt.blend_sources;
                    res.blend_sources.insert(rt.blend_sources.begin(), rt.blend_sources.end());
                    if (!lt.symbol_name.empty()) res.blend_sources.insert(lt.symbol_name);
                    if (!rt.symbol_name.empty()) res.blend_sources.insert(rt.symbol_name);
                }
                return res;
            }
            return ExprType{"Field", StageSpec{}, "", 0.0, 1, "", {}, {}, false};
        }
        if (const auto* be = std::get_if<BinaryExpr>(&e->node)) {
            ExprType lt = check_expr(be->lhs, ctx);
            ExprType rt = check_expr(be->rhs, ctx);
            if (be->op == "~>" || be->op == "=>") {
                if (!frames_compatible(lt.frame, lt.frame_name, rt.frame, rt.frame_name)) {
                    throw FrameMismatchError("Disjoint frames in '" + be->op + "' operation",
                                             e->loc.line, e->loc.col);
                }
                if (be->op == "=>" && (lt.curvature > 0.0 || rt.curvature > 0.0) &&
                    ctx.approx_bound <= 0.0) {
                    throw ApproximationUnmetError(
                        "Curved manifold mapping requires positive approx tolerance (> 0.0)",
                        e->loc.line, e->loc.col);
                }
                ExprType res = rt;
                res.frame = intersect_stages(lt.frame, rt.frame);
                res.frame_name = lt.frame_name.empty() ? rt.frame_name : lt.frame_name;
                return res;
            }
            if (be->op == "&&" || be->op == "||" || be->op == "<=" || be->op == ">=" ||
                be->op == "==" || be->op == "!=" || be->op == "<" || be->op == ">") {
                ExprType res{"Scalar", intersect_stages(lt.frame, rt.frame), "", 0.0, 1, "", {}, {}, false};
                return res;
            }
            return lt;
        }
        return ExprType{};
    }

    std::optional<double> extract_action_scalar(const ExprPtr& rhs, const std::string& arg_key,
                                                const FlowCheckCtx& ctx) const {
        auto eval_scalar = [&](const ExprPtr& value) -> std::optional<double> {
            if (auto exact = eval_const_number(value, &ctx.scalar_consts)) {
                return exact;
            }
            auto dynamic_gate_threshold = [&](const ExprPtr& candidate) -> std::optional<double> {
                const auto* path = candidate ? std::get_if<PathExpr>(&candidate->node) : nullptr;
                if (path == nullptr) {
                    return std::nullopt;
                }
                auto it = ctx.scalar_gate_thresholds.find(join_path(path->parts));
                return it == ctx.scalar_gate_thresholds.end()
                    ? std::nullopt
                    : std::optional<double>(it->second);
            };
            if (dynamic_gate_threshold(value)) {
                return 1.0;
            }
            if (const auto* binary = value ? std::get_if<BinaryExpr>(&value->node) : nullptr;
                binary != nullptr && binary->op == "*") {
                if (auto coefficient = eval_const_number(binary->lhs, &ctx.scalar_consts)) {
                    if (dynamic_gate_threshold(binary->rhs)) {
                        return *coefficient;
                    }
                }
                if (auto coefficient = eval_const_number(binary->rhs, &ctx.scalar_consts)) {
                    if (dynamic_gate_threshold(binary->lhs)) {
                        return *coefficient;
                    }
                }
            }
            return std::nullopt;
        };
        if (!rhs) {
            return 1.0;
        }
        if (const auto* ce = std::get_if<CallExpr>(&rhs->node)) {
            for (std::size_t i = 0; i < ce->args.size(); ++i) {
                if (ce->args[i].name == arg_key || (ce->args[i].name.empty() && i == 0)) {
                    return eval_scalar(ce->args[i].value);
                }
            }
            return 1.0;
        }
        if (const auto* foe = std::get_if<FieldOpExpr>(&rhs->node)) {
            for (const auto& a : foe->args) {
                if (a.name == arg_key) {
                    return eval_scalar(a.value);
                }
            }
            if (const auto* rce = std::get_if<CallExpr>(&foe->rhs->node)) {
                for (const auto& a : rce->args) {
                    if (a.name == arg_key) {
                        return eval_scalar(a.value);
                    }
                }
            }
            return 1.0;
        }
        if (const auto* fce = std::get_if<FieldCallExpr>(&rhs->node)) {
            for (const auto& a : fce->args) {
                if (a.name == arg_key) {
                    return eval_scalar(a.value);
                }
            }
            return 1.0;
        }
        return 1.0;
    }

    double check_stmt(const StmtPtr& s, FlowCheckCtx& ctx) {
        if (!s) {
            return 0.0;
        }
        if (const auto* ls = std::get_if<LetStmt>(&s->node)) {
            double before_sub_e = ctx.sub_energy_extra;
            ExprType val_t = check_expr(ls->value, ctx);
            double delta_sub_e = ctx.sub_energy_extra - before_sub_e;
            ctx.sub_energy_extra = before_sub_e;
            if (ls->type_ann) {
                std::string ann = format_type_ref(*ls->type_ann);
                if (val_t.geom_type != "Unknown" && val_t.geom_type != "Scalar" &&
                    !is_subtype(val_t.geom_type, ann)) {
                    throw ParseError("Type mismatch in let binding '" + ls->name + "': cannot assign " +
                                     val_t.geom_type + " to " + ann, s->loc.line, s->loc.col);
                }
            }
            ctx.env[ls->name] = val_t;
            if (auto cv = eval_const_number(ls->value, &ctx.scalar_consts)) {
                ctx.scalar_consts[ls->name] = *cv;
            } else {
                ctx.scalar_consts.erase(ls->name);
            }
            return delta_sub_e;
        }
        if (const auto* ws = std::get_if<WhenStmt>(&s->node)) {
            double before_sub_e = ctx.sub_energy_extra;
            ExprType ct = check_expr(ws->cond, ctx);
            double cond_sub_e = ctx.sub_energy_extra - before_sub_e;
            ctx.sub_energy_extra = before_sub_e;

            auto saved_env = ctx.env;
            auto saved_consts = ctx.scalar_consts;
            auto saved_gates = ctx.scalar_gate_thresholds;
            if (!ws->bind_name.empty()) {
                ExprType bt;
                bt.geom_type = "Scalar";
                bt.frame = ct.frame;
                bt.frame_name = ct.frame_name;
                ctx.env[ws->bind_name] = bt;
                ctx.scalar_consts.erase(ws->bind_name);
                if (ct.geom_type == "Pole" || ct.geom_type == "Dipole") {
                    double threshold = 0.5;
                    if (const auto* condition = std::get_if<BinaryExpr>(&ws->cond->node);
                        condition != nullptr && condition->op == "~>") {
                        if (const auto* call = std::get_if<CallExpr>(&condition->rhs->node)) {
                            for (std::size_t i = 0; i < call->args.size(); ++i) {
                                if (call->args[i].name == "threshold" ||
                                    (call->args[i].name.empty() && i == 0)) {
                                    auto value = eval_const_number(call->args[i].value, &ctx.scalar_consts);
                                    if (!value || *value <= 0.0) {
                                        throw EnergyUnboundedError(
                                            "Dynamic gate threshold must be a positive constant",
                                            call->args[i].loc.line,
                                            call->args[i].loc.col);
                                    }
                                    threshold = *value;
                                    break;
                                }
                            }
                        }
                    }
                    ctx.scalar_gate_thresholds[ws->bind_name] = threshold;
                }
            }
            double body_energy = 0.0;
            for (const auto& st : ws->body.stmts) {
                body_energy += check_stmt(st, ctx);
            }
            ctx.env = std::move(saved_env);
            ctx.scalar_consts = std::move(saved_consts);
            ctx.scalar_gate_thresholds = std::move(saved_gates);
            double else_energy = 0.0;
            if (ws->has_else) {
                auto saved_else = ctx.env;
                auto saved_else_consts = ctx.scalar_consts;
                for (const auto& st : ws->else_stmts) {
                    else_energy += check_stmt(st, ctx);
                }
                ctx.env = std::move(saved_else);
                ctx.scalar_consts = std::move(saved_else_consts);
            }
            return cond_sub_e + std::max(body_energy, else_energy);
        }
        if (const auto* rs = std::get_if<RodataStmt>(&s->node)) {
            std::string name = join_path(rs->target_path);
            auto sit = ctx.prog.symbols.find(name);
            if (sit == ctx.prog.symbols.end()) {
                throw UndefinedSymbolError("Undefined rodata table '" + name + "'", s->loc.line, s->loc.col);
            }
            if (sit->second.kind != "rodata") {
                throw ParseError("Target '" + name + "' of '" + rs->op + "' is not a rodata table",
                                 s->loc.line, s->loc.col);
            }
            for (const auto& a : rs->args) {
                check_expr(a.value, ctx);
            }
            return 0.0;
        }
        if (const auto* as = std::get_if<ActionStmt>(&s->node)) {
            ExprType lt = check_expr(as->lhs, ctx);
            ExprType rt = check_expr(as->rhs, ctx);
            if (!frames_compatible(lt.frame, lt.frame_name, rt.frame, rt.frame_name)) {
                throw FrameMismatchError("Incompatible frames in '" + as->op + "' operation",
                                         s->loc.line, s->loc.col);
            }
            std::string key = (as->op == "+>") ? "gain" : "scale";
            auto literal_gain = extract_action_scalar(as->rhs, key, ctx);
            if (rt.geom_type == "Pole") {
                if (literal_gain && *literal_gain < 0.0) {
                    throw UnipolarViolationError(
                        "Negative gain/scale on unipolar Pole '" + rt.symbol_name + "'",
                        s->loc.line, s->loc.col);
                }
            }
            if (rt.geom_type == "Field") {
                for (const auto& fp : ctx.preserved_declared_fields) {
                    if (as->op == "-|" && rt.symbol_name == fp) {
                        throw CrosstalkExceededError(
                            "Direct annihilation of preserved Field '" + fp + "'",
                            s->loc.line, s->loc.col);
                    }
                    if (rt.decoupled_from.count(fp) == 0) {
                        bool self_excite_ok = (as->op == "+>" && rt.symbol_name == fp);
                        bool blend_ok = (as->op == "+>" && rt.blend_sources.count(fp) > 0);
                        if (!self_excite_ok && !blend_ok) {
                            throw CrosstalkExceededError(
                                "Field operation against '" + rt.symbol_name +
                                "' lacks decouple witness for preserved Field '" + fp + "'",
                                s->loc.line, s->loc.col);
                        }
                    }
                }
            } else if (!rt.symbol_name.empty()) {
                std::string base_sym = rt.symbol_name;
                auto dot_pos = base_sym.find('.');
                if (dot_pos != std::string::npos) {
                    base_sym = base_sym.substr(0, dot_pos);
                }
                if (ctx.preserved_declared_symbols.count(base_sym) > 0) {
                    throw CrosstalkExceededError(
                        "Direct intervention on preserved symbol '" + base_sym + "'",
                        s->loc.line, s->loc.col);
                }
            }
            if (!literal_gain) {
                throw EnergyUnboundedError(
                    "Cannot statically bound " + key + " for '" + as->op + "' action",
                    s->loc.line, s->loc.col);
            }
            ctx.last_action_loc = s->loc;
            constexpr double kRefStreamEnergy = 32.0;
            return ((*literal_gain) * (*literal_gain)) / kRefStreamEnergy;
        }
        if (const auto* es = std::get_if<ExprStmt>(&s->node)) {
            double before_sub_e = ctx.sub_energy_extra;
            check_expr(es->expr, ctx);
            double delta_sub_e = ctx.sub_energy_extra - before_sub_e;
            ctx.sub_energy_extra = before_sub_e;
            return delta_sub_e;
        }
        return 0.0;
    }

    const TypedProgram* active_prog_ = nullptr;

    FlowEffectSummary run_flow_check(const std::string& flow_name,
                                     std::vector<std::string>& call_stack,
                                     const std::set<std::string>& extra_fields,
                                     const std::set<std::string>& extra_symbols,
                                     double caller_approx_cap) {
        const FlowEntry& entry = flow_registry_.at(flow_name);
        const FlowDecl* fd = entry.decl;
        const ResolvedFlow& rf = entry.base;

        double eff_approx = std::min(rf.approx_bound, caller_approx_cap);
        FlowCheckCtx ctx{
            *active_prog_,
            {},
            active_prog_->const_numbers,
            {},
            entry.preserved_declared_fields,
            entry.preserved_declared_symbols,
            rf.crosstalk_bound,
            eff_approx,
            rf.energy_bound,
            entry.has_xt,
            entry.has_approx,
            entry.has_energy,
            0.0,
            0.0,
            0.0,
            {},
            &call_stack
        };
        ctx.preserved_declared_fields.insert(extra_fields.begin(), extra_fields.end());
        ctx.preserved_declared_symbols.insert(extra_symbols.begin(), extra_symbols.end());

        std::string flow_frame_name;
        if (fd->return_frame) {
            flow_frame_name = resolve_frame_ref(*fd->return_frame).second;
        } else if (!fd->params.empty() && fd->params[0].frame) {
            flow_frame_name = resolve_frame_ref(*fd->params[0].frame).second;
        }

        bool has_state = false;
        for (const auto& p : fd->params) {
            if (p.name == "state") {
                has_state = true;
            }
            ExprType pt;
            pt.geom_type = format_type_ref(p.type);
            if (p.frame) {
                auto [st, fn] = resolve_frame_ref(*p.frame);
                pt.frame = st;
                pt.frame_name = fn;
            } else {
                pt.frame = rf.stage;
                pt.frame_name = flow_frame_name;
            }
            ctx.env[p.name] = pt;
        }
        if (!has_state) {
            ctx.env["state"] = ExprType{"Residual", rf.stage, flow_frame_name, 0.0, 1, "state", {}, {}, false};
        }

        call_stack.push_back(flow_name);
        double total_energy = 0.0;
        for (const auto& st : fd->body.stmts) {
            total_energy += check_stmt(st, ctx);
        }
        call_stack.pop_back();

        if (total_energy > rf.energy_bound + 1e-12) {
            SourceLoc eloc = ctx.last_action_loc.line > 0 ? ctx.last_action_loc : fd->body.loc;
            throw EnergyBudgetExceededError(
                "Literal energy consumption " + std::to_string(total_energy) +
                " exceeds contract energy <= " + std::to_string(rf.energy_bound),
                eloc.line, eloc.col);
        }

        FlowEffectSummary summary;
        summary.actual_energy = total_energy;
        summary.declared_xt = rf.crosstalk_bound;
        summary.declared_approx = rf.approx_bound;
        summary.declared_energy = rf.energy_bound;
        summary.has_xt = entry.has_xt;
        summary.has_approx = entry.has_approx;
        summary.has_energy = entry.has_energy;
        summary.return_geom_type = entry.return_geom_type;
        summary.return_stage = entry.return_stage;
        summary.return_frame_name = entry.return_frame_name;
        return summary;
    }

    void check_flows(TypedProgram& out) {
        active_prog_ = &out;
        std::vector<std::string> flow_order;

        for (const auto& item : ast_.items) {
            const auto* fd = std::get_if<FlowDecl>(&item.node);
            if (!fd) {
                continue;
            }
            FlowEntry fe;
            fe.item = &item;
            fe.decl = fd;

            ResolvedFlow rf;
            rf.decl = *fd;
            rf.target_mode_hint = target_mode_from_annotations(item.annotations);
            rf.is_helper = has_annotation(item.annotations, "imported_helper");

            StageSpec flow_stage = stage_from_annotations(item.annotations);
            std::string flow_frame_name;
            if (fd->return_frame) {
                auto [st, fn] = resolve_frame_ref(*fd->return_frame);
                flow_stage = st;
                flow_frame_name = fn;
            } else if (!fd->params.empty() && fd->params[0].frame) {
                auto [st, fn] = resolve_frame_ref(*fd->params[0].frame);
                flow_stage = st;
                flow_frame_name = fn;
            }
            rf.stage = flow_stage;

            if (fd->has_contract) {
                for (const auto& c : fd->contract.constraints) {
                    if (c.kind == "crosstalk") {
                        rf.crosstalk_bound = c.number_val;
                        rf.has_crosstalk_constraint = true;
                        fe.has_xt = true;
                    } else if (c.kind == "approx") {
                        rf.approx_bound = c.number_val;
                        rf.has_approx_constraint = true;
                        fe.has_approx = true;
                    } else if (c.kind == "energy") {
                        rf.energy_bound = c.number_val;
                        rf.has_energy_constraint = true;
                        fe.has_energy = true;
                    } else if (c.kind == "preserve") {
                        for (const auto& p : c.preserve_paths) {
                            rf.preserve.push_back(join_path(p));
                        }
                    }
                }
            }

            for (const auto& pname : rf.preserve) {
                auto sit = out.symbols.find(pname);
                if (sit != out.symbols.end()) {
                    if (sit->second.kind == "field") {
                        fe.preserved_declared_fields.insert(pname);
                    } else {
                        fe.preserved_declared_symbols.insert(pname);
                    }
                }
            }

            fe.return_geom_type = fd->return_type ? format_type_ref(*fd->return_type) : "Residual";
            fe.return_stage = flow_stage;
            fe.return_frame_name = flow_frame_name;
            fe.base = std::move(rf);
            flow_registry_[fd->name] = std::move(fe);
            flow_order.push_back(fd->name);
        }

        for (const auto& fname : flow_order) {
            std::vector<std::string> call_stack;
            run_flow_check(fname, call_stack, {}, {}, flow_registry_.at(fname).base.approx_bound);
        }

        for (const auto& fname : flow_order) {
            ResolvedFlow rf = flow_registry_.at(fname).base;
            if (called_flows_.count(fname) && !has_annotation(flow_registry_.at(fname).item->annotations, "entry")) {
                rf.is_helper = true;
            }
            out.flows.push_back(std::move(rf));
        }
    }

    void validate_test_rhs_expr(const ExprPtr& e, const TypedProgram& out, const SourceLoc& loc) const {
        if (!e) return;
        validate_const_expr_refs(e, out);
        auto num = eval_const_number(e, &out.const_numbers);
        if (!num || !std::isfinite(*num)) {
            throw ParseError("Expected finite numeric expression on RHS of expect assertion",
                             loc.line, loc.col);
        }
    }

    void validate_string_expr(const ExprPtr& e, const TypedProgram& out, const SourceLoc& loc) const {
        if (!e) {
            throw ParseError("Expected string expression in test expectation", loc.line, loc.col);
        }
        if (std::holds_alternative<StringLitExpr>(e->node)) {
            return;
        }
        if (const auto* pe = std::get_if<PathExpr>(&e->node)) {
            std::string name = join_path(pe->parts);
            if (out.const_strings.count(name)) {
                return;
            }
            if (out.const_numbers.count(name) || out.const_bools.count(name) ||
                out.const_lists.count(name) || out.symbols.count(name)) {
                throw ParseError("Expected string constant for '" + name + "' in test expectation",
                                 e->loc.line, e->loc.col);
            }
            throw UndefinedSymbolError("Undefined constant '" + name + "' in test expectation",
                                       e->loc.line, e->loc.col);
        }
        throw ParseError("Expected string expression in test expectation", loc.line, loc.col);
    }

    void validate_string_or_list_expr(const ExprPtr& e, const TypedProgram& out, const SourceLoc& loc) const {
        if (!e) {
            throw ParseError("Expected string or string list expression in kl_drift",
                             loc.line, loc.col);
        }
        if (std::holds_alternative<StringLitExpr>(e->node)) {
            return;
        }
        if (const auto* pe = std::get_if<PathExpr>(&e->node)) {
            std::string name = join_path(pe->parts);
            if (out.const_strings.count(name) || out.const_lists.count(name)) {
                return;
            }
            if (out.const_numbers.count(name) || out.const_bools.count(name) ||
                out.symbols.count(name)) {
                throw ParseError("Expected string or string list constant for '" + name + "'",
                                 e->loc.line, e->loc.col);
            }
            throw UndefinedSymbolError("Undefined constant '" + name + "' in kl_drift",
                                       e->loc.line, e->loc.col);
        }
        if (const auto* le = std::get_if<ListExpr>(&e->node)) {
            for (const auto& elem : le->elements) {
                validate_string_expr(elem, out, loc);
            }
            return;
        }
        throw ParseError("Expected string or list of strings in kl_drift", loc.line, loc.col);
    }

    std::string validate_nonempty_string_expr(const ExprPtr& e, const TypedProgram& out,
                                              const SourceLoc& loc) const {
        if (!e) {
            throw ParseError("Expected string expression in test expectation", loc.line, loc.col);
        }
        if (const auto* sl = std::get_if<StringLitExpr>(&e->node)) {
            if (sl->value.empty()) {
                throw ParseError("Expected non-empty string in test expectation", loc.line, loc.col);
            }
            return sl->value;
        }
        if (const auto* pe = std::get_if<PathExpr>(&e->node)) {
            std::string name = join_path(pe->parts);
            auto it = out.const_strings.find(name);
            if (it != out.const_strings.end()) {
                if (it->second.empty()) {
                    throw ParseError("Expected non-empty string constant '" + name + "' in test expectation",
                                     e->loc.line, e->loc.col);
                }
                return it->second;
            }
            if (out.const_numbers.count(name) || out.const_bools.count(name) ||
                out.const_lists.count(name) || out.symbols.count(name)) {
                throw ParseError("Expected string constant for '" + name + "' in test expectation",
                                 e->loc.line, e->loc.col);
            }
            throw UndefinedSymbolError("Undefined constant '" + name + "' in test expectation",
                                       e->loc.line, e->loc.col);
        }
        throw ParseError("Expected string expression in test expectation", loc.line, loc.col);
    }

    void check_tests(TypedProgram& out) const {
        std::set<std::string> seen_names;
        for (const auto& item : ast_.items) {
            const auto* td = std::get_if<TestDecl>(&item.node);
            if (!td) continue;
            if (!seen_names.insert(td->name).second) {
                throw ParseError("Duplicate test declaration '" + td->name + "'",
                                 item.loc.line, item.loc.col);
            }
            for (const auto& a : td->assertions) {
                if (a.rhs_is_base && a.kind != "resonance" && a.kind != "logit_margin") {
                    throw ParseError("'base' comparison is only allowed on resonance and logit_margin assertions",
                                     a.loc.line, a.loc.col);
                }
                if (a.kind == "top_token") {
                    if (a.cmp_op != "==" && a.cmp_op != "!=") {
                        throw ParseError("top_token expectation only supports '==' or '!='",
                                         a.loc.line, a.loc.col);
                    }
                    (void)validate_nonempty_string_expr(a.prompt_expr, out, a.loc);
                    (void)validate_nonempty_string_expr(a.target_str_expr, out, a.loc);
                    continue;
                }
                if (a.rhs_expr) {
                    validate_test_rhs_expr(a.rhs_expr, out, a.loc);
                } else if (!a.rhs_is_base) {
                    throw ParseError("Missing RHS expression in expect assertion", a.loc.line, a.loc.col);
                }
                if (a.kind == "resonance") {
                    validate_string_expr(a.prompt_expr, out, a.loc);
                    std::string sym_name = join_path(a.symbol_path);
                    auto sit = out.symbols.find(sym_name);
                    if (sit == out.symbols.end()) {
                        throw UndefinedSymbolError("Undefined symbol '" + sym_name + "' in resonance test",
                                                   a.loc.line, a.loc.col);
                    }
                    if (!a.pole_side.empty()) {
                        const auto& sym = sit->second;
                        if (sym.kind != "dipole" && sym.kind != "pole") {
                            throw ParseError("Pole side access '." + a.pole_side +
                                             "' requires a dipole or pole symbol",
                                             a.loc.line, a.loc.col);
                        }
                        if (a.pole_side != "+pole" && a.pole_side != "-pole") {
                            throw ParseError("Invalid pole side '." + a.pole_side + "'",
                                             a.loc.line, a.loc.col);
                        }
                        if (sym.kind == "dipole" && sym.poles.count(a.pole_side) == 0) {
                            throw UndefinedSymbolError("Dipole '" + sym_name + "' has no '" + a.pole_side + "'",
                                                       a.loc.line, a.loc.col);
                        }
                    }
                } else if (a.kind == "kl_drift") {
                    validate_string_or_list_expr(a.prompt_expr, out, a.loc);
                } else if (a.kind == "logit_margin") {
                    (void)validate_nonempty_string_expr(a.prompt_expr, out, a.loc);
                    (void)validate_nonempty_string_expr(a.target_str_expr, out, a.loc);
                    (void)validate_nonempty_string_expr(a.foil_str_expr, out, a.loc);
                }
            }
            out.tests.push_back(*td);
        }
    }
};

}

bool is_subtype(const std::string& sub, const std::string& sup) {
    ParsedType a = parse_type_str(sub);
    ParsedType b = parse_type_str(sup);
    if (a.base == b.base) {
        if (b.args.empty()) {
            return true;
        }
        if (a.base == "Subspace" && a.args.size() == 1 && b.args.size() == 1 &&
            is_digits(a.args[0]) && is_digits(b.args[0])) {
            return std::stoi(a.args[0]) <= std::stoi(b.args[0]);
        }
        if (a.base == "Manifold" && a.args.size() == 2 && b.args.size() == 2) {
            bool topo_ok = (a.args[0] == b.args[0] || b.args[0] == "Euclidean" || b.args[0] == "Any");
            bool dim_ok = true;
            if (is_digits(a.args[1]) && is_digits(b.args[1])) {
                dim_ok = std::stoi(a.args[1]) <= std::stoi(b.args[1]);
            }
            return topo_ok && dim_ok;
        }
        return a.args == b.args;
    }
    if (a.base == "Pole") {
        if (b.base == "Dipole") return true;
        if (b.base == "Subspace") {
            if (b.args.empty()) return true;
            return is_digits(b.args[0]) && std::stoi(b.args[0]) >= 1;
        }
        if (b.base == "Manifold") {
            if (b.args.empty()) return true;
            bool topo_ok = (b.args[0] == "Euclidean" || b.args[0] == "Flat");
            bool dim_ok = (b.args.size() < 2) || (is_digits(b.args[1]) && std::stoi(b.args[1]) >= 1);
            return topo_ok && dim_ok;
        }
        return false;
    }
    if (a.base == "Dipole") {
        if (b.base == "Subspace") {
            if (b.args.empty()) return true;
            return is_digits(b.args[0]) && std::stoi(b.args[0]) >= 1;
        }
        if (b.base == "Manifold") {
            if (b.args.empty()) return true;
            bool topo_ok = (b.args[0] == "Euclidean" || b.args[0] == "Flat");
            bool dim_ok = (b.args.size() < 2) || (is_digits(b.args[1]) && std::stoi(b.args[1]) >= 1);
            return topo_ok && dim_ok;
        }
        return false;
    }
    if (a.base == "Subspace") {
        if (b.base == "Manifold") {
            if (b.args.empty()) return true;
            bool topo_ok = (b.args[0] == "Euclidean" || b.args[0] == "Flat");
            bool dim_ok = true;
            if (!a.args.empty() && b.args.size() >= 2 && is_digits(a.args[0]) && is_digits(b.args[1])) {
                dim_ok = std::stoi(a.args[0]) <= std::stoi(b.args[1]);
            }
            return topo_ok && dim_ok;
        }
        return false;
    }
    return false;
}

bool stages_overlap(const StageSpec& a, const StageSpec& b) {
    return intervals_overlap(to_interval(a), to_interval(b));
}

TypedProgram typecheck_program(const ProgramAST& ast) {
    TypeChecker checker(ast);
    return checker.run();
}

}
