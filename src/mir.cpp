#include "aic/mir.hpp"
#include "aic/errors.hpp"
#include "aic/parser.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <utility>

namespace aic {

namespace {

MIRStageSpec to_mir_stage(const StageSpec& s) {
    return MIRStageSpec{s.start.empty() ? "All" : s.start, s.end, s.inclusive};
}

std::string join_parts(const std::vector<std::string>& parts) {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            out += "::";
        }
        out += parts[i];
    }
    return out;
}

std::string format_expr_text(const ExprPtr& e) {
    if (!e) {
        return "";
    }
    if (const auto* i = std::get_if<IntLitExpr>(&e->node)) {
        return std::to_string(i->value);
    }
    if (const auto* f = std::get_if<FloatLitExpr>(&e->node)) {
        std::ostringstream oss;
        oss << f->value;
        return oss.str();
    }
    if (const auto* s = std::get_if<StringLitExpr>(&e->node)) {
        return s->value;
    }
    if (const auto* b = std::get_if<BoolLitExpr>(&e->node)) {
        return b->value ? "true" : "false";
    }
    if (const auto* p = std::get_if<PathExpr>(&e->node)) {
        return join_parts(p->parts);
    }
    if (const auto* u = std::get_if<UnaryExpr>(&e->node)) {
        return u->op + format_expr_text(u->operand);
    }
    if (const auto* bin = std::get_if<BinaryExpr>(&e->node)) {
        return format_expr_text(bin->lhs) + " " + bin->op + " " + format_expr_text(bin->rhs);
    }
    if (const auto* m = std::get_if<MemberExpr>(&e->node)) {
        return format_expr_text(m->object) + "." + m->member;
    }
    if (const auto* c = std::get_if<CallExpr>(&e->node)) {
        return format_expr_text(c->callee);
    }
    return "<expr>";
}

std::optional<double> try_eval_number(const ExprPtr& e,
                                      const std::map<std::string, double>* consts = nullptr) {
    if (!e) return std::nullopt;
    if (const auto* f = std::get_if<FloatLitExpr>(&e->node)) return f->value;
    if (const auto* i = std::get_if<IntLitExpr>(&e->node)) return static_cast<double>(i->value);
    if (const auto* p = std::get_if<PathExpr>(&e->node)) {
        if (consts) {
            auto it = consts->find(join_parts(p->parts));
            if (it != consts->end()) return it->second;
        }
        return std::nullopt;
    }
    if (const auto* u = std::get_if<UnaryExpr>(&e->node)) {
        auto v = try_eval_number(u->operand, consts);
        if (!v) return std::nullopt;
        if (u->op == "-") return -(*v);
        if (u->op == "+") return *v;
    }
    if (const auto* b = std::get_if<BinaryExpr>(&e->node)) {
        auto lv = try_eval_number(b->lhs, consts);
        auto rv = try_eval_number(b->rhs, consts);
        if (lv && rv) {
            if (b->op == "+") return *lv + *rv;
            if (b->op == "-") return *lv - *rv;
            if (b->op == "*") return *lv * *rv;
            if (b->op == "/" && *rv != 0.0) return *lv / *rv;
        }
    }
    return std::nullopt;
}

double extract_leading_scalar(const ExprPtr& e, double fallback,
                              const std::map<std::string, double>* consts = nullptr) {
    if (!e) return fallback;
    if (auto exact = try_eval_number(e, consts)) {
        return *exact;
    }
    if (const auto* b = std::get_if<BinaryExpr>(&e->node)) {
        if (b->op == "*") {
            if (auto lv = try_eval_number(b->lhs, consts)) return *lv;
            if (auto rv = try_eval_number(b->rhs, consts)) return *rv;
        }
    }
    return fallback;
}

ArgValue expr_to_arg_value(const ExprPtr& e, const TypedProgram& prog) {
    if (!e) {
        return std::string("");
    }
    if (auto num = try_eval_number(e, &prog.const_numbers)) {
        if (std::holds_alternative<IntLitExpr>(e->node)) {
            return std::get<IntLitExpr>(e->node).value;
        }
        if (const auto* u = std::get_if<UnaryExpr>(&e->node)) {
            if (u->op == "-" && std::holds_alternative<IntLitExpr>(u->operand->node)) {
                return -std::get<IntLitExpr>(u->operand->node).value;
            }
        }
        return *num;
    }
    if (const auto* b = std::get_if<BoolLitExpr>(&e->node)) {
        return b->value;
    }
    if (const auto* s = std::get_if<StringLitExpr>(&e->node)) {
        return s->value;
    }
    if (const auto* p = std::get_if<PathExpr>(&e->node)) {
        std::string key = join_parts(p->parts);
        auto sit = prog.const_strings.find(key);
        if (sit != prog.const_strings.end()) return sit->second;
        auto bit = prog.const_bools.find(key);
        if (bit != prog.const_bools.end()) return bit->second;
        auto lit = prog.const_lists.find(key);
        if (lit != prog.const_lists.end()) return lit->second;
        return key;
    }
    if (const auto* le = std::get_if<ListExpr>(&e->node)) {
        std::vector<std::string> items;
        items.reserve(le->elements.size());
        for (const auto& elem : le->elements) {
            if (const auto* es = std::get_if<StringLitExpr>(&elem->node)) {
                items.push_back(es->value);
            } else {
                items.push_back(format_expr_text(elem));
            }
        }
        return items;
    }
    return format_expr_text(e);
}

MIRAnchor lower_anchor(const AnchorDecl& ad, const TypedProgram& prog) {
    MIRAnchor ma;
    ma.kind = ad.kind;
    for (const auto& a : ad.args) {
        ArgValue val = expr_to_arg_value(a.value, prog);
        if (a.name.empty()) {
            ma.pos_args.push_back(std::move(val));
        } else {
            ma.named_args[a.name] = std::move(val);
        }
    }
    return ma;
}

std::optional<double> arg_as_double(const ArgValue& v) {
    if (const auto* d = std::get_if<double>(&v)) return *d;
    if (const auto* i = std::get_if<std::int64_t>(&v)) return static_cast<double>(*i);
    return std::nullopt;
}

class MIRLowerer {
public:
    explicit MIRLowerer(const TypedProgram& prog) : prog_(prog) {
        for (const auto& rf : prog_.flows) {
            flow_decls_[rf.decl.name] = &rf.decl;
        }
    }

    MIRModule run() {
        MIRModule mod;
        mod.name = prog_.module_name;
        mod.imports = prog_.imports;

        for (const auto& [name, sym] : prog_.symbols) {
            MIRSymbolDecl decl;
            decl.name = sym.name;
            decl.kind = sym.kind;
            decl.frame = to_mir_stage(sym.frame);
            decl.dim = sym.dim;
            decl.topology = sym.topology;
            decl.curvature = sym.curvature;
            decl.rank = sym.rank;
            decl.metric = sym.metric;
            decl.poles = sym.poles;
            for (const auto& ad : sym.anchors) {
                decl.anchors.push_back(lower_anchor(ad, prog_));
            }
            for (const auto& re : sym.rodata_entries) {
                MIRRodataEntry entry;
                entry.key = format_expr_text(re.key);
                entry.value = format_expr_text(re.value);
                for (const auto& attr : re.attrs) {
                    if (!attr.name.empty()) {
                        if (auto num = try_eval_number(attr.value, &prog_.const_numbers)) {
                            entry.attrs[attr.name] = *num;
                        }
                    }
                }
                decl.rodata_entries.push_back(std::move(entry));
            }
            decl.connections = sym.connections;
            mod.declarations[name] = std::move(decl);
        }

        for (const auto& rf : prog_.flows) {
            MIRFlow mf;
            mf.name = rf.decl.name;
            mf.stage = to_mir_stage(rf.stage);
            mf.target_mode_hint = rf.target_mode_hint;
            mf.contract.crosstalk_bound = rf.crosstalk_bound;
            mf.contract.approx_bound = rf.approx_bound;
            mf.contract.energy_bound = rf.energy_bound;
            mf.contract.preserve = rf.preserve;
            mf.is_helper = rf.is_helper;

            LowerCtx ctx;
            lower_stmts(rf.decl.body.stmts, mf.ops, ctx);
            mod.flows.push_back(std::move(mf));
        }

        for (const auto& td : prog_.tests) {
            MIRTestCase tc;
            tc.name = td.name;
            for (const auto& a : td.assertions) {
                MIRExpectation me;
                me.kind = a.kind;
                me.cmp_op = a.cmp_op;
                me.rhs_is_base = a.rhs_is_base;
                me.rhs_val = a.rhs_expr ? try_eval_number(a.rhs_expr, &prog_.const_numbers).value_or(0.0) : 0.0;
                if (a.kind == "resonance") {
                    me.prompt = resolve_const_string(a.prompt_expr);
                    me.symbol = join_parts(a.symbol_path);
                    me.pole = a.pole_side;
                } else if (a.kind == "top_token") {
                    me.prompt = resolve_const_string(a.prompt_expr);
                    me.target_token_str = resolve_const_string(a.target_str_expr);
                } else if (a.kind == "logit_margin") {
                    me.prompt = resolve_const_string(a.prompt_expr);
                    me.target_token_str = resolve_const_string(a.target_str_expr);
                    me.foil_token_str = resolve_const_string(a.foil_str_expr);
                } else if (a.kind == "kl_drift") {
                    if (a.prompt_expr && std::holds_alternative<ListExpr>(a.prompt_expr->node)) {
                        const auto& le = std::get<ListExpr>(a.prompt_expr->node);
                        for (const auto& elem : le.elements) {
                            me.prompts.push_back(resolve_const_string(elem));
                        }
                    } else if (a.prompt_expr && std::holds_alternative<PathExpr>(a.prompt_expr->node)) {
                        std::string k = join_parts(std::get<PathExpr>(a.prompt_expr->node).parts);
                        auto lit = prog_.const_lists.find(k);
                        if (lit != prog_.const_lists.end()) {
                            me.prompts = lit->second;
                        } else {
                            me.prompt = resolve_const_string(a.prompt_expr);
                            me.prompts.push_back(me.prompt);
                        }
                    } else {
                        me.prompt = resolve_const_string(a.prompt_expr);
                        me.prompts.push_back(me.prompt);
                    }
                }
                tc.expectations.push_back(std::move(me));
            }
            mod.tests.push_back(std::move(tc));
        }

        return mod;
    }

private:
    const TypedProgram& prog_;
    std::map<std::string, const FlowDecl*> flow_decls_;
    int next_inline_id_ = 0;

    std::string resolve_const_string(const ExprPtr& e) const {
        if (!e) return "";
        if (const auto* sl = std::get_if<StringLitExpr>(&e->node)) {
            return sl->value;
        }
        if (const auto* pe = std::get_if<PathExpr>(&e->node)) {
            std::string k = join_parts(pe->parts);
            auto it = prog_.const_strings.find(k);
            if (it != prog_.const_strings.end()) {
                return it->second;
            }
        }
        return format_expr_text(e);
    }

    struct LowerCtx {
        int* temp_counter = nullptr;
        int owned_temp_counter = 0;
        std::string local_prefix;
        std::map<std::string, std::string> var_renames;
        std::map<std::string, std::pair<std::string, std::string>> target_aliases;
        std::set<std::string> gate_bindings;

        LowerCtx() : temp_counter(&owned_temp_counter) {}

        std::string next_temp() {
            return "%t" + std::to_string((*temp_counter)++);
        }

        std::string resolve_name(const std::string& n) const {
            auto it = var_renames.find(n);
            if (it != var_renames.end()) {
                return it->second;
            }
            return n;
        }
    };

    MIRStageSpec resolve_frame_ref_stage(const FrameRef& ref) const {
        if (ref.is_explicit_frame) {
            return to_mir_stage(ref.stage);
        }
        std::string name = join_parts(ref.path);
        auto it = prog_.symbols.find(name);
        if (it != prog_.symbols.end()) {
            return to_mir_stage(it->second.frame);
        }
        return MIRStageSpec{name, "", false};
    }

    std::pair<double, std::string> extract_scalar_and_gate(
        const ExprPtr& e,
        const LowerCtx& ctx) const {
        if (auto exact = try_eval_number(e, &prog_.const_numbers)) {
            return {*exact, ""};
        }
        auto dynamic_path = [&](const ExprPtr& candidate) -> std::string {
            const auto* path = candidate ? std::get_if<PathExpr>(&candidate->node) : nullptr;
            if (path == nullptr) {
                return "";
            }
            const std::string name = ctx.resolve_name(join_parts(path->parts));
            return ctx.gate_bindings.contains(name) ? name : "";
        };
        if (const std::string gate = dynamic_path(e); !gate.empty()) {
            return {1.0, gate};
        }
        if (const auto* binary = e ? std::get_if<BinaryExpr>(&e->node) : nullptr;
            binary != nullptr && binary->op == "*") {
            if (auto coefficient = try_eval_number(binary->lhs, &prog_.const_numbers)) {
                if (const std::string gate = dynamic_path(binary->rhs); !gate.empty()) {
                    return {*coefficient, gate};
                }
            }
            if (auto coefficient = try_eval_number(binary->rhs, &prog_.const_numbers)) {
                if (const std::string gate = dynamic_path(binary->lhs); !gate.empty()) {
                    return {*coefficient, gate};
                }
            }
        }
        return {extract_leading_scalar(e, 1.0, &prog_.const_numbers), ""};
    }

    std::pair<std::string, std::string> extract_target_and_pole(const ExprPtr& e, const LowerCtx& ctx) const {
        if (!e) {
            return {"", ""};
        }
        if (const auto* ce = std::get_if<CallExpr>(&e->node)) {
            return extract_target_and_pole(ce->callee, ctx);
        }
        if (const auto* me = std::get_if<MemberExpr>(&e->node)) {
            auto base = extract_target_and_pole(me->object, ctx);
            if (me->member == "+pole" || me->member == "-pole") {
                return {base.first, me->member};
            }
            return {base.first, me->member};
        }
        if (const auto* pe = std::get_if<PathExpr>(&e->node)) {
            std::string name = ctx.resolve_name(join_parts(pe->parts));
            auto it = ctx.target_aliases.find(name);
            if (it != ctx.target_aliases.end()) {
                return it->second;
            }
            return {name, ""};
        }
        return {format_expr_text(e), ""};
    }

    std::string lower_expr_to_name(const ExprPtr& e, std::vector<MIROp>& out_ops, LowerCtx& ctx) {
        if (!e) {
            return "";
        }
        if (const auto* pe = std::get_if<PathExpr>(&e->node)) {
            return ctx.resolve_name(join_parts(pe->parts));
        }
        if (std::holds_alternative<AttendExpr>(e->node) ||
            std::holds_alternative<TransportExpr>(e->node) ||
            std::holds_alternative<FieldOpExpr>(e->node) ||
            std::holds_alternative<FieldCallExpr>(e->node)) {
            std::string dst = ctx.next_temp();
            lower_assigned_expr(dst, e, out_ops, ctx);
            return dst;
        }
        if (const auto* ce = std::get_if<CallExpr>(&e->node)) {
            if (const auto* pe = std::get_if<PathExpr>(&ce->callee->node)) {
                std::string fname = join_parts(pe->parts);
                if (flow_decls_.count(fname)) {
                    return inline_flow_call(fname, ce->args, out_ops, ctx);
                }
            }
        }
        if (const auto* be = std::get_if<BinaryExpr>(&e->node)) {
            if (be->op == "=>") {
                std::string dst = ctx.next_temp();
                lower_assigned_expr(dst, e, out_ops, ctx);
                return dst;
            }
        }
        return format_expr_text(e);
    }

    std::string inline_flow_call(const std::string& fname,
                                 const std::vector<Arg>& args,
                                 std::vector<MIROp>& out_ops,
                                 LowerCtx& ctx) {
        const FlowDecl* fd = flow_decls_.at(fname);
        LowerCtx sub_ctx;
        sub_ctx.temp_counter = ctx.temp_counter;
        sub_ctx.local_prefix = "__in" + std::to_string(next_inline_id_++) + "_";

        std::string ret_target = ctx.resolve_name("state");
        if (!fd->params.empty()) {
            if (args.empty() && fd->params.size() == 1) {
                sub_ctx.var_renames[fd->params[0].name] = ret_target;
            } else {
                for (std::size_t i = 0; i < args.size(); ++i) {
                    std::string arg_name = lower_expr_to_name(args[i].value, out_ops, ctx);
                    if (i == 0) {
                        ret_target = arg_name;
                    }
                    if (!args[i].name.empty()) {
                        sub_ctx.var_renames[args[i].name] = arg_name;
                        if (args[i].name == fd->params[0].name) {
                            ret_target = arg_name;
                        }
                    } else if (i < fd->params.size()) {
                        sub_ctx.var_renames[fd->params[i].name] = arg_name;
                    }
                }
            }
        }
        lower_stmts(fd->body.stmts, out_ops, sub_ctx);
        return ret_target;
    }

    ArgValue resolved_arg_value(const ExprPtr& e, const LowerCtx& ctx) const {
        if (const auto* pe = std::get_if<PathExpr>(&e->node)) {
            std::string raw = join_parts(pe->parts);
            auto it = ctx.var_renames.find(raw);
            if (it != ctx.var_renames.end()) {
                return it->second;
            }
        }
        return expr_to_arg_value(e, prog_);
    }

    void fill_args_into_map(const std::vector<Arg>& args,
                            std::map<std::string, ArgValue>& out_map,
                            std::vector<std::string>& pos_names,
                            std::vector<MIROp>& out_ops,
                            LowerCtx& ctx) {
        for (const auto& a : args) {
            if (a.name.empty()) {
                if (std::holds_alternative<AttendExpr>(a.value->node) ||
                    std::holds_alternative<TransportExpr>(a.value->node) ||
                    std::holds_alternative<FieldOpExpr>(a.value->node) ||
                    std::holds_alternative<FieldCallExpr>(a.value->node)) {
                    pos_names.push_back(lower_expr_to_name(a.value, out_ops, ctx));
                } else {
                    ArgValue val = resolved_arg_value(a.value, ctx);
                    if (const auto* sp = std::get_if<std::string>(&val)) {
                        pos_names.push_back(*sp);
                    } else {
                        pos_names.push_back(format_expr_text(a.value));
                    }
                    out_map["arg" + std::to_string(pos_names.size() - 1)] = std::move(val);
                }
            } else {
                out_map[a.name] = resolved_arg_value(a.value, ctx);
            }
        }
    }

    bool lower_assigned_expr(const std::string& dst, const ExprPtr& e,
                             std::vector<MIROp>& out_ops, LowerCtx& ctx) {
        if (!e) {
            return false;
        }
        if (const auto* ce = std::get_if<CallExpr>(&e->node)) {
            if (const auto* pe = std::get_if<PathExpr>(&ce->callee->node)) {
                std::string fname = join_parts(pe->parts);
                if (flow_decls_.count(fname)) {
                    std::string ret_target = inline_flow_call(fname, ce->args, out_ops, ctx);
                    if (!dst.empty()) {
                        ctx.var_renames[dst] = ret_target;
                    }
                    return true;
                }
            }
        }
        if (const auto* ae = std::get_if<AttendExpr>(&e->node)) {
            MIROp op;
            op.op = "attend";
            op.dst = dst;
            std::vector<std::string> pos;
            fill_args_into_map(ae->args, op.args, pos, out_ops, ctx);
            if (op.args.count("key")) {
                if (const auto* s = std::get_if<std::string>(&op.args["key"])) {
                    op.target = *s;
                }
            } else if (pos.size() >= 2) {
                op.target = pos[1];
            }
            if (op.args.count("value")) {
                if (const auto* s = std::get_if<std::string>(&op.args["value"])) {
                    op.rhs_target = *s;
                }
            } else if (pos.size() >= 3) {
                op.rhs_target = pos[2];
            }
            out_ops.push_back(std::move(op));
            return true;
        }
        if (const auto* te = std::get_if<TransportExpr>(&e->node)) {
            std::string inner = lower_expr_to_name(te->expr, out_ops, ctx);
            MIROp op;
            op.op = "frame_transport";
            op.dst = dst;
            op.target = inner;
            op.src_frame = resolve_frame_ref_stage(te->src_frame);
            op.dst_frame = resolve_frame_ref_stage(te->dst_frame);
            out_ops.push_back(std::move(op));
            return true;
        }
        if (const auto* foe = std::get_if<FieldOpExpr>(&e->node)) {
            std::string lhs_name = lower_expr_to_name(foe->lhs, out_ops, ctx);
            std::vector<Arg> merged_args = foe->args;
            ExprPtr rhs_core = foe->rhs;
            if (const auto* rcall = std::get_if<CallExpr>(&foe->rhs->node)) {
                rhs_core = rcall->callee;
                for (const auto& a : rcall->args) {
                    merged_args.push_back(a);
                }
            }
            std::string rhs_name = lower_expr_to_name(rhs_core, out_ops, ctx);

            MIROp op;
            op.op = "field_" + foe->op;
            op.dst = dst;
            op.target = lhs_name;
            op.rhs_target = rhs_name;
            std::vector<std::string> pos;
            fill_args_into_map(merged_args, op.args, pos, out_ops, ctx);
            if (foe->op == "decouple") {
                op.scalar_arg = 1e-4;
                if (op.args.count("reg")) {
                    if (auto d = arg_as_double(op.args["reg"])) op.scalar_arg = *d;
                }
            } else if (foe->op == "temper") {
                op.scalar_arg = 1.0;
                if (op.args.count("gain")) {
                    if (auto d = arg_as_double(op.args["gain"])) op.scalar_arg = *d;
                }
            } else if (foe->op == "blend") {
                op.scalar_arg = 0.5;
                if (op.args.count("alpha")) {
                    if (auto d = arg_as_double(op.args["alpha"])) op.scalar_arg = *d;
                }
            }
            out_ops.push_back(std::move(op));
            return true;
        }
        if (const auto* fce = std::get_if<FieldCallExpr>(&e->node)) {
            MIROp op;
            op.op = "field_" + fce->op;
            op.dst = dst;
            std::vector<std::string> pos;
            fill_args_into_map(fce->args, op.args, pos, out_ops, ctx);
            if (!pos.empty()) {
                op.target = pos[0];
                op.args.erase("arg0");
            }
            if (pos.size() >= 2) {
                op.rhs_target = pos[1];
                op.args.erase("arg1");
            }
            if (fce->op == "temper") {
                op.scalar_arg = 1.0;
                if (op.args.count("gain")) {
                    if (auto d = arg_as_double(op.args["gain"])) op.scalar_arg = *d;
                }
            } else if (fce->op == "blend") {
                op.scalar_arg = 0.5;
                if (op.args.count("alpha")) {
                    if (auto d = arg_as_double(op.args["alpha"])) op.scalar_arg = *d;
                } else if (pos.size() >= 3) {
                    if (op.args.count("arg2")) {
                        if (auto d = arg_as_double(op.args["arg2"])) op.scalar_arg = *d;
                    }
                }
            } else if (fce->op == "decouple") {
                op.scalar_arg = 1e-4;
                if (op.args.count("reg")) {
                    if (auto d = arg_as_double(op.args["reg"])) op.scalar_arg = *d;
                }
            }
            out_ops.push_back(std::move(op));
            return true;
        }
        if (const auto* be = std::get_if<BinaryExpr>(&e->node)) {
            if (be->op == "=>") {
                std::string lhs_name = lower_expr_to_name(be->lhs, out_ops, ctx);
                MIROp op;
                op.op = "transport_map";
                op.dst = dst;
                op.args["lhs"] = lhs_name;
                if (const auto* rcall = std::get_if<CallExpr>(&be->rhs->node)) {
                    auto [tgt, meth] = extract_target_and_pole(rcall->callee, ctx);
                    op.target = tgt;
                    op.rhs_target = meth;
                    if (!meth.empty()) {
                        op.args["method"] = meth;
                    }
                    std::vector<std::string> pos;
                    fill_args_into_map(rcall->args, op.args, pos, out_ops, ctx);
                } else {
                    auto [tgt, meth] = extract_target_and_pole(be->rhs, ctx);
                    op.target = tgt;
                    op.rhs_target = meth.empty() ? lhs_name : meth;
                }
                out_ops.push_back(std::move(op));
                return true;
            }
        }
        return false;
    }

    MIROp make_when_op(const ExprPtr& cond, const std::string& bind_name,
                       std::vector<MIROp> body_ops, std::vector<MIROp> else_ops,
                       std::vector<MIROp>& pre_ops, LowerCtx& ctx) {
        if (const auto* be = std::get_if<BinaryExpr>(&cond->node)) {
            if (be->op == "~>") {
                std::string lhs_name = lower_expr_to_name(be->lhs, pre_ops, ctx);
                MIROp op;
                op.op = "when";
                op.dst = bind_name;
                op.rhs_target = lhs_name;
                op.scalar_arg = 0.5;
                if (const auto* rcall = std::get_if<CallExpr>(&be->rhs->node)) {
                    auto [tgt, pol] = extract_target_and_pole(rcall->callee, ctx);
                    op.target = tgt;
                    op.pole = pol;
                    std::vector<std::string> pos;
                    fill_args_into_map(rcall->args, op.args, pos, pre_ops, ctx);
                    if (op.args.count("threshold")) {
                        if (auto d = arg_as_double(op.args["threshold"])) {
                            op.scalar_arg = *d;
                        }
                    } else if (!pos.empty() && op.args.count("arg0")) {
                        if (auto d = arg_as_double(op.args["arg0"])) {
                            op.scalar_arg = *d;
                        }
                    }
                } else {
                    auto [tgt, pol] = extract_target_and_pole(be->rhs, ctx);
                    op.target = tgt;
                    op.pole = pol;
                }
                op.body = std::move(body_ops);
                op.else_body = std::move(else_ops);
                return op;
            }
        }
        MIROp op;
        op.op = "when";
        op.dst = bind_name;
        auto [tgt, pol] = extract_target_and_pole(cond, ctx);
        op.target = tgt;
        op.pole = pol;
        op.scalar_arg = 0.5;
        op.body = std::move(body_ops);
        op.else_body = std::move(else_ops);
        return op;
    }

    void lower_when_cond(const ExprPtr& cond, const std::string& bind_name,
                         std::vector<MIROp> body_ops, std::vector<MIROp> else_ops,
                         std::vector<MIROp>& out_ops, LowerCtx& ctx) {
        if (const auto* be = std::get_if<BinaryExpr>(&cond->node)) {
            if (be->op == "&&") {
                std::vector<MIROp> inner_list;
                lower_when_cond(be->rhs, bind_name, std::move(body_ops), {}, inner_list, ctx);
                lower_when_cond(be->lhs, "", std::move(inner_list), std::move(else_ops), out_ops, ctx);
                return;
            }
            if (be->op == "||") {
                lower_when_cond(be->lhs, bind_name, body_ops, {}, out_ops, ctx);
                lower_when_cond(be->rhs, bind_name, std::move(body_ops), std::move(else_ops), out_ops, ctx);
                return;
            }
        }
        out_ops.push_back(make_when_op(cond, bind_name, std::move(body_ops), std::move(else_ops), out_ops, ctx));
    }

    void lower_stmts(const std::vector<StmtPtr>& stmts, std::vector<MIROp>& out_ops, LowerCtx& ctx) {
        for (const auto& s : stmts) {
            if (!s) continue;
            if (const auto* ls = std::get_if<LetStmt>(&s->node)) {
                std::string target_dst = ctx.local_prefix.empty() ? ls->name : (ctx.local_prefix + ls->name);
                if (!ctx.local_prefix.empty()) {
                    ctx.var_renames[ls->name] = target_dst;
                }
                if (!lower_assigned_expr(target_dst, ls->value, out_ops, ctx)) {
                    auto tp = extract_target_and_pole(ls->value, ctx);
                    if (!tp.first.empty()) {
                        ctx.target_aliases[target_dst] = tp;
                    }
                }
            } else if (const auto* ws = std::get_if<WhenStmt>(&s->node)) {
                std::string bind_dst = ws->bind_name;
                LowerCtx body_ctx = ctx;
                if (!ws->bind_name.empty() && !ctx.local_prefix.empty()) {
                    bind_dst = ctx.local_prefix + ws->bind_name;
                    body_ctx.var_renames[ws->bind_name] = bind_dst;
                }
                if (!bind_dst.empty()) {
                    body_ctx.gate_bindings.insert(bind_dst);
                }
                std::vector<MIROp> body_ops;
                lower_stmts(ws->body.stmts, body_ops, body_ctx);
                std::vector<MIROp> else_ops;
                if (ws->has_else) {
                    LowerCtx else_ctx = ctx;
                    lower_stmts(ws->else_stmts, else_ops, else_ctx);
                }
                lower_when_cond(ws->cond, bind_dst, std::move(body_ops), std::move(else_ops), out_ops, ctx);
            } else if (const auto* rs = std::get_if<RodataStmt>(&s->node)) {
                MIROp op;
                op.op = "rodata_" + rs->op;
                op.target = join_parts(rs->target_path);
                std::vector<std::string> pos;
                fill_args_into_map(rs->args, op.args, pos, out_ops, ctx);
                if (op.args.count("sharpness")) {
                    if (auto d = arg_as_double(op.args["sharpness"])) {
                        op.scalar_arg = *d;
                    }
                }
                out_ops.push_back(std::move(op));
            } else if (const auto* as = std::get_if<ActionStmt>(&s->node)) {
                std::string lhs_name = lower_expr_to_name(as->lhs, out_ops, ctx);
                MIROp op;
                op.op = (as->op == "+>") ? "excite" : "annihilate";
                op.dst = lhs_name;
                op.scalar_arg = 1.0;
                std::string key = (as->op == "+>") ? "gain" : "scale";

                if (const auto* ce = std::get_if<CallExpr>(&as->rhs->node)) {
                    if (std::holds_alternative<FieldOpExpr>(ce->callee->node) ||
                        std::holds_alternative<FieldCallExpr>(ce->callee->node) ||
                        std::holds_alternative<TransportExpr>(ce->callee->node)) {
                        op.target = lower_expr_to_name(ce->callee, out_ops, ctx);
                    } else {
                        auto [tgt, pol] = extract_target_and_pole(ce->callee, ctx);
                        op.target = tgt;
                        op.pole = pol;
                    }
                    std::vector<std::string> pos;
                    fill_args_into_map(ce->args, op.args, pos, out_ops, ctx);
                    for (std::size_t i = 0; i < ce->args.size(); ++i) {
                        if (ce->args[i].name == key || (ce->args[i].name.empty() && i == 0)) {
                            auto [scalar, gate] = extract_scalar_and_gate(ce->args[i].value, ctx);
                            op.scalar_arg = scalar;
                            op.scalar_gate = std::move(gate);
                            break;
                        }
                    }
                } else if (const auto* foe = std::get_if<FieldOpExpr>(&as->rhs->node)) {
                    op.target = lower_expr_to_name(as->rhs, out_ops, ctx);
                    std::vector<std::string> pos;
                    fill_args_into_map(foe->args, op.args, pos, out_ops, ctx);
                    for (const auto& a : foe->args) {
                        if (a.name == key) {
                            auto [scalar, gate] = extract_scalar_and_gate(a.value, ctx);
                            op.scalar_arg = scalar;
                            op.scalar_gate = std::move(gate);
                            break;
                        }
                    }
                } else if (const auto* fce = std::get_if<FieldCallExpr>(&as->rhs->node)) {
                    op.target = lower_expr_to_name(as->rhs, out_ops, ctx);
                    std::vector<std::string> pos;
                    fill_args_into_map(fce->args, op.args, pos, out_ops, ctx);
                    for (const auto& a : fce->args) {
                        if (a.name == key) {
                            auto [scalar, gate] = extract_scalar_and_gate(a.value, ctx);
                            op.scalar_arg = scalar;
                            op.scalar_gate = std::move(gate);
                            break;
                        }
                    }
                } else if (std::holds_alternative<TransportExpr>(as->rhs->node)) {
                    op.target = lower_expr_to_name(as->rhs, out_ops, ctx);
                } else {
                    auto [tgt, pol] = extract_target_and_pole(as->rhs, ctx);
                    op.target = tgt;
                    op.pole = pol;
                }
                out_ops.push_back(std::move(op));
            } else if (const auto* es = std::get_if<ExprStmt>(&s->node)) {
                if (std::holds_alternative<AttendExpr>(es->expr->node) ||
                    std::holds_alternative<TransportExpr>(es->expr->node) ||
                    std::holds_alternative<FieldOpExpr>(es->expr->node) ||
                    std::holds_alternative<FieldCallExpr>(es->expr->node) ||
                    std::holds_alternative<CallExpr>(es->expr->node)) {
                    lower_assigned_expr("", es->expr, out_ops, ctx);
                } else if (const auto* be = std::get_if<BinaryExpr>(&es->expr->node)) {
                    if (be->op == "=>") {
                        lower_assigned_expr("", es->expr, out_ops, ctx);
                    }
                }
            }
        }
    }
};

std::string format_stage(const MIRStageSpec& s) {
    if (s.end.empty()) {
        return "Frame<" + s.start + ">";
    }
    return "Frame<" + s.start + (s.inclusive ? "..=" : "..") + s.end + ">";
}

std::string format_arg_value(const ArgValue& v) {
    if (const auto* d = std::get_if<double>(&v)) {
        std::ostringstream oss;
        oss << *d;
        return oss.str();
    }
    if (const auto* i = std::get_if<std::int64_t>(&v)) {
        return std::to_string(*i);
    }
    if (const auto* b = std::get_if<bool>(&v)) {
        return *b ? "true" : "false";
    }
    if (const auto* s = std::get_if<std::string>(&v)) {
        return "\"" + *s + "\"";
    }
    if (const auto* vec = std::get_if<std::vector<std::string>>(&v)) {
        std::string out = "[";
        for (std::size_t i = 0; i < vec->size(); ++i) {
            if (i > 0) out += ", ";
            out += "\"" + (*vec)[i] + "\"";
        }
        out += "]";
        return out;
    }
    return "";
}

void format_op(std::ostringstream& oss, const MIROp& op, int indent) {
    std::string pad(static_cast<std::size_t>(indent), ' ');
    oss << pad;
    if (!op.dst.empty() && op.op != "excite" && op.op != "annihilate" && op.op != "when") {
        oss << "%" << op.dst << " = ";
    }
    oss << "mir." << op.op;
    if (!op.target.empty()) {
        oss << " @" << op.target;
        if (!op.pole.empty()) {
            oss << "." << op.pole;
        }
    }
    if (!op.rhs_target.empty()) {
        oss << ", @" << op.rhs_target;
    }
    if (op.op == "frame_transport") {
        oss << " (" << format_stage(op.src_frame) << " -> " << format_stage(op.dst_frame) << ")";
    }
    if (!op.dst.empty() && (op.op == "excite" || op.op == "annihilate")) {
        oss << " on %" << op.dst;
    }
    if (!op.dst.empty() && op.op == "when") {
        oss << " as %" << op.dst;
    }
    oss << " scalar=" << op.scalar_arg;
    if (!op.scalar_gate.empty()) {
        oss << ", gate=%" << op.scalar_gate;
    }
    for (const auto& [k, v] : op.args) {
        oss << ", " << k << "=" << format_arg_value(v);
    }
    if (!op.body.empty()) {
        oss << " {\n";
        for (const auto& sub : op.body) {
            format_op(oss, sub, indent + 2);
        }
        oss << pad << "}";
    }
    if (!op.else_body.empty()) {
        oss << " else {\n";
        for (const auto& sub : op.else_body) {
            format_op(oss, sub, indent + 2);
        }
        oss << pad << "}";
    }
    oss << "\n";
}

std::string to_lower_ascii(const std::string& s) {
    std::string out = s;
    for (char& ch : out) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return out;
}

std::string read_text_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw IOError("Cannot open Strata source file: " + path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

std::optional<std::filesystem::path> resolve_import_candidate(
    const ImportDecl& imp,
    const std::filesystem::path& current_dir,
    const std::vector<std::string>& include_dirs) {
    if (imp.path.empty()) {
        return std::nullopt;
    }
    std::filesystem::path rel_exact;
    std::filesystem::path rel_lower;
    for (std::size_t i = 0; i < imp.path.size(); ++i) {
        if (i + 1 < imp.path.size()) {
            rel_exact /= imp.path[i];
            rel_lower /= to_lower_ascii(imp.path[i]);
        } else {
            rel_exact /= imp.path[i] + ".strata";
            rel_lower /= to_lower_ascii(imp.path[i]) + ".strata";
        }
    }
    std::filesystem::path rel_leaf = imp.path.back() + ".strata";
    std::filesystem::path rel_leaf_lower = to_lower_ascii(imp.path.back()) + ".strata";

    std::vector<std::filesystem::path> search_dirs;
    if (!current_dir.empty()) {
        search_dirs.push_back(current_dir);
    } else {
        search_dirs.push_back(".");
    }
    for (const auto& d : include_dirs) {
        search_dirs.emplace_back(d);
    }

    std::vector<std::filesystem::path> rels = {rel_exact, rel_lower, rel_leaf, rel_leaf_lower};
    std::error_code ec;
    for (const auto& base : search_dirs) {
        for (const auto& rel : rels) {
            std::filesystem::path cand = base / rel;
            if (std::filesystem::is_regular_file(cand, ec)) {
                return cand;
            }
        }
    }
    return std::nullopt;
}

struct ImportResolverState {
    std::vector<std::string> include_dirs;
    std::vector<std::string> active_canon;
    std::vector<std::string> active_display;
    std::set<std::string> visited_canon;
    std::set<std::string> seen_virtual_imports;
    std::vector<ImportDecl> virtual_imports;
    std::vector<TopItem> merged_decls;

    void process_file(const std::filesystem::path& file_path, bool is_entry,
                      const ProgramAST* pre_parsed = nullptr) {
        std::error_code ec;
        std::filesystem::path norm = std::filesystem::weakly_canonical(file_path, ec);
        std::string canon = ec ? file_path.lexically_normal().string() : norm.string();
        std::string display = file_path.filename().string();
        if (display.empty()) {
            display = file_path.string();
        }

        if (!is_entry && visited_canon.count(canon)) {
            return;
        }

        ProgramAST parsed_local;
        const ProgramAST* ast_ptr = pre_parsed;
        if (!ast_ptr) {
            std::string src = read_text_file(file_path.string());
            parsed_local = parse_strata(src, file_path.string());
            ast_ptr = &parsed_local;
        }
        const ProgramAST& ast = *ast_ptr;

        active_canon.push_back(canon);
        active_display.push_back(display);

        std::filesystem::path dir = file_path.parent_path();
        for (const auto& imp : ast.imports) {
            auto found = resolve_import_candidate(imp, dir, include_dirs);
            if (!found) {
                if (!imp.path.empty() && imp.path[0] == "std") {
                    std::string key = join_parts(imp.path) + "::" + imp.alias;
                    if (!seen_virtual_imports.count(key)) {
                        seen_virtual_imports.insert(key);
                        virtual_imports.push_back(imp);
                    }
                    continue;
                }
                throw UndefinedSymbolError(
                    "Cannot resolve import '" + join_parts(imp.path) + "' in " + file_path.string(),
                    imp.loc.line, imp.loc.col);
            }

            std::filesystem::path tgt_norm = std::filesystem::weakly_canonical(*found, ec);
            std::string tgt_canon = ec ? found->lexically_normal().string() : tgt_norm.string();
            std::string tgt_display = found->filename().string();

            auto active_it = std::find(active_canon.begin(), active_canon.end(), tgt_canon);
            if (active_it != active_canon.end()) {
                std::size_t idx = static_cast<std::size_t>(active_it - active_canon.begin());
                std::string chain;
                for (std::size_t i = idx; i < active_display.size(); ++i) {
                    if (i > idx) {
                        chain += " -> ";
                    }
                    chain += active_display[i];
                }
                chain += " -> " + tgt_display;
                throw ParseError("Circular import detected: " + chain, imp.loc.line, imp.loc.col);
            }

            process_file(*found, false);
        }

        active_canon.pop_back();
        active_display.pop_back();
        visited_canon.insert(canon);

        if (!is_entry) {
            for (const auto& item : ast.items) {
                if (std::holds_alternative<TestDecl>(item.node)) {
                    continue;
                }
                if (std::holds_alternative<FlowDecl>(item.node)) {
                    TopItem tagged = item;
                    tagged.annotations.push_back(Annotation{"imported_helper", {}, item.loc});
                    merged_decls.push_back(std::move(tagged));
                } else {
                    merged_decls.push_back(item);
                }
            }
        }
    }
};

}

MIRModule lower_to_mir(const TypedProgram& prog) {
    MIRLowerer lowerer(prog);
    return lowerer.run();
}

MIRModule compile_source_to_mir(std::string_view source, const std::string& filename) {
    ProgramAST ast = parse_strata(source, filename);
    TypedProgram typed = typecheck_program(ast);
    return lower_to_mir(typed);
}

MIRModule compile_file_with_imports(const std::string& entry_path,
                                    const std::vector<std::string>& include_dirs) {
    std::filesystem::path entry_p(entry_path);
    if (!std::filesystem::exists(entry_p)) {
        throw IOError("Entry Strata file does not exist: " + entry_path);
    }
    std::string entry_src = read_text_file(entry_path);
    ProgramAST entry_ast = parse_strata(entry_src, entry_path);

    ImportResolverState state;
    state.include_dirs = include_dirs;
    state.process_file(entry_p, true, &entry_ast);

    ProgramAST unified;
    unified.filename = entry_path;
    unified.module_path = entry_ast.module_path;
    unified.imports = state.virtual_imports;
    unified.items = std::move(state.merged_decls);
    for (auto& item : entry_ast.items) {
        unified.items.push_back(std::move(item));
    }

    TypedProgram typed = typecheck_program(unified);
    return lower_to_mir(typed);
}

std::string format_mir(const MIRModule& mod) {
    std::ostringstream oss;
    oss << "mir.module @" << mod.name << " {\n";
    for (const auto& imp : mod.imports) {
        oss << "  mir.import @" << imp << "\n";
    }
    for (const auto& [name, decl] : mod.declarations) {
        oss << "  mir.decl @" << name << " : " << decl.kind << " in " << format_stage(decl.frame);
        if (decl.kind == "subspace" || decl.kind == "manifold") {
            oss << " dim=" << decl.dim << " topo=" << decl.topology << " kappa=" << decl.curvature;
        } else if (decl.kind == "field") {
            oss << " spectrum=" << decl.topology << " rank=" << decl.rank << " metric=" << decl.metric;
        } else if (decl.kind == "rodata") {
            oss << " entries=" << decl.rodata_entries.size();
        }
        oss << "\n";
    }
    for (const auto& flow : mod.flows) {
        oss << "  mir.flow @" << flow.name << " in " << format_stage(flow.stage);
        if (!flow.target_mode_hint.empty()) {
            oss << " target=\"" << flow.target_mode_hint << "\"";
        }
        if (flow.is_helper) {
            oss << " helper=true";
        }
        oss << " {\n";
        oss << "    mir.contract { crosstalk <= " << flow.contract.crosstalk_bound
            << ", approx <= " << flow.contract.approx_bound
            << ", energy <= " << flow.contract.energy_bound << ", preserve = [";
        for (std::size_t i = 0; i < flow.contract.preserve.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "@" << flow.contract.preserve[i];
        }
        oss << "] }\n";
        for (const auto& op : flow.ops) {
            format_op(oss, op, 4);
        }
        oss << "  }\n";
    }
    for (const auto& tc : mod.tests) {
        oss << "  mir.test @" << tc.name << " {\n";
        for (const auto& ex : tc.expectations) {
            oss << "    mir.expect " << ex.kind;
            if (ex.kind == "resonance") {
                oss << "(\"" << ex.prompt << "\", @" << ex.symbol;
                if (!ex.pole.empty()) {
                    oss << "." << ex.pole;
                }
                oss << ")";
            } else if (ex.kind == "top_token") {
                oss << "(\"" << ex.prompt << "\") " << ex.cmp_op << " \"" << ex.target_token_str << "\"\n";
                continue;
            } else if (ex.kind == "logit_margin") {
                oss << "(\"" << ex.prompt << "\", \"" << ex.target_token_str
                    << "\", \"" << ex.foil_token_str << "\")";
            } else if (ex.kind == "kl_drift") {
                oss << "([";
                for (std::size_t i = 0; i < ex.prompts.size(); ++i) {
                    if (i > 0) oss << ", ";
                    oss << "\"" << ex.prompts[i] << "\"";
                }
                oss << "])";
            }
            oss << " " << ex.cmp_op << " ";
            if (ex.rhs_is_base) {
                oss << "base";
                if (ex.rhs_val > 0.0) {
                    oss << " + " << ex.rhs_val;
                } else if (ex.rhs_val < 0.0) {
                    oss << " - " << (-ex.rhs_val);
                }
            } else {
                oss << ex.rhs_val;
            }
            oss << "\n";
        }
        oss << "  }\n";
    }
    oss << "}\n";
    return oss.str();
}

}
