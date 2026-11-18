#include "aic/linter.hpp"
#include "aic/ast.hpp"
#include "aic/errors.hpp"
#include "aic/mir.hpp"
#include "aic/parser.hpp"
#include "aic/typecheck.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>

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

std::optional<double> eval_lint_number(const ExprPtr& e,
                                       const std::map<std::string, double>& consts) {
    if (!e) return std::nullopt;
    if (const auto* f = std::get_if<FloatLitExpr>(&e->node)) return f->value;
    if (const auto* i = std::get_if<IntLitExpr>(&e->node)) return static_cast<double>(i->value);
    if (const auto* p = std::get_if<PathExpr>(&e->node)) {
        auto it = consts.find(join_path(p->parts));
        if (it != consts.end()) return it->second;
        return std::nullopt;
    }
    if (const auto* u = std::get_if<UnaryExpr>(&e->node)) {
        auto sub = eval_lint_number(u->operand, consts);
        if (!sub) return std::nullopt;
        if (u->op == "-") return -(*sub);
        if (u->op == "+") return *sub;
        return std::nullopt;
    }
    if (const auto* b = std::get_if<BinaryExpr>(&e->node)) {
        auto lv = eval_lint_number(b->lhs, consts);
        auto rv = eval_lint_number(b->rhs, consts);
        if (!lv || !rv) return std::nullopt;
        if (b->op == "+") return *lv + *rv;
        if (b->op == "-") return *lv - *rv;
        if (b->op == "*") return *lv * *rv;
        if (b->op == "/" && *rv != 0.0) return *lv / *rv;
    }
    return std::nullopt;
}

void collect_type_ref_names(const TypeRef& tr, std::set<std::string>& used) {
    used.insert(tr.name);
    for (const auto& a : tr.args) {
        if (a.type_ref) {
            collect_type_ref_names(*a.type_ref, used);
        }
    }
}

void collect_frame_ref_names(const FrameRef& fr, std::set<std::string>& used) {
    if (!fr.is_explicit_frame && !fr.path.empty()) {
        used.insert(join_path(fr.path));
    }
}

void collect_expr_names(const ExprPtr& e, std::set<std::string>& used) {
    if (!e) return;
    if (const auto* pe = std::get_if<PathExpr>(&e->node)) {
        used.insert(join_path(pe->parts));
        if (!pe->parts.empty()) {
            used.insert(pe->parts[0]);
        }
    } else if (const auto* le = std::get_if<ListExpr>(&e->node)) {
        for (const auto& elem : le->elements) {
            collect_expr_names(elem, used);
        }
    } else if (const auto* ue = std::get_if<UnaryExpr>(&e->node)) {
        collect_expr_names(ue->operand, used);
    } else if (const auto* be = std::get_if<BinaryExpr>(&e->node)) {
        collect_expr_names(be->lhs, used);
        collect_expr_names(be->rhs, used);
    } else if (const auto* me = std::get_if<MemberExpr>(&e->node)) {
        collect_expr_names(me->object, used);
    } else if (const auto* ce = std::get_if<CallExpr>(&e->node)) {
        collect_expr_names(ce->callee, used);
        for (const auto& a : ce->args) {
            collect_expr_names(a.value, used);
        }
    } else if (const auto* ae = std::get_if<AttendExpr>(&e->node)) {
        for (const auto& a : ae->args) {
            collect_expr_names(a.value, used);
        }
    } else if (const auto* te = std::get_if<TransportExpr>(&e->node)) {
        collect_expr_names(te->expr, used);
        collect_frame_ref_names(te->src_frame, used);
        collect_frame_ref_names(te->dst_frame, used);
    } else if (const auto* foe = std::get_if<FieldOpExpr>(&e->node)) {
        collect_expr_names(foe->lhs, used);
        collect_expr_names(foe->rhs, used);
        for (const auto& a : foe->args) {
            collect_expr_names(a.value, used);
        }
    } else if (const auto* fce = std::get_if<FieldCallExpr>(&e->node)) {
        for (const auto& a : fce->args) {
            collect_expr_names(a.value, used);
        }
    }
}

void collect_meta_names(const DeclMeta& meta, std::set<std::string>& used) {
    collect_expr_names(meta.chart_expr, used);
    collect_expr_names(meta.curvature, used);
    for (const auto& anc : meta.anchors) {
        for (const auto& a : anc.args) {
            collect_expr_names(a.value, used);
        }
    }
}

void collect_stmt_names(const StmtPtr& s, std::set<std::string>& used) {
    if (!s) return;
    if (const auto* ls = std::get_if<LetStmt>(&s->node)) {
        if (ls->type_ann) collect_type_ref_names(*ls->type_ann, used);
        collect_expr_names(ls->value, used);
    } else if (const auto* ws = std::get_if<WhenStmt>(&s->node)) {
        collect_expr_names(ws->cond, used);
        for (const auto& st : ws->body.stmts) {
            collect_stmt_names(st, used);
        }
        for (const auto& st : ws->else_stmts) {
            collect_stmt_names(st, used);
        }
    } else if (const auto* rs = std::get_if<RodataStmt>(&s->node)) {
        used.insert(join_path(rs->target_path));
        for (const auto& a : rs->args) {
            collect_expr_names(a.value, used);
        }
    } else if (const auto* as = std::get_if<ActionStmt>(&s->node)) {
        collect_expr_names(as->lhs, used);
        collect_expr_names(as->rhs, used);
    } else if (const auto* es = std::get_if<ExprStmt>(&s->node)) {
        collect_expr_names(es->expr, used);
    }
}

std::optional<double> extract_rhs_arg_number(const ExprPtr& rhs,
                                             const std::string& arg_name,
                                             const std::map<std::string, double>& consts) {
    if (!rhs) return std::nullopt;
    if (const auto* ce = std::get_if<CallExpr>(&rhs->node)) {
        for (std::size_t i = 0; i < ce->args.size(); ++i) {
            if (ce->args[i].name == arg_name || (ce->args[i].name.empty() && i == 0)) {
                return eval_lint_number(ce->args[i].value, consts);
            }
        }
    } else if (const auto* foe = std::get_if<FieldOpExpr>(&rhs->node)) {
        for (const auto& a : foe->args) {
            if (a.name == arg_name) {
                return eval_lint_number(a.value, consts);
            }
        }
    } else if (const auto* fce = std::get_if<FieldCallExpr>(&rhs->node)) {
        for (const auto& a : fce->args) {
            if (a.name == arg_name) {
                return eval_lint_number(a.value, consts);
            }
        }
    }
    return std::nullopt;
}

void lint_cond_expr(const ExprPtr& e,
                    const std::map<std::string, double>& consts,
                    std::vector<LintDiagnostic>& diags) {
    if (!e) return;
    if (const auto* be = std::get_if<BinaryExpr>(&e->node)) {
        if (be->op == "&&" || be->op == "||") {
            lint_cond_expr(be->lhs, consts, diags);
            lint_cond_expr(be->rhs, consts, diags);
            return;
        }
        if (be->op == "~>") {
            if (auto th = extract_rhs_arg_number(be->rhs, "threshold", consts)) {
                if (*th <= 0.0 || *th > 10.0) {
                    diags.push_back(LintDiagnostic{
                        "W002",
                        "Extreme resonance threshold " + std::to_string(*th) +
                            " (expected within (0.0, 10.0])",
                        e->loc.line,
                        e->loc.col
                    });
                }
            }
        }
    }
}

void lint_stmts_block(const std::vector<StmtPtr>& stmts,
                      const std::map<std::string, double>& consts,
                      std::vector<LintDiagnostic>& diags) {
    double max_annihilate_scale = 0.0;
    for (const auto& s : stmts) {
        if (!s) continue;
        if (const auto* as = std::get_if<ActionStmt>(&s->node)) {
            if (as->op == "-|") {
                double sc = extract_rhs_arg_number(as->rhs, "scale", consts).value_or(1.0);
                max_annihilate_scale = std::max(max_annihilate_scale, sc);
            }
        }
    }

    for (const auto& s : stmts) {
        if (!s) continue;
        if (const auto* ws = std::get_if<WhenStmt>(&s->node)) {
            lint_cond_expr(ws->cond, consts, diags);
            lint_stmts_block(ws->body.stmts, consts, diags);
            if (ws->has_else) {
                lint_stmts_block(ws->else_stmts, consts, diags);
            }
        } else if (const auto* as = std::get_if<ActionStmt>(&s->node)) {
            if (as->op == "+>") {
                if (auto g = extract_rhs_arg_number(as->rhs, "gain", consts)) {
                    if (*g > 2.0 && max_annihilate_scale < 0.5 * (*g)) {
                        diags.push_back(LintDiagnostic{
                            "W003",
                            "Uncompensated high excitation gain " + std::to_string(*g) +
                                " (> 2.0) may cause RMSNorm dilution",
                            s->loc.line,
                            s->loc.col
                        });
                    }
                }
            }
        }
    }
}

std::vector<LintDiagnostic> run_lint_checks(const ProgramAST& entry_ast,
                                            const std::map<std::string, double>& consts) {
    std::vector<LintDiagnostic> diags;
    std::set<std::string> used_names;

    for (const auto& item : entry_ast.items) {
        if (const auto* fd = std::get_if<FrameDecl>(&item.node)) {
            for (const auto& c : fd->connections) {
                used_names.insert(c.target);
                if (!c.via.empty()) used_names.insert(c.via);
            }
            for (const auto& anc : fd->anchors) {
                for (const auto& a : anc.args) collect_expr_names(a.value, used_names);
            }
        } else if (const auto* cd = std::get_if<ConstDecl>(&item.node)) {
            if (cd->type_ann) collect_type_ref_names(*cd->type_ann, used_names);
            collect_expr_names(cd->value, used_names);
        } else if (const auto* pd = std::get_if<PoleDecl>(&item.node)) {
            collect_frame_ref_names(pd->frame, used_names);
            collect_meta_names(pd->meta, used_names);
        } else if (const auto* dd = std::get_if<DipoleDecl>(&item.node)) {
            collect_frame_ref_names(dd->frame, used_names);
            collect_meta_names(dd->meta, used_names);
            for (const auto& side : dd->poles) {
                for (const auto& anc : side.anchors) {
                    for (const auto& a : anc.args) collect_expr_names(a.value, used_names);
                }
            }
        } else if (const auto* sd = std::get_if<SubspaceDecl>(&item.node)) {
            collect_frame_ref_names(sd->frame, used_names);
            collect_meta_names(sd->meta, used_names);
        } else if (const auto* md = std::get_if<ManifoldDecl>(&item.node)) {
            collect_frame_ref_names(md->frame, used_names);
            collect_meta_names(md->meta, used_names);
        } else if (const auto* fld = std::get_if<FieldDecl>(&item.node)) {
            collect_frame_ref_names(fld->frame, used_names);
            collect_meta_names(fld->meta, used_names);
        } else if (const auto* rd = std::get_if<RodataDecl>(&item.node)) {
            collect_frame_ref_names(rd->frame, used_names);
            collect_meta_names(rd->meta, used_names);
            for (const auto& ent : rd->entries) {
                collect_expr_names(ent.key, used_names);
                collect_expr_names(ent.value, used_names);
                for (const auto& a : ent.attrs) collect_expr_names(a.value, used_names);
            }
        } else if (const auto* fl = std::get_if<FlowDecl>(&item.node)) {
            for (const auto& p : fl->params) {
                collect_type_ref_names(p.type, used_names);
                if (p.frame) collect_frame_ref_names(*p.frame, used_names);
            }
            if (fl->return_type) collect_type_ref_names(*fl->return_type, used_names);
            if (fl->return_frame) collect_frame_ref_names(*fl->return_frame, used_names);
            for (const auto& c : fl->contract.constraints) {
                for (const auto& p : c.preserve_paths) {
                    used_names.insert(join_path(p));
                }
            }
            for (const auto& st : fl->body.stmts) {
                collect_stmt_names(st, used_names);
            }
        } else if (const auto* td = std::get_if<TestDecl>(&item.node)) {
            for (const auto& a : td->assertions) {
                collect_expr_names(a.prompt_expr, used_names);
                collect_expr_names(a.target_str_expr, used_names);
                collect_expr_names(a.foil_str_expr, used_names);
                if (!a.symbol_path.empty()) {
                    used_names.insert(join_path(a.symbol_path));
                    used_names.insert(a.symbol_path[0]);
                }
                collect_expr_names(a.rhs_expr, used_names);
            }
        }
    }

    for (const auto& item : entry_ast.items) {
        std::string decl_name;
        bool check_unused = false;
        if (const auto* fd = std::get_if<FrameDecl>(&item.node)) {
            if (fd->connections.empty()) {
                decl_name = fd->name;
                check_unused = true;
            }
        } else if (const auto* cd = std::get_if<ConstDecl>(&item.node)) {
            decl_name = cd->name;
            check_unused = true;
        } else if (const auto* pd = std::get_if<PoleDecl>(&item.node)) {
            decl_name = pd->name;
            check_unused = true;
        } else if (const auto* dd = std::get_if<DipoleDecl>(&item.node)) {
            decl_name = dd->name;
            check_unused = true;
        } else if (const auto* sd = std::get_if<SubspaceDecl>(&item.node)) {
            decl_name = sd->name;
            check_unused = true;
        } else if (const auto* md = std::get_if<ManifoldDecl>(&item.node)) {
            decl_name = md->name;
            check_unused = true;
        } else if (const auto* fld = std::get_if<FieldDecl>(&item.node)) {
            decl_name = fld->name;
            check_unused = true;
        } else if (const auto* rd = std::get_if<RodataDecl>(&item.node)) {
            decl_name = rd->name;
            check_unused = true;
        }

        if (check_unused && !decl_name.empty() && !used_names.count(decl_name)) {
            diags.push_back(LintDiagnostic{
                "W001",
                "Unused declaration '" + decl_name + "'",
                item.loc.line,
                item.loc.col
            });
        }

        if (const auto* fl = std::get_if<FlowDecl>(&item.node)) {
            if (fl->body.stmts.empty()) {
                diags.push_back(LintDiagnostic{
                    "W004",
                    "Empty flow block '" + fl->name + "'",
                    fl->body.loc.line,
                    fl->body.loc.col
                });
            } else {
                lint_stmts_block(fl->body.stmts, consts, diags);
            }
        } else if (const auto* td = std::get_if<TestDecl>(&item.node)) {
            if (td->assertions.empty()) {
                diags.push_back(LintDiagnostic{
                    "W005",
                    "Empty test block '" + td->name + "'",
                    item.loc.line,
                    item.loc.col
                });
            }
        }
    }

    std::sort(diags.begin(), diags.end(), [](const LintDiagnostic& a, const LintDiagnostic& b) {
        if (a.line != b.line) return a.line < b.line;
        if (a.col != b.col) return a.col < b.col;
        return a.code < b.code;
    });

    return diags;
}

}

std::vector<LintDiagnostic> lint_source(std::string_view source, const std::string& filename) {
    ProgramAST ast = parse_strata(source, filename);
    TypedProgram typed = typecheck_program(ast);
    return run_lint_checks(ast, typed.const_numbers);
}

std::vector<LintDiagnostic> lint_file_with_imports(const std::string& entry_path,
                                                   const std::vector<std::string>& include_dirs) {
    (void)compile_file_with_imports(entry_path, include_dirs);
    std::ifstream ifs(entry_path);
    if (!ifs) {
        throw IOError("Cannot open Strata file: " + entry_path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    ProgramAST entry_ast = parse_strata(oss.str(), entry_path);
    std::map<std::string, double> consts;
    for (const auto& item : entry_ast.items) {
        if (const auto* cd = std::get_if<ConstDecl>(&item.node)) {
            if (auto v = eval_lint_number(cd->value, consts)) {
                consts[cd->name] = *v;
            }
        }
    }
    return run_lint_checks(entry_ast, consts);
}

}
