#include "aic/formatter.hpp"
#include "aic/parser.hpp"

#include <cctype>
#include <iomanip>
#include <sstream>
#include <string>

namespace aic {

namespace {

std::string format_double(double d) {
    if (d == 0.0) {
        return "0.0";
    }
    std::ostringstream oss;
    oss << std::setprecision(12) << d;
    std::string s = oss.str();
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos) {
        s += ".0";
    }
    return s;
}

std::string escape_string(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    out.push_back('"');
    return out;
}

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

bool is_ident_str(const std::string& s) {
    if (s.empty()) {
        return false;
    }
    unsigned char c0 = static_cast<unsigned char>(s[0]);
    if (std::isalpha(c0) == 0 && c0 != '_') {
        return false;
    }
    for (std::size_t i = 1; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (std::isalnum(c) == 0 && c != '_') {
            return false;
        }
    }
    return true;
}

std::string format_stage_spec(const StageSpec& s) {
    if (s.end.empty()) {
        return s.start;
    }
    return s.start + (s.inclusive ? "..=" : "..") + s.end;
}

std::string format_frame_ref(const FrameRef& fr) {
    if (fr.is_explicit_frame) {
        return "Frame<" + format_stage_spec(fr.stage) + ">";
    }
    return join_path(fr.path);
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
                out += format_double(a.float_val);
            } else if (a.type_ref) {
                out += format_type_ref(*a.type_ref);
            }
        }
        out += ">";
    }
    return out;
}

int expr_precedence(const ExprPtr& e) {
    if (!e) {
        return 9;
    }
    if (std::holds_alternative<FieldOpExpr>(e->node)) {
        return 0;
    }
    if (const auto* b = std::get_if<BinaryExpr>(&e->node)) {
        if (b->op == "||") return 1;
        if (b->op == "&&") return 2;
        if (b->op == "=>" || b->op == "~>") return 3;
        if (b->op == ".." || b->op == "..=") return 4;
        if (b->op == "<=" || b->op == ">=" || b->op == "==" ||
            b->op == "!=" || b->op == "<" || b->op == ">") {
            return 5;
        }
        if (b->op == "+" || b->op == "-") return 6;
        if (b->op == "*" || b->op == "/") return 7;
    }
    if (std::holds_alternative<UnaryExpr>(e->node)) {
        return 8;
    }
    return 9;
}

std::string format_expr(const ExprPtr& e);

std::string format_args(const std::vector<Arg>& args) {
    std::string out;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        if (!args[i].name.empty()) {
            out += args[i].name + ": ";
        }
        out += format_expr(args[i].value);
    }
    return out;
}

std::string format_expr_child(const ExprPtr& child, int parent_prec, bool force_parens_on_eq = false) {
    int cp = expr_precedence(child);
    std::string s = format_expr(child);
    if (cp < parent_prec || (force_parens_on_eq && cp == parent_prec)) {
        return "(" + s + ")";
    }
    return s;
}

std::string format_expr(const ExprPtr& e) {
    if (!e) {
        return "";
    }
    if (const auto* i = std::get_if<IntLitExpr>(&e->node)) {
        return std::to_string(i->value);
    }
    if (const auto* f = std::get_if<FloatLitExpr>(&e->node)) {
        return format_double(f->value);
    }
    if (const auto* s = std::get_if<StringLitExpr>(&e->node)) {
        return escape_string(s->value);
    }
    if (const auto* b = std::get_if<BoolLitExpr>(&e->node)) {
        return b->value ? "true" : "false";
    }
    if (const auto* p = std::get_if<PathExpr>(&e->node)) {
        return join_path(p->parts);
    }
    if (const auto* le = std::get_if<ListExpr>(&e->node)) {
        std::string out = "[";
        for (std::size_t i = 0; i < le->elements.size(); ++i) {
            if (i > 0) {
                out += ", ";
            }
            out += format_expr(le->elements[i]);
        }
        out += "]";
        return out;
    }
    if (const auto* ue = std::get_if<UnaryExpr>(&e->node)) {
        if (std::holds_alternative<UnaryExpr>(ue->operand->node)) {
            return ue->op + "(" + format_expr(ue->operand) + ")";
        }
        return ue->op + format_expr_child(ue->operand, 8);
    }
    if (const auto* be = std::get_if<BinaryExpr>(&e->node)) {
        int prec = expr_precedence(e);
        bool right_paren_on_eq = (be->op == "-" || be->op == "/" || be->op == ".." ||
                                  be->op == "..=" || be->op == "<=" || be->op == ">=" ||
                                  be->op == "==" || be->op == "!=" || be->op == "<" || be->op == ">");
        std::string sp = (be->op == ".." || be->op == "..=") ? be->op : (" " + be->op + " ");
        return format_expr_child(be->lhs, prec, false) + sp +
               format_expr_child(be->rhs, prec, right_paren_on_eq);
    }
    if (const auto* me = std::get_if<MemberExpr>(&e->node)) {
        return format_expr_child(me->object, 9) + "." + me->member;
    }
    if (const auto* ce = std::get_if<CallExpr>(&e->node)) {
        return format_expr_child(ce->callee, 9) + "(" + format_args(ce->args) + ")";
    }
    if (const auto* ae = std::get_if<AttendExpr>(&e->node)) {
        return "@attend(" + format_args(ae->args) + ")";
    }
    if (const auto* te = std::get_if<TransportExpr>(&e->node)) {
        return "transport(" + format_expr(te->expr) + ", " +
               format_frame_ref(te->src_frame) + " -> " +
               format_frame_ref(te->dst_frame) + ")";
    }
    if (const auto* foe = std::get_if<FieldOpExpr>(&e->node)) {
        std::string out = format_expr_child(foe->lhs, 0) + " " + foe->op + " " +
                          format_expr_child(foe->rhs, 0, true);
        if (!foe->args.empty()) {
            out += " with (" + format_args(foe->args) + ")";
        }
        return out;
    }
    if (const auto* fce = std::get_if<FieldCallExpr>(&e->node)) {
        return fce->op + "(" + format_args(fce->args) + ")";
    }
    return "";
}

bool decl_meta_empty(const DeclMeta& m) {
    return m.chart_name.empty() && !m.chart_expr && m.metric.empty() &&
           !m.rank && !m.curvature && m.anchors.empty();
}

void emit_indent(std::ostringstream& oss, int indent) {
    for (int i = 0; i < indent * 4; ++i) {
        oss << ' ';
    }
}

void format_anchor(std::ostringstream& oss, const AnchorDecl& a, int indent) {
    emit_indent(oss, indent);
    oss << "anchor " << a.kind << "(" << format_args(a.args) << ");\n";
}

void format_decl_meta(std::ostringstream& oss, const DeclMeta& m, int indent) {
    if (m.curvature) {
        emit_indent(oss, indent);
        oss << "curvature " << format_expr(m.curvature) << ";\n";
    }
    if (m.chart_expr) {
        emit_indent(oss, indent);
        if (!m.chart_name.empty()) {
            oss << "chart " << m.chart_name << ": " << format_expr(m.chart_expr) << ";\n";
        } else {
            oss << "chart " << format_expr(m.chart_expr) << ";\n";
        }
    }
    if (m.rank) {
        emit_indent(oss, indent);
        oss << "rank " << *m.rank << ";\n";
    }
    if (!m.metric.empty()) {
        emit_indent(oss, indent);
        if (is_ident_str(m.metric)) {
            oss << "metric " << m.metric << ";\n";
        } else {
            oss << "metric " << escape_string(m.metric) << ";\n";
        }
    }
    for (const auto& a : m.anchors) {
        format_anchor(oss, a, indent);
    }
}

void format_stmt(std::ostringstream& oss, const StmtPtr& s, int indent);

void format_when(std::ostringstream& oss, const WhenStmt& ws, int indent, bool is_else_if) {
    if (!is_else_if) {
        emit_indent(oss, indent);
    }
    oss << "when " << format_expr(ws.cond);
    if (!ws.bind_name.empty()) {
        oss << " as " << ws.bind_name;
    }
    oss << " {\n";
    for (const auto& sub : ws.body.stmts) {
        format_stmt(oss, sub, indent + 1);
    }
    emit_indent(oss, indent);
    oss << "}";
    if (ws.has_else) {
        if (ws.else_stmts.size() == 1 &&
            std::holds_alternative<WhenStmt>(ws.else_stmts[0]->node)) {
            oss << " else ";
            format_when(oss, std::get<WhenStmt>(ws.else_stmts[0]->node), indent, true);
            return;
        }
        oss << " else {\n";
        for (const auto& sub : ws.else_stmts) {
            format_stmt(oss, sub, indent + 1);
        }
        emit_indent(oss, indent);
        oss << "}";
    }
    oss << "\n";
}

void format_stmt(std::ostringstream& oss, const StmtPtr& s, int indent) {
    if (!s) {
        return;
    }
    if (const auto* ls = std::get_if<LetStmt>(&s->node)) {
        emit_indent(oss, indent);
        oss << "let " << ls->name;
        if (ls->type_ann) {
            oss << ": " << format_type_ref(*ls->type_ann);
        }
        oss << " = " << format_expr(ls->value) << ";\n";
    } else if (const auto* ws = std::get_if<WhenStmt>(&s->node)) {
        format_when(oss, *ws, indent, false);
    } else if (const auto* rs = std::get_if<RodataStmt>(&s->node)) {
        emit_indent(oss, indent);
        oss << rs->op << " " << join_path(rs->target_path) << "(" << format_args(rs->args) << ");\n";
    } else if (const auto* as = std::get_if<ActionStmt>(&s->node)) {
        emit_indent(oss, indent);
        oss << format_expr(as->lhs) << " " << as->op << " " << format_expr(as->rhs) << ";\n";
    } else if (const auto* es = std::get_if<ExprStmt>(&s->node)) {
        emit_indent(oss, indent);
        oss << format_expr(es->expr) << ";\n";
    }
}

void format_contract(std::ostringstream& oss, const EffectContractAST& ec) {
    oss << "![";
    for (std::size_t i = 0; i < ec.constraints.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        const auto& c = ec.constraints[i];
        if (c.kind == "preserve") {
            oss << "preserve(";
            for (std::size_t j = 0; j < c.preserve_paths.size(); ++j) {
                if (j > 0) {
                    oss << ", ";
                }
                oss << join_path(c.preserve_paths[j]);
            }
            oss << ")";
        } else {
            oss << c.kind << " " << c.cmp_op << " " << format_double(c.number_val);
        }
    }
    oss << "]";
}

void format_top_item(std::ostringstream& oss, const TopItem& item) {
    for (const auto& ann : item.annotations) {
        oss << "@" << ann.name;
        if (!ann.args.empty()) {
            oss << "(" << format_args(ann.args) << ")";
        }
        oss << "\n";
    }

    if (const auto* fd = std::get_if<FrameDecl>(&item.node)) {
        oss << "frame " << fd->name;
        if (fd->stage) {
            oss << " in " << format_stage_spec(*fd->stage);
        }
        if (fd->connections.empty() && fd->anchors.empty()) {
            oss << ";\n";
        } else {
            oss << " {\n";
            for (const auto& conn : fd->connections) {
                emit_indent(oss, 1);
                oss << "connect to " << conn.target;
                if (!conn.via.empty()) {
                    oss << " via " << conn.via;
                }
                oss << ";\n";
            }
            for (const auto& a : fd->anchors) {
                format_anchor(oss, a, 1);
            }
            oss << "}\n";
        }
    } else if (const auto* cd = std::get_if<ConstDecl>(&item.node)) {
        oss << "const " << cd->name;
        if (cd->type_ann) {
            oss << ": " << format_type_ref(*cd->type_ann);
        }
        oss << " = " << format_expr(cd->value) << ";\n";
    } else if (const auto* pd = std::get_if<PoleDecl>(&item.node)) {
        oss << "pole " << pd->name << " in " << format_frame_ref(pd->frame);
        if (decl_meta_empty(pd->meta)) {
            oss << ";\n";
        } else {
            oss << " {\n";
            format_decl_meta(oss, pd->meta, 1);
            oss << "}\n";
        }
    } else if (const auto* dd = std::get_if<DipoleDecl>(&item.node)) {
        oss << "dipole " << dd->name << " in " << format_frame_ref(dd->frame) << " {\n";
        for (const auto& side : dd->poles) {
            emit_indent(oss, 1);
            oss << side.polarity << ": " << side.name;
            if (side.anchors.empty()) {
                oss << ";\n";
            } else {
                oss << " {\n";
                for (const auto& a : side.anchors) {
                    format_anchor(oss, a, 2);
                }
                emit_indent(oss, 1);
                oss << "}\n";
            }
        }
        format_decl_meta(oss, dd->meta, 1);
        oss << "}\n";
    } else if (const auto* sd = std::get_if<SubspaceDecl>(&item.node)) {
        oss << "subspace " << sd->name << "<" << sd->dim << "> in " << format_frame_ref(sd->frame);
        if (decl_meta_empty(sd->meta)) {
            oss << ";\n";
        } else {
            oss << " {\n";
            format_decl_meta(oss, sd->meta, 1);
            oss << "}\n";
        }
    } else if (const auto* md = std::get_if<ManifoldDecl>(&item.node)) {
        oss << "manifold " << md->name << " : " << format_type_ref(md->topo)
            << " in " << format_frame_ref(md->frame);
        if (decl_meta_empty(md->meta)) {
            oss << ";\n";
        } else {
            oss << " {\n";
            format_decl_meta(oss, md->meta, 1);
            oss << "}\n";
        }
    } else if (const auto* fld = std::get_if<FieldDecl>(&item.node)) {
        oss << "field " << fld->name;
        if (fld->topo) {
            oss << " : " << format_type_ref(*fld->topo);
        }
        oss << " in " << format_frame_ref(fld->frame);
        if (decl_meta_empty(fld->meta)) {
            oss << ";\n";
        } else {
            oss << " {\n";
            format_decl_meta(oss, fld->meta, 1);
            oss << "}\n";
        }
    } else if (const auto* rd = std::get_if<RodataDecl>(&item.node)) {
        oss << "rodata " << rd->name << "<" << format_type_ref(rd->key_type)
            << ", " << format_type_ref(rd->val_type) << "> in " << format_frame_ref(rd->frame);
        if (rd->entries.empty() && decl_meta_empty(rd->meta)) {
            oss << ";\n";
        } else {
            oss << " {\n";
            format_decl_meta(oss, rd->meta, 1);
            for (const auto& entry : rd->entries) {
                emit_indent(oss, 1);
                oss << "entry " << format_expr(entry.key) << " => " << format_expr(entry.value);
                if (!entry.attrs.empty()) {
                    oss << " with (" << format_args(entry.attrs) << ")";
                }
                oss << ";\n";
            }
            oss << "}\n";
        }
    } else if (const auto* fl = std::get_if<FlowDecl>(&item.node)) {
        oss << "flow " << fl->name;
        if (!fl->params.empty()) {
            oss << "(";
            for (std::size_t i = 0; i < fl->params.size(); ++i) {
                if (i > 0) {
                    oss << ", ";
                }
                const auto& p = fl->params[i];
                oss << p.name << ": " << format_type_ref(p.type);
                if (p.frame) {
                    oss << " in " << format_frame_ref(*p.frame);
                }
            }
            oss << ")";
        }
        if (fl->return_type) {
            oss << " -> " << format_type_ref(*fl->return_type);
            if (fl->return_frame) {
                oss << " in " << format_frame_ref(*fl->return_frame);
            }
        }
        if (fl->has_contract) {
            oss << " ";
            format_contract(oss, fl->contract);
        }
        oss << " {\n";
        for (const auto& st : fl->body.stmts) {
            format_stmt(oss, st, 1);
        }
        oss << "}\n";
    } else if (const auto* td = std::get_if<TestDecl>(&item.node)) {
        oss << "test " << td->name << " {\n";
        for (const auto& a : td->assertions) {
            emit_indent(oss, 1);
            oss << "expect " << a.kind;
            if (a.kind == "resonance") {
                oss << "(" << format_expr(a.prompt_expr) << ", " << join_path(a.symbol_path);
                if (!a.pole_side.empty()) {
                    oss << "." << a.pole_side;
                }
                oss << ")";
            } else if (a.kind == "top_token") {
                oss << "(" << format_expr(a.prompt_expr) << ") " << a.cmp_op << " "
                    << format_expr(a.target_str_expr) << ";\n";
                continue;
            } else if (a.kind == "logit_margin") {
                oss << "(" << format_expr(a.prompt_expr) << ", "
                    << format_expr(a.target_str_expr) << ", "
                    << format_expr(a.foil_str_expr) << ")";
            } else if (a.kind == "kl_drift") {
                oss << "(" << format_expr(a.prompt_expr) << ")";
            }
            oss << " " << a.cmp_op << " ";
            if (a.rhs_is_base) {
                oss << "base";
                if (a.rhs_expr) {
                    if (const auto* ue = std::get_if<UnaryExpr>(&a.rhs_expr->node)) {
                        if (ue->op == "-") {
                            oss << " - " << format_expr_child(ue->operand, 6, true);
                        } else {
                            oss << " + " << format_expr_child(a.rhs_expr, 6, false);
                        }
                    } else {
                        oss << " + " << format_expr_child(a.rhs_expr, 6, false);
                    }
                }
            } else {
                oss << format_expr(a.rhs_expr);
            }
            oss << ";\n";
        }
        oss << "}\n";
    }
}

}

std::string format_ast(const ProgramAST& ast) {
    std::ostringstream oss;
    bool need_blank = false;

    if (!ast.module_path.empty()) {
        oss << "module " << join_path(ast.module_path) << ";\n";
        need_blank = true;
    }

    if (!ast.imports.empty()) {
        if (need_blank) {
            oss << "\n";
        }
        for (const auto& imp : ast.imports) {
            oss << "import " << join_path(imp.path);
            if (!imp.alias.empty()) {
                oss << " as " << imp.alias;
            }
            oss << ";\n";
        }
        need_blank = true;
    }

    for (std::size_t i = 0; i < ast.items.size(); ++i) {
        if (need_blank || i > 0) {
            oss << "\n";
        }
        format_top_item(oss, ast.items[i]);
        need_blank = true;
    }

    return oss.str();
}

std::string format_strata_source(std::string_view source, const std::string& filename) {
    ProgramAST ast = parse_strata(source, filename);
    return format_ast(ast);
}

}
