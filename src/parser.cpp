#include "aic/parser.hpp"
#include "aic/errors.hpp"

#include <utility>

namespace aic {

namespace {

class Parser {
public:
    Parser(const std::vector<Token>& tokens, std::string filename)
        : tokens_(tokens), filename_(std::move(filename)) {}

    ProgramAST parse_program() {
        ProgramAST prog;
        prog.filename = filename_;

        if (match(TokenKind::KwModule)) {
            prog.module_path = parse_path();
            expect(TokenKind::Semi, "Expected ';' after module declaration");
        }

        while (check(TokenKind::KwImport)) {
            prog.imports.push_back(parse_import());
        }

        while (!check(TokenKind::Eof)) {
            prog.items.push_back(parse_top_item());
        }

        return prog;
    }

private:
    const std::vector<Token>& tokens_;
    std::string filename_;
    std::size_t pos_ = 0;

    const Token& peek(std::size_t offset = 0) const {
        if (pos_ + offset >= tokens_.size()) {
            return tokens_.back();
        }
        return tokens_[pos_ + offset];
    }

    const Token& advance() {
        const Token& tok = peek();
        if (pos_ + 1 < tokens_.size()) {
            ++pos_;
        }
        return tok;
    }

    bool check(TokenKind k, std::size_t offset = 0) const {
        return peek(offset).kind == k;
    }

    bool match(TokenKind k) {
        if (check(k)) {
            advance();
            return true;
        }
        return false;
    }

    const Token& expect(TokenKind k, const std::string& msg) {
        if (!check(k)) {
            const Token& tok = peek();
            throw ParseError(msg + " (got '" + tok.text + "' [" + token_kind_name(tok.kind) + "])",
                             tok.line, tok.col);
        }
        return advance();
    }

    SourceLoc loc_of(const Token& tok) const {
        return SourceLoc{tok.line, tok.col};
    }

    std::vector<std::string> parse_path() {
        std::vector<std::string> parts;
        parts.push_back(expect(TokenKind::Ident, "Expected identifier in path").text);
        while (match(TokenKind::ColonColon)) {
            parts.push_back(expect(TokenKind::Ident, "Expected identifier after '::'").text);
        }
        return parts;
    }

    ImportDecl parse_import() {
        const Token& kw = expect(TokenKind::KwImport, "Expected 'import'");
        ImportDecl imp;
        imp.loc = loc_of(kw);
        imp.path = parse_path();
        if (match(TokenKind::KwAs)) {
            imp.alias = expect(TokenKind::Ident, "Expected alias identifier after 'as'").text;
        }
        expect(TokenKind::Semi, "Expected ';' after import declaration");
        return imp;
    }

    Annotation parse_annotation() {
        const Token& at = expect(TokenKind::At, "Expected '@'");
        Annotation ann;
        ann.loc = loc_of(at);
        ann.name = expect(TokenKind::Ident, "Expected annotation name").text;
        if (match(TokenKind::LParen)) {
            if (!check(TokenKind::RParen)) {
                ann.args = parse_arg_list();
            }
            expect(TokenKind::RParen, "Expected ')' after annotation arguments");
        }
        return ann;
    }

    TopItem parse_top_item() {
        TopItem item;
        item.loc = loc_of(peek());
        while (check(TokenKind::At)) {
            item.annotations.push_back(parse_annotation());
        }
        const Token& tok = peek();
        switch (tok.kind) {
            case TokenKind::KwFrame:
                item.node = parse_frame_decl();
                break;
            case TokenKind::KwConst:
                item.node = parse_const_decl();
                break;
            case TokenKind::KwPole:
                item.node = parse_pole_decl();
                break;
            case TokenKind::KwDipole:
                item.node = parse_dipole_decl();
                break;
            case TokenKind::KwSubspace:
                item.node = parse_subspace_decl();
                break;
            case TokenKind::KwManifold:
                item.node = parse_manifold_decl();
                break;
            case TokenKind::KwField:
                item.node = parse_field_decl();
                break;
            case TokenKind::KwRodata:
                item.node = parse_rodata_decl();
                break;
            case TokenKind::KwFlow:
                item.node = parse_flow_decl();
                break;
            case TokenKind::Ident:
                if (tok.text == "test") {
                    item.node = parse_test_decl();
                    break;
                }
                throw ParseError("Expected top-level declaration, got '" + tok.text + "'",
                                 tok.line, tok.col);
            default:
                throw ParseError("Expected top-level declaration, got '" + tok.text + "'",
                                 tok.line, tok.col);
        }
        return item;
    }

    std::string parse_stage_bound() {
        const Token& tok = peek();
        if (tok.kind == TokenKind::Ident || tok.kind == TokenKind::Int) {
            advance();
            return tok.text;
        }
        throw ParseError("Expected stage bound identifier or integer", tok.line, tok.col);
    }

    StageSpec parse_stage_spec() {
        StageSpec spec;
        spec.loc = loc_of(peek());
        spec.start = parse_stage_bound();
        if (match(TokenKind::DotDotEq)) {
            spec.inclusive = true;
            spec.end = parse_stage_bound();
        } else if (match(TokenKind::DotDot)) {
            spec.inclusive = false;
            spec.end = parse_stage_bound();
        }
        return spec;
    }

    FrameRef parse_frame_ref() {
        FrameRef ref;
        ref.loc = loc_of(peek());
        if (match(TokenKind::KwFrameType)) {
            ref.is_explicit_frame = true;
            expect(TokenKind::Lt, "Expected '<' after 'Frame'");
            ref.stage = parse_stage_spec();
            expect(TokenKind::Gt, "Expected '>' after stage spec");
            return ref;
        }
        ref.is_explicit_frame = false;
        ref.path = parse_path();
        return ref;
    }

    AnchorDecl parse_anchor_block() {
        const Token& kw = expect(TokenKind::KwAnchor, "Expected 'anchor'");
        AnchorDecl anchor;
        anchor.loc = loc_of(kw);
        anchor.kind = expect(TokenKind::Ident, "Expected anchor kind identifier").text;
        expect(TokenKind::LParen, "Expected '(' after anchor kind");
        if (!check(TokenKind::RParen)) {
            anchor.args = parse_arg_list();
        }
        expect(TokenKind::RParen, "Expected ')' after anchor arguments");
        expect(TokenKind::Semi, "Expected ';' after anchor declaration");
        return anchor;
    }

    ConstDecl parse_const_decl() {
        expect(TokenKind::KwConst, "Expected 'const'");
        ConstDecl decl;
        decl.name = expect(TokenKind::Ident, "Expected constant name").text;
        if (match(TokenKind::Colon)) {
            decl.type_ann = parse_type_ref();
        }
        expect(TokenKind::Eq, "Expected '=' in const declaration");
        decl.value = parse_expr();
        expect(TokenKind::Semi, "Expected ';' after const declaration");
        return decl;
    }

    FrameDecl parse_frame_decl() {
        expect(TokenKind::KwFrame, "Expected 'frame'");
        FrameDecl decl;
        decl.name = expect(TokenKind::Ident, "Expected frame name").text;
        if (match(TokenKind::KwIn)) {
            decl.stage = parse_stage_spec();
        }
        if (match(TokenKind::Semi)) {
            return decl;
        }
        expect(TokenKind::LBrace, "Expected '{' or ';' in frame declaration");
        while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
            if (check(TokenKind::KwConnect)) {
                const Token& ctok = advance();
                expect(TokenKind::KwTo, "Expected 'to' after 'connect'");
                FrameConnection conn;
                conn.loc = loc_of(ctok);
                conn.target = expect(TokenKind::Ident, "Expected target frame name").text;
                if (match(TokenKind::KwVia)) {
                    conn.via = expect(TokenKind::Ident, "Expected bridge name after 'via'").text;
                }
                expect(TokenKind::Semi, "Expected ';' after frame connection");
                decl.connections.push_back(conn);
            } else if (check(TokenKind::KwAnchor)) {
                decl.anchors.push_back(parse_anchor_block());
            } else {
                const Token& tok = peek();
                throw ParseError("Expected 'connect' or 'anchor' in frame body", tok.line, tok.col);
            }
        }
        expect(TokenKind::RBrace, "Expected '}' after frame body");
        return decl;
    }

    bool parse_decl_member(DeclMeta& meta) {
        if (check(TokenKind::KwAnchor)) {
            meta.anchors.push_back(parse_anchor_block());
            return true;
        }
        if (check(TokenKind::Ident)) {
            const std::string& kw = peek().text;
            if (kw == "chart") {
                advance();
                if (check(TokenKind::Ident, 0) && check(TokenKind::Colon, 1)) {
                    meta.chart_name = advance().text;
                    advance();
                } else {
                    meta.chart_name.clear();
                }
                meta.chart_expr = parse_expr();
                expect(TokenKind::Semi, "Expected ';' after chart specification");
                return true;
            }
            if (kw == "metric") {
                advance();
                if (check(TokenKind::Ident) || check(TokenKind::String)) {
                    meta.metric = advance().text;
                } else {
                    const Token& tok = peek();
                    throw ParseError("Expected metric identifier", tok.line, tok.col);
                }
                expect(TokenKind::Semi, "Expected ';' after metric specification");
                return true;
            }
            if (kw == "rank") {
                advance();
                const Token& rtok = expect(TokenKind::Int, "Expected integer rank");
                meta.rank = static_cast<int>(rtok.int_val);
                expect(TokenKind::Semi, "Expected ';' after rank specification");
                return true;
            }
            if (kw == "curvature") {
                advance();
                meta.curvature = parse_expr();
                expect(TokenKind::Semi, "Expected ';' after curvature specification");
                return true;
            }
        }
        return false;
    }

    DeclMeta parse_decl_body() {
        DeclMeta meta;
        expect(TokenKind::LBrace, "Expected '{'");
        while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
            if (!parse_decl_member(meta)) {
                const Token& tok = peek();
                throw ParseError("Unexpected token in declaration body: '" + tok.text + "'",
                                 tok.line, tok.col);
            }
        }
        expect(TokenKind::RBrace, "Expected '}'");
        return meta;
    }

    PoleDecl parse_pole_decl() {
        expect(TokenKind::KwPole, "Expected 'pole'");
        PoleDecl decl;
        decl.name = expect(TokenKind::Ident, "Expected pole name").text;
        expect(TokenKind::KwIn, "Expected 'in' after pole name");
        decl.frame = parse_frame_ref();
        if (!match(TokenKind::Semi)) {
            decl.meta = parse_decl_body();
        }
        return decl;
    }

    DipoleDecl parse_dipole_decl() {
        expect(TokenKind::KwDipole, "Expected 'dipole'");
        DipoleDecl decl;
        decl.name = expect(TokenKind::Ident, "Expected dipole name").text;
        expect(TokenKind::KwIn, "Expected 'in' after dipole name");
        decl.frame = parse_frame_ref();
        expect(TokenKind::LBrace, "Expected '{' in dipole declaration");
        while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
            if (check(TokenKind::Plus) || check(TokenKind::Minus)) {
                const Token& sign = advance();
                expect(TokenKind::KwPole, "Expected 'pole' after '+' or '-'");
                expect(TokenKind::Colon, "Expected ':' after 'pole'");
                PoleSide side;
                side.loc = loc_of(sign);
                side.polarity = (sign.kind == TokenKind::Plus) ? "+pole" : "-pole";
                side.name = expect(TokenKind::Ident, "Expected pole side name").text;
                if (!match(TokenKind::Semi)) {
                    expect(TokenKind::LBrace, "Expected '{' or ';' after pole side name");
                    while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
                        side.anchors.push_back(parse_anchor_block());
                    }
                    expect(TokenKind::RBrace, "Expected '}' after pole side body");
                }
                decl.poles.push_back(std::move(side));
            } else if (!parse_decl_member(decl.meta)) {
                const Token& tok = peek();
                throw ParseError("Expected dipole member, got '" + tok.text + "'",
                                 tok.line, tok.col);
            }
        }
        expect(TokenKind::RBrace, "Expected '}' after dipole declaration");
        return decl;
    }

    SubspaceDecl parse_subspace_decl() {
        expect(TokenKind::KwSubspace, "Expected 'subspace'");
        SubspaceDecl decl;
        decl.name = expect(TokenKind::Ident, "Expected subspace name").text;
        expect(TokenKind::Lt, "Expected '<' after subspace name");
        const Token& dim_tok = expect(TokenKind::Int, "Expected integer dimension");
        decl.dim = static_cast<int>(dim_tok.int_val);
        expect(TokenKind::Gt, "Expected '>' after subspace dimension");
        expect(TokenKind::KwIn, "Expected 'in' after subspace dimension");
        decl.frame = parse_frame_ref();
        if (!match(TokenKind::Semi)) {
            decl.meta = parse_decl_body();
        }
        return decl;
    }

    TypeRef parse_topo_spec() {
        TypeRef topo;
        const Token& id = expect(TokenKind::Ident, "Expected topology identifier");
        topo.loc = loc_of(id);
        topo.name = id.text;
        if (match(TokenKind::Lt)) {
            do {
                TypeArg targ;
                if (check(TokenKind::Int)) {
                    targ.kind = TypeArg::Kind::Int;
                    targ.int_val = advance().int_val;
                } else if (check(TokenKind::Float)) {
                    targ.kind = TypeArg::Kind::Float;
                    targ.float_val = advance().float_val;
                } else {
                    const Token& tid = expect(TokenKind::Ident, "Expected type argument");
                    auto tr = std::make_shared<TypeRef>();
                    tr->loc = loc_of(tid);
                    tr->name = tid.text;
                    targ.kind = TypeArg::Kind::Type;
                    targ.type_ref = std::move(tr);
                }
                topo.args.push_back(std::move(targ));
            } while (match(TokenKind::Comma));
            expect(TokenKind::Gt, "Expected '>' after topology arguments");
        }
        return topo;
    }

    TypeRef parse_type_ref() {
        TypeRef tr;
        const Token& id = expect(TokenKind::Ident, "Expected type identifier");
        tr.loc = loc_of(id);
        tr.name = id.text;
        if (match(TokenKind::Lt)) {
            do {
                TypeArg targ;
                if (check(TokenKind::Int)) {
                    targ.kind = TypeArg::Kind::Int;
                    targ.int_val = advance().int_val;
                } else if (check(TokenKind::Float)) {
                    targ.kind = TypeArg::Kind::Float;
                    targ.float_val = advance().float_val;
                } else {
                    targ.kind = TypeArg::Kind::Type;
                    targ.type_ref = std::make_shared<TypeRef>(parse_type_ref());
                }
                tr.args.push_back(std::move(targ));
            } while (match(TokenKind::Comma));
            expect(TokenKind::Gt, "Expected '>' after type arguments");
        }
        return tr;
    }

    ManifoldDecl parse_manifold_decl() {
        expect(TokenKind::KwManifold, "Expected 'manifold'");
        ManifoldDecl decl;
        decl.name = expect(TokenKind::Ident, "Expected manifold name").text;
        expect(TokenKind::Colon, "Expected ':' after manifold name");
        decl.topo = parse_topo_spec();
        expect(TokenKind::KwIn, "Expected 'in' after manifold topology");
        decl.frame = parse_frame_ref();
        if (!match(TokenKind::Semi)) {
            decl.meta = parse_decl_body();
        }
        return decl;
    }

    FieldDecl parse_field_decl() {
        expect(TokenKind::KwField, "Expected 'field'");
        FieldDecl decl;
        decl.name = expect(TokenKind::Ident, "Expected field name").text;
        if (match(TokenKind::Colon)) {
            decl.topo = parse_topo_spec();
        }
        expect(TokenKind::KwIn, "Expected 'in' after field name");
        decl.frame = parse_frame_ref();
        if (!match(TokenKind::Semi)) {
            decl.meta = parse_decl_body();
        }
        return decl;
    }

    RodataDecl parse_rodata_decl() {
        expect(TokenKind::KwRodata, "Expected 'rodata'");
        RodataDecl decl;
        decl.name = expect(TokenKind::Ident, "Expected rodata name").text;
        expect(TokenKind::Lt, "Expected '<' after rodata name");
        decl.key_type = parse_type_ref();
        expect(TokenKind::Comma, "Expected ',' between rodata key and value types");
        decl.val_type = parse_type_ref();
        expect(TokenKind::Gt, "Expected '>' after rodata type arguments");
        expect(TokenKind::KwIn, "Expected 'in' after rodata type");
        decl.frame = parse_frame_ref();
        if (match(TokenKind::Semi)) {
            return decl;
        }
        expect(TokenKind::LBrace, "Expected '{' or ';' in rodata declaration");
        while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
            if (check(TokenKind::KwEntry)) {
                const Token& etok = advance();
                RodataEntryAST entry;
                entry.loc = loc_of(etok);
                entry.key = parse_cmp_expr();
                expect(TokenKind::MapArrow, "Expected '=>' in rodata entry");
                entry.value = parse_cmp_expr();
                if (match(TokenKind::KwWith)) {
                    expect(TokenKind::LParen, "Expected '(' after 'with'");
                    if (!check(TokenKind::RParen)) {
                        entry.attrs = parse_arg_list();
                    }
                    expect(TokenKind::RParen, "Expected ')' after entry attributes");
                }
                expect(TokenKind::Semi, "Expected ';' after rodata entry");
                decl.entries.push_back(std::move(entry));
            } else if (!parse_decl_member(decl.meta)) {
                const Token& tok = peek();
                throw ParseError("Expected rodata item, got '" + tok.text + "'",
                                 tok.line, tok.col);
            }
        }
        expect(TokenKind::RBrace, "Expected '}' after rodata body");
        return decl;
    }

    ParamDecl parse_param() {
        ParamDecl p;
        const Token& id = expect(TokenKind::Ident, "Expected parameter name");
        p.loc = loc_of(id);
        p.name = id.text;
        expect(TokenKind::Colon, "Expected ':' after parameter name");
        p.type = parse_type_ref();
        if (match(TokenKind::KwIn)) {
            p.frame = parse_frame_ref();
        }
        return p;
    }

    double parse_number_lit() {
        bool neg = false;
        if (match(TokenKind::Minus)) {
            neg = true;
        }
        double val = 0.0;
        if (check(TokenKind::Float)) {
            val = advance().float_val;
        } else if (check(TokenKind::Int)) {
            val = static_cast<double>(advance().int_val);
        } else {
            const Token& tok = peek();
            throw ParseError("Expected numeric literal in effect contract", tok.line, tok.col);
        }
        return neg ? -val : val;
    }

    std::string parse_cmp_op() {
        const Token& tok = peek();
        switch (tok.kind) {
            case TokenKind::Le:
            case TokenKind::Ge:
            case TokenKind::EqEq:
            case TokenKind::Lt:
            case TokenKind::Gt:
                advance();
                return tok.text;
            default:
                throw ParseError("Expected comparison operator in constraint", tok.line, tok.col);
        }
    }

    EffectContractAST parse_effect_contract() {
        const Token& open = expect(TokenKind::EffectOpen, "Expected '!['");
        EffectContractAST contract;
        contract.loc = loc_of(open);
        do {
            ConstraintAST c;
            const Token& ctok = peek();
            c.loc = loc_of(ctok);
            if (match(TokenKind::KwCrosstalk)) {
                c.kind = "crosstalk";
                c.cmp_op = parse_cmp_op();
                c.number_val = parse_number_lit();
            } else if (match(TokenKind::KwApprox)) {
                c.kind = "approx";
                c.cmp_op = parse_cmp_op();
                c.number_val = parse_number_lit();
            } else if (match(TokenKind::KwEnergy)) {
                c.kind = "energy";
                c.cmp_op = parse_cmp_op();
                c.number_val = parse_number_lit();
            } else if (match(TokenKind::KwPreserve)) {
                c.kind = "preserve";
                expect(TokenKind::LParen, "Expected '(' after 'preserve'");
                if (!check(TokenKind::RParen)) {
                    do {
                        c.preserve_paths.push_back(parse_path());
                    } while (match(TokenKind::Comma));
                }
                expect(TokenKind::RParen, "Expected ')' after preserve list");
            } else {
                throw ParseError("Expected effect constraint, got '" + ctok.text + "'",
                                 ctok.line, ctok.col);
            }
            contract.constraints.push_back(std::move(c));
        } while (match(TokenKind::Comma));
        expect(TokenKind::RBracket, "Expected ']' closing effect contract");
        return contract;
    }

    FlowDecl parse_flow_decl() {
        expect(TokenKind::KwFlow, "Expected 'flow'");
        FlowDecl decl;
        decl.name = expect(TokenKind::Ident, "Expected flow name").text;
        if (match(TokenKind::LParen)) {
            if (!check(TokenKind::RParen)) {
                do {
                    decl.params.push_back(parse_param());
                } while (match(TokenKind::Comma));
            }
            expect(TokenKind::RParen, "Expected ')' after flow parameters");
        }
        if (match(TokenKind::Arrow)) {
            decl.return_type = parse_type_ref();
            if (match(TokenKind::KwIn)) {
                decl.return_frame = parse_frame_ref();
            }
        }
        if (check(TokenKind::EffectOpen)) {
            decl.contract = parse_effect_contract();
            decl.has_contract = true;
        }
        decl.body = parse_block();
        return decl;
    }

    std::string parse_expect_cmp_op() {
        const Token& tok = peek();
        switch (tok.kind) {
            case TokenKind::Lt:
            case TokenKind::Le:
            case TokenKind::Gt:
            case TokenKind::Ge:
            case TokenKind::EqEq:
            case TokenKind::Ne:
                advance();
                return tok.text;
            default:
                throw ParseError("Expected comparison operator in expect assertion", tok.line, tok.col);
        }
    }

    void parse_expect_rhs(ExpectAssertionAST& a) {
        if (check(TokenKind::Ident) && peek().text == "base") {
            advance();
            a.rhs_is_base = true;
            if (match(TokenKind::Plus)) {
                a.rhs_expr = parse_expr();
            } else if (check(TokenKind::Minus)) {
                SourceLoc mloc = loc_of(advance());
                ExprPtr sub = parse_expr();
                a.rhs_expr = make_expr(mloc, UnaryExpr{"-", std::move(sub)});
            } else {
                a.rhs_expr = nullptr;
            }
            return;
        }
        a.rhs_is_base = false;
        a.rhs_expr = parse_expr();
    }

    ExpectAssertionAST parse_expect_stmt() {
        const Token& ekw = expect(TokenKind::Ident, "Expected 'expect'");
        if (ekw.text != "expect") {
            throw ParseError("Expected 'expect' statement in test block, got '" + ekw.text + "'",
                             ekw.line, ekw.col);
        }
        ExpectAssertionAST a;
        a.loc = loc_of(ekw);

        if (match(TokenKind::KwCrosstalk)) {
            a.kind = "crosstalk";
            a.cmp_op = parse_expect_cmp_op();
            parse_expect_rhs(a);
            expect(TokenKind::Semi, "Expected ';' after expect assertion");
            return a;
        }
        if (match(TokenKind::KwApprox)) {
            a.kind = "approx";
            a.cmp_op = parse_expect_cmp_op();
            parse_expect_rhs(a);
            expect(TokenKind::Semi, "Expected ';' after expect assertion");
            return a;
        }
        const Token& ktok = expect(TokenKind::Ident, "Expected expectation kind");
        if (ktok.text == "resonance") {
            a.kind = "resonance";
            expect(TokenKind::LParen, "Expected '(' after 'resonance'");
            a.prompt_expr = parse_expr();
            expect(TokenKind::Comma, "Expected ',' after resonance prompt");
            a.symbol_path = parse_path();
            if (match(TokenKind::Dot)) {
                a.pole_side = parse_member_name();
            }
            expect(TokenKind::RParen, "Expected ')' after resonance target");
            a.cmp_op = parse_expect_cmp_op();
            parse_expect_rhs(a);
            expect(TokenKind::Semi, "Expected ';' after expect assertion");
            return a;
        }
        if (ktok.text == "kl_drift") {
            a.kind = "kl_drift";
            expect(TokenKind::LParen, "Expected '(' after 'kl_drift'");
            a.prompt_expr = parse_expr();
            expect(TokenKind::RParen, "Expected ')' after kl_drift prompt");
            a.cmp_op = parse_expect_cmp_op();
            parse_expect_rhs(a);
            expect(TokenKind::Semi, "Expected ';' after expect assertion");
            return a;
        }
        if (ktok.text == "top_token") {
            a.kind = "top_token";
            expect(TokenKind::LParen, "Expected '(' after 'top_token'");
            a.prompt_expr = parse_expr();
            expect(TokenKind::RParen, "Expected ')' after top_token prompt");
            a.cmp_op = parse_expect_cmp_op();
            a.target_str_expr = parse_expr();
            expect(TokenKind::Semi, "Expected ';' after expect assertion");
            return a;
        }
        if (ktok.text == "logit_margin") {
            a.kind = "logit_margin";
            expect(TokenKind::LParen, "Expected '(' after 'logit_margin'");
            a.prompt_expr = parse_expr();
            expect(TokenKind::Comma, "Expected ',' after logit_margin prompt");
            a.target_str_expr = parse_expr();
            expect(TokenKind::Comma, "Expected ',' after logit_margin target token");
            a.foil_str_expr = parse_expr();
            expect(TokenKind::RParen, "Expected ')' after logit_margin foil token");
            a.cmp_op = parse_expect_cmp_op();
            parse_expect_rhs(a);
            expect(TokenKind::Semi, "Expected ';' after expect assertion");
            return a;
        }
        throw ParseError("Unknown expect assertion kind '" + ktok.text + "'", ktok.line, ktok.col);
    }

    TestDecl parse_test_decl() {
        const Token& tkw = expect(TokenKind::Ident, "Expected 'test'");
        if (tkw.text != "test") {
            throw ParseError("Expected 'test'", tkw.line, tkw.col);
        }
        TestDecl decl;
        decl.name = expect(TokenKind::Ident, "Expected test block name").text;
        expect(TokenKind::LBrace, "Expected '{' after test name");
        while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
            decl.assertions.push_back(parse_expect_stmt());
        }
        expect(TokenKind::RBrace, "Expected '}' after test body");
        return decl;
    }

    Block parse_block() {
        const Token& lb = expect(TokenKind::LBrace, "Expected '{'");
        Block blk;
        blk.loc = loc_of(lb);
        while (!check(TokenKind::RBrace) && !check(TokenKind::Eof)) {
            blk.stmts.push_back(parse_stmt());
        }
        expect(TokenKind::RBrace, "Expected '}'");
        return blk;
    }

    StmtPtr parse_stmt() {
        if (check(TokenKind::KwLet)) {
            return parse_let_stmt();
        }
        if (check(TokenKind::KwWhen)) {
            return parse_when_stmt();
        }
        if (check(TokenKind::KwInsert) || check(TokenKind::KwPrune) || check(TokenKind::KwRewrite)) {
            return parse_rodata_stmt();
        }
        return parse_action_or_expr_stmt();
    }

    StmtPtr parse_let_stmt() {
        const Token& kw = expect(TokenKind::KwLet, "Expected 'let'");
        auto s = std::make_shared<Stmt>();
        s->loc = loc_of(kw);
        LetStmt ls;
        ls.name = expect(TokenKind::Ident, "Expected variable name after 'let'").text;
        if (match(TokenKind::Colon)) {
            ls.type_ann = parse_type_ref();
        }
        expect(TokenKind::Eq, "Expected '=' in let binding");
        ls.value = parse_expr();
        expect(TokenKind::Semi, "Expected ';' after let statement");
        s->node = std::move(ls);
        return s;
    }

    StmtPtr parse_when_stmt() {
        const Token& kw = expect(TokenKind::KwWhen, "Expected 'when'");
        auto s = std::make_shared<Stmt>();
        s->loc = loc_of(kw);
        WhenStmt ws;
        ws.cond = parse_expr();
        if (match(TokenKind::KwAs)) {
            ws.bind_name = expect(TokenKind::Ident, "Expected binding name after 'as'").text;
        }
        ws.body = parse_block();
        if (match(TokenKind::KwElse)) {
            ws.has_else = true;
            if (check(TokenKind::KwWhen)) {
                ws.else_stmts.push_back(parse_when_stmt());
            } else {
                Block eb = parse_block();
                ws.else_stmts = std::move(eb.stmts);
            }
        }
        s->node = std::move(ws);
        return s;
    }

    StmtPtr parse_rodata_stmt() {
        const Token& kw = advance();
        auto s = std::make_shared<Stmt>();
        s->loc = loc_of(kw);
        RodataStmt rs;
        rs.op = kw.text;
        rs.target_path = parse_path();
        expect(TokenKind::LParen, "Expected '(' after rodata target");
        if (!check(TokenKind::RParen)) {
            rs.args = parse_arg_list();
        }
        expect(TokenKind::RParen, "Expected ')' after rodata statement arguments");
        expect(TokenKind::Semi, "Expected ';' after rodata statement");
        s->node = std::move(rs);
        return s;
    }

    StmtPtr parse_action_or_expr_stmt() {
        SourceLoc loc = loc_of(peek());
        ExprPtr lhs = parse_expr();
        auto s = std::make_shared<Stmt>();
        s->loc = loc;
        if (check(TokenKind::Excite) || check(TokenKind::Annihilate)) {
            std::string op = advance().text;
            ExprPtr rhs = parse_expr();
            expect(TokenKind::Semi, "Expected ';' after action statement");
            s->node = ActionStmt{op, std::move(lhs), std::move(rhs)};
            return s;
        }
        expect(TokenKind::Semi, "Expected ';' after expression statement");
        s->node = ExprStmt{std::move(lhs)};
        return s;
    }

    std::vector<Arg> parse_arg_list() {
        std::vector<Arg> args;
        do {
            args.push_back(parse_arg());
        } while (match(TokenKind::Comma));
        return args;
    }

    Arg parse_arg() {
        Arg a;
        a.loc = loc_of(peek());
        if (check(TokenKind::Ident, 0) && check(TokenKind::Colon, 1)) {
            a.name = advance().text;
            advance();
        }
        a.value = parse_expr();
        return a;
    }

    ExprPtr make_expr(SourceLoc loc, decltype(Expr::node) node) {
        auto e = std::make_shared<Expr>();
        e->loc = loc;
        e->node = std::move(node);
        return e;
    }

    ExprPtr parse_expr() {
        return parse_field_expr();
    }

    ExprPtr parse_field_expr() {
        ExprPtr lhs = parse_logic_or_expr();
        while (check(TokenKind::KwDecouple) || check(TokenKind::KwTemper) || check(TokenKind::KwBlend)) {
            const Token& op_tok = advance();
            ExprPtr rhs = parse_logic_or_expr();
            std::vector<Arg> args;
            if (match(TokenKind::KwWith)) {
                expect(TokenKind::LParen, "Expected '(' after 'with'");
                if (!check(TokenKind::RParen)) {
                    args = parse_arg_list();
                }
                expect(TokenKind::RParen, "Expected ')' after field operator arguments");
            }
            lhs = make_expr(loc_of(op_tok), FieldOpExpr{op_tok.text, lhs, rhs, std::move(args)});
        }
        return lhs;
    }

    ExprPtr parse_logic_or_expr() {
        ExprPtr lhs = parse_logic_and_expr();
        while (check(TokenKind::OrOr)) {
            const Token& op_tok = advance();
            ExprPtr rhs = parse_logic_and_expr();
            lhs = make_expr(loc_of(op_tok), BinaryExpr{"||", lhs, rhs});
        }
        return lhs;
    }

    ExprPtr parse_logic_and_expr() {
        ExprPtr lhs = parse_flow_op_expr();
        while (check(TokenKind::AndAnd)) {
            const Token& op_tok = advance();
            ExprPtr rhs = parse_flow_op_expr();
            lhs = make_expr(loc_of(op_tok), BinaryExpr{"&&", lhs, rhs});
        }
        return lhs;
    }

    ExprPtr parse_flow_op_expr() {
        ExprPtr lhs = parse_range_expr();
        while (check(TokenKind::MapArrow) || check(TokenKind::Resonance)) {
            const Token& op_tok = advance();
            ExprPtr rhs = parse_range_expr();
            lhs = make_expr(loc_of(op_tok), BinaryExpr{op_tok.text, lhs, rhs});
        }
        return lhs;
    }

    ExprPtr parse_range_expr() {
        ExprPtr lhs = parse_cmp_expr();
        if (check(TokenKind::DotDotEq) || check(TokenKind::DotDot)) {
            const Token& op_tok = advance();
            ExprPtr rhs = parse_cmp_expr();
            lhs = make_expr(loc_of(op_tok), BinaryExpr{op_tok.text, lhs, rhs});
        }
        return lhs;
    }

    ExprPtr parse_cmp_expr() {
        ExprPtr lhs = parse_add_expr();
        if (check(TokenKind::Le) || check(TokenKind::Ge) || check(TokenKind::EqEq) ||
            check(TokenKind::Ne) || check(TokenKind::Lt) || check(TokenKind::Gt)) {
            const Token& op_tok = advance();
            ExprPtr rhs = parse_add_expr();
            lhs = make_expr(loc_of(op_tok), BinaryExpr{op_tok.text, lhs, rhs});
        }
        return lhs;
    }

    ExprPtr parse_add_expr() {
        ExprPtr lhs = parse_mul_expr();
        while (check(TokenKind::Plus) || check(TokenKind::Minus)) {
            const Token& op_tok = advance();
            ExprPtr rhs = parse_mul_expr();
            lhs = make_expr(loc_of(op_tok), BinaryExpr{op_tok.text, lhs, rhs});
        }
        return lhs;
    }

    ExprPtr parse_mul_expr() {
        ExprPtr lhs = parse_unary_expr();
        while (check(TokenKind::Star) || check(TokenKind::Slash)) {
            const Token& op_tok = advance();
            ExprPtr rhs = parse_unary_expr();
            lhs = make_expr(loc_of(op_tok), BinaryExpr{op_tok.text, lhs, rhs});
        }
        return lhs;
    }

    ExprPtr parse_unary_expr() {
        if (check(TokenKind::Minus) || check(TokenKind::Plus) || check(TokenKind::Bang)) {
            const Token& op_tok = advance();
            ExprPtr operand = parse_unary_expr();
            return make_expr(loc_of(op_tok), UnaryExpr{op_tok.text, std::move(operand)});
        }
        return parse_postfix_expr();
    }

    std::string parse_member_name() {
        if (check(TokenKind::Plus) || check(TokenKind::Minus)) {
            std::string sign = advance().text;
            expect(TokenKind::KwPole, "Expected 'pole' after '+' or '-' in member access");
            return sign + "pole";
        }
        return expect(TokenKind::Ident, "Expected member identifier after '.'").text;
    }

    ExprPtr parse_postfix_expr() {
        ExprPtr expr = parse_primary_expr();
        while (true) {
            if (match(TokenKind::Dot)) {
                SourceLoc loc = loc_of(peek());
                std::string member = parse_member_name();
                expr = make_expr(loc, MemberExpr{expr, member});
                if (match(TokenKind::LParen)) {
                    std::vector<Arg> args;
                    if (!check(TokenKind::RParen)) {
                        args = parse_arg_list();
                    }
                    expect(TokenKind::RParen, "Expected ')' after method arguments");
                    expr = make_expr(loc, CallExpr{expr, std::move(args)});
                }
            } else if (match(TokenKind::LParen)) {
                SourceLoc loc = expr->loc;
                std::vector<Arg> args;
                if (!check(TokenKind::RParen)) {
                    args = parse_arg_list();
                }
                expect(TokenKind::RParen, "Expected ')' after call arguments");
                expr = make_expr(loc, CallExpr{expr, std::move(args)});
            } else {
                break;
            }
        }
        return expr;
    }

    ExprPtr parse_primary_expr() {
        const Token& tok = peek();
        SourceLoc loc = loc_of(tok);

        if (match(TokenKind::Float)) {
            return make_expr(loc, FloatLitExpr{tok.float_val});
        }
        if (match(TokenKind::Int)) {
            return make_expr(loc, IntLitExpr{tok.int_val});
        }
        if (match(TokenKind::String)) {
            return make_expr(loc, StringLitExpr{tok.text});
        }
        if (match(TokenKind::KwTrue)) {
            return make_expr(loc, BoolLitExpr{true});
        }
        if (match(TokenKind::KwFalse)) {
            return make_expr(loc, BoolLitExpr{false});
        }
        if (match(TokenKind::Attend)) {
            expect(TokenKind::LParen, "Expected '(' after @attend");
            std::vector<Arg> args;
            if (!check(TokenKind::RParen)) {
                args = parse_arg_list();
            }
            expect(TokenKind::RParen, "Expected ')' after @attend arguments");
            return make_expr(loc, AttendExpr{std::move(args)});
        }
        if (match(TokenKind::KwTransport)) {
            expect(TokenKind::LParen, "Expected '(' after 'transport'");
            ExprPtr inner = parse_expr();
            expect(TokenKind::Comma, "Expected ',' in transport(expr, src -> dst)");
            FrameRef src_f = parse_frame_ref();
            expect(TokenKind::Arrow, "Expected '->' in transport(expr, src -> dst)");
            FrameRef dst_f = parse_frame_ref();
            expect(TokenKind::RParen, "Expected ')' after transport call");
            return make_expr(loc, TransportExpr{std::move(inner), std::move(src_f), std::move(dst_f)});
        }
        if (check(TokenKind::KwDecouple) || check(TokenKind::KwTemper) || check(TokenKind::KwBlend)) {
            std::string op = advance().text;
            expect(TokenKind::LParen, "Expected '(' after field operator '" + op + "'");
            std::vector<Arg> args;
            if (!check(TokenKind::RParen)) {
                args = parse_arg_list();
            }
            expect(TokenKind::RParen, "Expected ')' after field operator arguments");
            return make_expr(loc, FieldCallExpr{op, std::move(args)});
        }
        if (match(TokenKind::LBracket)) {
            std::vector<ExprPtr> elems;
            if (!check(TokenKind::RBracket)) {
                do {
                    elems.push_back(parse_expr());
                } while (match(TokenKind::Comma));
            }
            expect(TokenKind::RBracket, "Expected ']' after list literal");
            return make_expr(loc, ListExpr{std::move(elems)});
        }
        if (match(TokenKind::LParen)) {
            ExprPtr inner = parse_expr();
            expect(TokenKind::RParen, "Expected ')' after parenthesized expression");
            return inner;
        }
        if (check(TokenKind::Ident)) {
            return make_expr(loc, PathExpr{parse_path()});
        }

        throw ParseError("Unexpected token in expression: '" + tok.text + "'", tok.line, tok.col);
    }
};

}

ProgramAST parse_tokens(const std::vector<Token>& tokens, const std::string& filename) {
    if (tokens.empty()) {
        throw ParseError("Empty token stream", 1, 1);
    }
    Parser parser(tokens, filename);
    return parser.parse_program();
}

ProgramAST parse_strata(std::string_view src, const std::string& filename) {
    auto tokens = tokenize(src);
    return parse_tokens(tokens, filename);
}

}
