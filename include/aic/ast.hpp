#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace aic {

struct SourceLoc {
    int line = 0;
    int col = 0;
};

struct StageSpec {
    std::string start = "All";
    std::string end;
    bool inclusive = false;
    SourceLoc loc;
};

struct FrameRef {
    bool is_explicit_frame = false;
    StageSpec stage;
    std::vector<std::string> path;
    SourceLoc loc;
};

struct TypeRef;

struct TypeArg {
    enum class Kind { Int, Float, Type };
    Kind kind = Kind::Type;
    std::int64_t int_val = 0;
    double float_val = 0.0;
    std::shared_ptr<TypeRef> type_ref;
};

struct TypeRef {
    std::string name;
    std::vector<TypeArg> args;
    SourceLoc loc;
};

struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

struct Arg {
    std::string name;
    ExprPtr value;
    SourceLoc loc;
};

struct IntLitExpr {
    std::int64_t value = 0;
};

struct FloatLitExpr {
    double value = 0.0;
};

struct StringLitExpr {
    std::string value;
};

struct BoolLitExpr {
    bool value = false;
};

struct PathExpr {
    std::vector<std::string> parts;
};

struct ListExpr {
    std::vector<ExprPtr> elements;
};

struct UnaryExpr {
    std::string op;
    ExprPtr operand;
};

struct BinaryExpr {
    std::string op;
    ExprPtr lhs;
    ExprPtr rhs;
};

struct MemberExpr {
    ExprPtr object;
    std::string member;
};

struct CallExpr {
    ExprPtr callee;
    std::vector<Arg> args;
};

struct AttendExpr {
    std::vector<Arg> args;
};

struct TransportExpr {
    ExprPtr expr;
    FrameRef src_frame;
    FrameRef dst_frame;
};

struct FieldOpExpr {
    std::string op;
    ExprPtr lhs;
    ExprPtr rhs;
    std::vector<Arg> args;
};

struct FieldCallExpr {
    std::string op;
    std::vector<Arg> args;
};

struct Expr {
    SourceLoc loc;
    std::variant<
        IntLitExpr,
        FloatLitExpr,
        StringLitExpr,
        BoolLitExpr,
        PathExpr,
        ListExpr,
        UnaryExpr,
        BinaryExpr,
        MemberExpr,
        CallExpr,
        AttendExpr,
        TransportExpr,
        FieldOpExpr,
        FieldCallExpr
    > node;
};

struct Stmt;
using StmtPtr = std::shared_ptr<Stmt>;

struct Block {
    std::vector<StmtPtr> stmts;
    SourceLoc loc;
};

struct LetStmt {
    std::string name;
    std::optional<TypeRef> type_ann;
    ExprPtr value;
};

struct WhenStmt {
    ExprPtr cond;
    std::string bind_name;
    Block body;
    std::vector<StmtPtr> else_stmts;
    bool has_else = false;
};

struct RodataStmt {
    std::string op;
    std::vector<std::string> target_path;
    std::vector<Arg> args;
};

struct ActionStmt {
    std::string op;
    ExprPtr lhs;
    ExprPtr rhs;
};

struct ExprStmt {
    ExprPtr expr;
};

struct Stmt {
    SourceLoc loc;
    std::variant<
        LetStmt,
        WhenStmt,
        RodataStmt,
        ActionStmt,
        ExprStmt
    > node;
};

struct Annotation {
    std::string name;
    std::vector<Arg> args;
    SourceLoc loc;
};

struct AnchorDecl {
    std::string kind;
    std::vector<Arg> args;
    SourceLoc loc;
};

struct FrameConnection {
    std::string target;
    std::string via;
    SourceLoc loc;
};

struct DeclMeta {
    std::string chart_name;
    ExprPtr chart_expr;
    std::string metric;
    std::optional<int> rank;
    ExprPtr curvature;
    std::vector<AnchorDecl> anchors;
};

struct PoleSide {
    std::string polarity;
    std::string name;
    std::vector<AnchorDecl> anchors;
    SourceLoc loc;
};

struct RodataEntryAST {
    ExprPtr key;
    ExprPtr value;
    std::vector<Arg> attrs;
    SourceLoc loc;
};

struct ParamDecl {
    std::string name;
    TypeRef type;
    std::optional<FrameRef> frame;
    SourceLoc loc;
};

struct ConstraintAST {
    std::string kind;
    std::string cmp_op;
    double number_val = 0.0;
    std::vector<std::vector<std::string>> preserve_paths;
    SourceLoc loc;
};

struct EffectContractAST {
    std::vector<ConstraintAST> constraints;
    SourceLoc loc;
};

struct FrameDecl {
    std::string name;
    std::optional<StageSpec> stage;
    std::vector<FrameConnection> connections;
    std::vector<AnchorDecl> anchors;
};

struct ConstDecl {
    std::string name;
    std::optional<TypeRef> type_ann;
    ExprPtr value;
};

struct PoleDecl {
    std::string name;
    FrameRef frame;
    DeclMeta meta;
};

struct DipoleDecl {
    std::string name;
    FrameRef frame;
    std::vector<PoleSide> poles;
    DeclMeta meta;
};

struct SubspaceDecl {
    std::string name;
    int dim = 1;
    FrameRef frame;
    DeclMeta meta;
};

struct ManifoldDecl {
    std::string name;
    TypeRef topo;
    FrameRef frame;
    DeclMeta meta;
};

struct FieldDecl {
    std::string name;
    std::optional<TypeRef> topo;
    FrameRef frame;
    DeclMeta meta;
};

struct RodataDecl {
    std::string name;
    TypeRef key_type;
    TypeRef val_type;
    FrameRef frame;
    std::vector<RodataEntryAST> entries;
    DeclMeta meta;
};

struct FlowDecl {
    std::string name;
    std::vector<ParamDecl> params;
    std::optional<TypeRef> return_type;
    std::optional<FrameRef> return_frame;
    EffectContractAST contract;
    bool has_contract = false;
    Block body;
};

struct ExpectAssertionAST {
    std::string kind;
    ExprPtr prompt_expr;
    std::vector<std::string> symbol_path;
    std::string pole_side;
    std::string cmp_op;
    bool rhs_is_base = false;
    ExprPtr rhs_expr;
    ExprPtr target_str_expr;
    ExprPtr foil_str_expr;
    SourceLoc loc;
};

struct TestDecl {
    std::string name;
    std::vector<ExpectAssertionAST> assertions;
};

struct TopItem {
    SourceLoc loc;
    std::vector<Annotation> annotations;
    std::variant<
        FrameDecl,
        ConstDecl,
        PoleDecl,
        DipoleDecl,
        SubspaceDecl,
        ManifoldDecl,
        FieldDecl,
        RodataDecl,
        FlowDecl,
        TestDecl
    > node;
};

struct ImportDecl {
    std::vector<std::string> path;
    std::string alias;
    SourceLoc loc;
};

struct ProgramAST {
    std::string filename;
    std::vector<std::string> module_path;
    std::vector<ImportDecl> imports;
    std::vector<TopItem> items;
};

}
