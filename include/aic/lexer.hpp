#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace aic {

enum class TokenKind {
    Eof,
    Ident,
    Int,
    Float,
    String,

    KwModule,
    KwImport,
    KwAs,
    KwConst,
    KwFrame,
    KwConnect,
    KwTo,
    KwVia,
    KwIn,
    KwPole,
    KwDipole,
    KwSubspace,
    KwManifold,
    KwField,
    KwRodata,
    KwAnchor,
    KwEntry,
    KwWith,
    KwFlow,
    KwLet,
    KwWhen,
    KwElse,
    KwTransport,
    KwInsert,
    KwPrune,
    KwRewrite,
    KwDecouple,
    KwTemper,
    KwBlend,
    KwPreserve,
    KwCrosstalk,
    KwApprox,
    KwEnergy,
    KwTrue,
    KwFalse,
    KwFrameType,

    Resonance,
    Excite,
    Annihilate,
    MapArrow,
    Attend,
    Arrow,
    EffectOpen,

    ColonColon,
    DotDotEq,
    DotDot,
    Le,
    Ge,
    EqEq,
    Ne,
    OrOr,
    AndAnd,

    At,
    Colon,
    Semi,
    Comma,
    Dot,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Lt,
    Gt,
    Eq,
    Plus,
    Minus,
    Star,
    Slash,
    Bang
};

struct Token {
    TokenKind kind = TokenKind::Eof;
    std::string text;
    int line = 1;
    int col = 1;
    std::int64_t int_val = 0;
    double float_val = 0.0;
};

std::string token_kind_name(TokenKind kind);
std::vector<Token> tokenize(std::string_view src);

}
