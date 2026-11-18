#include "aic/lexer.hpp"
#include "aic/errors.hpp"

#include <cctype>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace aic {

namespace {

bool is_ident_start(unsigned char c) {
    return std::isalpha(c) != 0 || c == '_';
}

bool is_ident_cont(unsigned char c) {
    return std::isalnum(c) != 0 || c == '_';
}

bool is_ascii_digit(unsigned char c) {
    return c >= '0' && c <= '9';
}

const std::unordered_map<std::string_view, TokenKind>& keyword_map() {
    static const std::unordered_map<std::string_view, TokenKind> kws = {
        {"module", TokenKind::KwModule},
        {"import", TokenKind::KwImport},
        {"as", TokenKind::KwAs},
        {"const", TokenKind::KwConst},
        {"frame", TokenKind::KwFrame},
        {"connect", TokenKind::KwConnect},
        {"to", TokenKind::KwTo},
        {"via", TokenKind::KwVia},
        {"in", TokenKind::KwIn},
        {"pole", TokenKind::KwPole},
        {"dipole", TokenKind::KwDipole},
        {"subspace", TokenKind::KwSubspace},
        {"manifold", TokenKind::KwManifold},
        {"field", TokenKind::KwField},
        {"rodata", TokenKind::KwRodata},
        {"anchor", TokenKind::KwAnchor},
        {"entry", TokenKind::KwEntry},
        {"with", TokenKind::KwWith},
        {"flow", TokenKind::KwFlow},
        {"let", TokenKind::KwLet},
        {"when", TokenKind::KwWhen},
        {"else", TokenKind::KwElse},
        {"transport", TokenKind::KwTransport},
        {"insert", TokenKind::KwInsert},
        {"prune", TokenKind::KwPrune},
        {"rewrite", TokenKind::KwRewrite},
        {"decouple", TokenKind::KwDecouple},
        {"temper", TokenKind::KwTemper},
        {"blend", TokenKind::KwBlend},
        {"preserve", TokenKind::KwPreserve},
        {"crosstalk", TokenKind::KwCrosstalk},
        {"approx", TokenKind::KwApprox},
        {"energy", TokenKind::KwEnergy},
        {"true", TokenKind::KwTrue},
        {"false", TokenKind::KwFalse},
        {"Frame", TokenKind::KwFrameType}
    };
    return kws;
}

class Lexer {
public:
    explicit Lexer(std::string_view src) : src_(src) {}

    std::vector<Token> run() {
        std::vector<Token> tokens;
        while (pos_ < src_.size()) {
            skip_ws_and_comments();
            if (pos_ >= src_.size()) {
                break;
            }
            tokens.push_back(next_token());
        }
        tokens.push_back(Token{TokenKind::Eof, "", line_, col_, 0, 0.0});
        return tokens;
    }

private:
    std::string_view src_;
    std::size_t pos_ = 0;
    int line_ = 1;
    int col_ = 1;

    char peek(std::size_t offset = 0) const {
        if (pos_ + offset >= src_.size()) {
            return '\0';
        }
        return src_[pos_ + offset];
    }

    char advance() {
        char c = src_[pos_++];
        if (c == '\n') {
            ++line_;
            col_ = 1;
        } else {
            ++col_;
        }
        return c;
    }

    bool match_sv(std::string_view s) {
        if (pos_ + s.size() > src_.size()) {
            return false;
        }
        if (src_.substr(pos_, s.size()) != s) {
            return false;
        }
        for (std::size_t i = 0; i < s.size(); ++i) {
            advance();
        }
        return true;
    }

    void skip_ws_and_comments() {
        while (pos_ < src_.size()) {
            char c = peek();
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                advance();
                continue;
            }
            if (c == '/' && peek(1) == '/') {
                advance();
                advance();
                while (pos_ < src_.size() && peek() != '\n') {
                    advance();
                }
                continue;
            }
            if (c == '/' && peek(1) == '*') {
                int sl = line_;
                int sc = col_;
                advance();
                advance();
                bool closed = false;
                while (pos_ < src_.size()) {
                    if (peek() == '*' && peek(1) == '/') {
                        advance();
                        advance();
                        closed = true;
                        break;
                    }
                    advance();
                }
                if (!closed) {
                    throw LexError("Unterminated block comment", sl, sc);
                }
                continue;
            }
            break;
        }
    }

    Token next_token() {
        int sl = line_;
        int sc = col_;
        unsigned char uc = static_cast<unsigned char>(peek());

        if (is_ident_start(uc)) {
            std::size_t start = pos_;
            while (pos_ < src_.size() && is_ident_cont(static_cast<unsigned char>(peek()))) {
                advance();
            }
            std::string_view word = src_.substr(start, pos_ - start);
            const auto& kws = keyword_map();
            auto it = kws.find(word);
            if (it != kws.end()) {
                return Token{it->second, std::string(word), sl, sc, 0, 0.0};
            }
            return Token{TokenKind::Ident, std::string(word), sl, sc, 0, 0.0};
        }

        if (is_ascii_digit(uc)) {
            return lex_number(sl, sc);
        }

        if (uc == '"') {
            return lex_string(sl, sc);
        }

        if (uc == '@') {
            if (pos_ + 7 <= src_.size() && src_.substr(pos_, 7) == "@attend") {
                bool end_ok = (pos_ + 7 == src_.size()) ||
                              !is_ident_cont(static_cast<unsigned char>(src_[pos_ + 7]));
                if (end_ok) {
                    match_sv("@attend");
                    return Token{TokenKind::Attend, "@attend", sl, sc, 0, 0.0};
                }
            }
            advance();
            return Token{TokenKind::At, "@", sl, sc, 0, 0.0};
        }

        struct Alias {
            std::string_view seq;
            TokenKind kind;
            std::string_view canonical;
        };
        static const Alias unicode_aliases[] = {
            {"\u25B7", TokenKind::Resonance, "~>" },
            {"\u2295", TokenKind::Excite, "+>" },
            {"\u22A3", TokenKind::Annihilate, "-|" },
            {"\u21A6", TokenKind::MapArrow, "=>" },
            {"\u22C6", TokenKind::Attend, "@attend" },
            {"\u2192", TokenKind::Arrow, "->" },
            {"\u22A5", TokenKind::KwDecouple, "decouple" },
            {"\u27E6", TokenKind::EffectOpen, "![" },
            {"\u27E7", TokenKind::RBracket, "]" }
        };
        for (const auto& ua : unicode_aliases) {
            if (match_sv(ua.seq)) {
                return Token{ua.kind, std::string(ua.canonical), sl, sc, 0, 0.0};
            }
        }

        static const Alias multi_ops[] = {
            {"..=", TokenKind::DotDotEq, "..="},
            {"~>", TokenKind::Resonance, "~>"},
            {"+>", TokenKind::Excite, "+>"},
            {"-|", TokenKind::Annihilate, "-|"},
            {"=>", TokenKind::MapArrow, "=>"},
            {"->", TokenKind::Arrow, "->"},
            {"![", TokenKind::EffectOpen, "!["},
            {"::", TokenKind::ColonColon, "::"},
            {"..", TokenKind::DotDot, ".."},
            {"<=", TokenKind::Le, "<="},
            {">=", TokenKind::Ge, ">="},
            {"==", TokenKind::EqEq, "=="},
            {"!=", TokenKind::Ne, "!="},
            {"||", TokenKind::OrOr, "||"},
            {"&&", TokenKind::AndAnd, "&&"}
        };
        for (const auto& mo : multi_ops) {
            if (match_sv(mo.seq)) {
                return Token{mo.kind, std::string(mo.canonical), sl, sc, 0, 0.0};
            }
        }

        char c = advance();
        switch (c) {
            case ':': return Token{TokenKind::Colon, ":", sl, sc, 0, 0.0};
            case ';': return Token{TokenKind::Semi, ";", sl, sc, 0, 0.0};
            case ',': return Token{TokenKind::Comma, ",", sl, sc, 0, 0.0};
            case '.': return Token{TokenKind::Dot, ".", sl, sc, 0, 0.0};
            case '(': return Token{TokenKind::LParen, "(", sl, sc, 0, 0.0};
            case ')': return Token{TokenKind::RParen, ")", sl, sc, 0, 0.0};
            case '{': return Token{TokenKind::LBrace, "{", sl, sc, 0, 0.0};
            case '}': return Token{TokenKind::RBrace, "}", sl, sc, 0, 0.0};
            case '[': return Token{TokenKind::LBracket, "[", sl, sc, 0, 0.0};
            case ']': return Token{TokenKind::RBracket, "]", sl, sc, 0, 0.0};
            case '<': return Token{TokenKind::Lt, "<", sl, sc, 0, 0.0};
            case '>': return Token{TokenKind::Gt, ">", sl, sc, 0, 0.0};
            case '=': return Token{TokenKind::Eq, "=", sl, sc, 0, 0.0};
            case '+': return Token{TokenKind::Plus, "+", sl, sc, 0, 0.0};
            case '-': return Token{TokenKind::Minus, "-", sl, sc, 0, 0.0};
            case '*': return Token{TokenKind::Star, "*", sl, sc, 0, 0.0};
            case '/': return Token{TokenKind::Slash, "/", sl, sc, 0, 0.0};
            case '!': return Token{TokenKind::Bang, "!", sl, sc, 0, 0.0};
            default:
                throw LexError(std::string("Unexpected character '") + c + "'", sl, sc);
        }
    }

    Token lex_number(int sl, int sc) {
        std::size_t start = pos_;
        while (pos_ < src_.size() && is_ascii_digit(static_cast<unsigned char>(peek()))) {
            advance();
        }
        bool is_float = false;
        if (peek() == '.' && is_ascii_digit(static_cast<unsigned char>(peek(1)))) {
            is_float = true;
            advance();
            while (pos_ < src_.size() && is_ascii_digit(static_cast<unsigned char>(peek()))) {
                advance();
            }
        }
        if (peek() == 'e' || peek() == 'E') {
            bool has_exp = false;
            if (is_ascii_digit(static_cast<unsigned char>(peek(1)))) {
                has_exp = true;
            } else if ((peek(1) == '+' || peek(1) == '-') &&
                       is_ascii_digit(static_cast<unsigned char>(peek(2)))) {
                has_exp = true;
            }
            if (has_exp) {
                is_float = true;
                advance();
                if (peek() == '+' || peek() == '-') {
                    advance();
                }
                while (pos_ < src_.size() && is_ascii_digit(static_cast<unsigned char>(peek()))) {
                    advance();
                }
            }
        }
        std::string text(src_.substr(start, pos_ - start));
        if (is_float) {
            char* end_ptr = nullptr;
            double val = std::strtod(text.c_str(), &end_ptr);
            if (end_ptr != text.c_str() + text.size()) {
                throw LexError("Invalid float literal: " + text, sl, sc);
            }
            return Token{TokenKind::Float, text, sl, sc, 0, val};
        }
        char* end_ptr = nullptr;
        long long val = std::strtoll(text.c_str(), &end_ptr, 10);
        if (end_ptr != text.c_str() + text.size()) {
            throw LexError("Invalid int literal: " + text, sl, sc);
        }
        return Token{TokenKind::Int, text, sl, sc, static_cast<std::int64_t>(val), 0.0};
    }

    Token lex_string(int sl, int sc) {
        advance();
        std::string out;
        while (pos_ < src_.size()) {
            char c = advance();
            if (c == '"') {
                return Token{TokenKind::String, out, sl, sc, 0, 0.0};
            }
            if (c == '\\') {
                if (pos_ >= src_.size()) {
                    throw LexError("Unterminated escape in string literal", sl, sc);
                }
                char esc = advance();
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default:
                        throw LexError(std::string("Unsupported string escape \\") + esc, sl, sc);
                }
            } else {
                out.push_back(c);
            }
        }
        throw LexError("Unterminated string literal", sl, sc);
    }
};

}

std::string token_kind_name(TokenKind kind) {
    switch (kind) {
        case TokenKind::Eof: return "EOF";
        case TokenKind::Ident: return "IDENT";
        case TokenKind::Int: return "INT";
        case TokenKind::Float: return "FLOAT";
        case TokenKind::String: return "STRING";
        case TokenKind::KwModule: return "module";
        case TokenKind::KwImport: return "import";
        case TokenKind::KwAs: return "as";
        case TokenKind::KwConst: return "const";
        case TokenKind::KwFrame: return "frame";
        case TokenKind::KwConnect: return "connect";
        case TokenKind::KwTo: return "to";
        case TokenKind::KwVia: return "via";
        case TokenKind::KwIn: return "in";
        case TokenKind::KwPole: return "pole";
        case TokenKind::KwDipole: return "dipole";
        case TokenKind::KwSubspace: return "subspace";
        case TokenKind::KwManifold: return "manifold";
        case TokenKind::KwField: return "field";
        case TokenKind::KwRodata: return "rodata";
        case TokenKind::KwAnchor: return "anchor";
        case TokenKind::KwEntry: return "entry";
        case TokenKind::KwWith: return "with";
        case TokenKind::KwFlow: return "flow";
        case TokenKind::KwLet: return "let";
        case TokenKind::KwWhen: return "when";
        case TokenKind::KwElse: return "else";
        case TokenKind::KwTransport: return "transport";
        case TokenKind::KwInsert: return "insert";
        case TokenKind::KwPrune: return "prune";
        case TokenKind::KwRewrite: return "rewrite";
        case TokenKind::KwDecouple: return "decouple";
        case TokenKind::KwTemper: return "temper";
        case TokenKind::KwBlend: return "blend";
        case TokenKind::KwPreserve: return "preserve";
        case TokenKind::KwCrosstalk: return "crosstalk";
        case TokenKind::KwApprox: return "approx";
        case TokenKind::KwEnergy: return "energy";
        case TokenKind::KwTrue: return "true";
        case TokenKind::KwFalse: return "false";
        case TokenKind::KwFrameType: return "Frame";
        case TokenKind::Resonance: return "~>";
        case TokenKind::Excite: return "+>";
        case TokenKind::Annihilate: return "-|";
        case TokenKind::MapArrow: return "=>";
        case TokenKind::Attend: return "@attend";
        case TokenKind::Arrow: return "->";
        case TokenKind::EffectOpen: return "![";
        case TokenKind::ColonColon: return "::";
        case TokenKind::DotDotEq: return "..=";
        case TokenKind::DotDot: return "..";
        case TokenKind::Le: return "<=";
        case TokenKind::Ge: return ">=";
        case TokenKind::EqEq: return "==";
        case TokenKind::Ne: return "!=";
        case TokenKind::OrOr: return "||";
        case TokenKind::AndAnd: return "&&";
        case TokenKind::At: return "@";
        case TokenKind::Colon: return ":";
        case TokenKind::Semi: return ";";
        case TokenKind::Comma: return ",";
        case TokenKind::Dot: return ".";
        case TokenKind::LParen: return "(";
        case TokenKind::RParen: return ")";
        case TokenKind::LBrace: return "{";
        case TokenKind::RBrace: return "}";
        case TokenKind::LBracket: return "[";
        case TokenKind::RBracket: return "]";
        case TokenKind::Lt: return "<";
        case TokenKind::Gt: return ">";
        case TokenKind::Eq: return "=";
        case TokenKind::Plus: return "+";
        case TokenKind::Minus: return "-";
        case TokenKind::Star: return "*";
        case TokenKind::Slash: return "/";
        case TokenKind::Bang: return "!";
    }
    return "UNKNOWN";
}

std::vector<Token> tokenize(std::string_view src) {
    Lexer lexer(src);
    return lexer.run();
}

}
