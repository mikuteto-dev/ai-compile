#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace aic {

enum class ErrorKind {
    Lex,
    Parse,
    FrameMismatch,
    UnipolarViolation,
    CrosstalkExceeded,
    ApproximationUnmet,
    EnergyBudgetExceeded,
    EnergyUnbounded,
    UndefinedSymbol,
    OutOfDepth,
    NullspaceExhausted,
    GQAHeadSharingConflict,
    IO
};

inline std::string error_kind_name(ErrorKind k) {
    switch (k) {
        case ErrorKind::Lex: return "LexError";
        case ErrorKind::Parse: return "ParseError";
        case ErrorKind::FrameMismatch: return "FrameMismatchError";
        case ErrorKind::UnipolarViolation: return "UnipolarViolationError";
        case ErrorKind::CrosstalkExceeded: return "CrosstalkExceededError";
        case ErrorKind::ApproximationUnmet: return "ApproximationUnmetError";
        case ErrorKind::EnergyBudgetExceeded: return "EnergyBudgetExceededError";
        case ErrorKind::EnergyUnbounded: return "EnergyUnboundedError";
        case ErrorKind::UndefinedSymbol: return "UndefinedSymbolError";
        case ErrorKind::OutOfDepth: return "OutOfDepthError";
        case ErrorKind::NullspaceExhausted: return "NullspaceExhaustedError";
        case ErrorKind::GQAHeadSharingConflict: return "GQAHeadSharingConflictError";
        case ErrorKind::IO: return "IOError";
    }
    return "StrataError";
}

inline std::string format_strata_error(ErrorKind k, const std::string& msg, int line, int col) {
    std::string out = error_kind_name(k);
    if (line > 0) {
        out += " at " + std::to_string(line);
        if (col > 0) {
            out += ":" + std::to_string(col);
        }
    }
    if (!msg.empty()) {
        out += ": " + msg;
    }
    return out;
}

class StrataError : public std::runtime_error {
public:
    ErrorKind kind;
    int line = 0;
    int col = 0;
    std::string message;

    StrataError(ErrorKind k, const std::string& msg, int l = 0, int c = 0)
        : std::runtime_error(format_strata_error(k, msg, l, c)),
          kind(k),
          line(l),
          col(c),
          message(msg) {}
};

class LexError : public StrataError {
public:
    explicit LexError(const std::string& msg, int l = 0, int c = 0)
        : StrataError(ErrorKind::Lex, msg, l, c) {}
};

class ParseError : public StrataError {
public:
    explicit ParseError(const std::string& msg, int l = 0, int c = 0)
        : StrataError(ErrorKind::Parse, msg, l, c) {}
};

class FrameMismatchError : public StrataError {
public:
    explicit FrameMismatchError(const std::string& msg, int l = 0, int c = 0)
        : StrataError(ErrorKind::FrameMismatch, msg, l, c) {}
};

class UnipolarViolationError : public StrataError {
public:
    explicit UnipolarViolationError(const std::string& msg, int l = 0, int c = 0)
        : StrataError(ErrorKind::UnipolarViolation, msg, l, c) {}
};

class CrosstalkExceededError : public StrataError {
public:
    explicit CrosstalkExceededError(const std::string& msg, int l = 0, int c = 0)
        : StrataError(ErrorKind::CrosstalkExceeded, msg, l, c) {}
};

class ApproximationUnmetError : public StrataError {
public:
    explicit ApproximationUnmetError(const std::string& msg, int l = 0, int c = 0)
        : StrataError(ErrorKind::ApproximationUnmet, msg, l, c) {}
};

class EnergyBudgetExceededError : public StrataError {
public:
    explicit EnergyBudgetExceededError(const std::string& msg, int l = 0, int c = 0)
        : StrataError(ErrorKind::EnergyBudgetExceeded, msg, l, c) {}
};

class EnergyUnboundedError : public StrataError {
public:
    explicit EnergyUnboundedError(const std::string& msg, int l = 0, int c = 0)
        : StrataError(ErrorKind::EnergyUnbounded, msg, l, c) {}
};

class UndefinedSymbolError : public StrataError {
public:
    explicit UndefinedSymbolError(const std::string& msg, int l = 0, int c = 0)
        : StrataError(ErrorKind::UndefinedSymbol, msg, l, c) {}
};

class OutOfDepthError : public StrataError {
public:
    explicit OutOfDepthError(const std::string& msg, int l = 0, int c = 0)
        : StrataError(ErrorKind::OutOfDepth, msg, l, c) {}
};

class NullspaceExhaustedError : public StrataError {
public:
    explicit NullspaceExhaustedError(const std::string& msg, int l = 0, int c = 0)
        : StrataError(ErrorKind::NullspaceExhausted, msg, l, c) {}
};

class GQAHeadSharingConflictError : public StrataError {
public:
    explicit GQAHeadSharingConflictError(const std::string& msg, int l = 0, int c = 0)
        : StrataError(ErrorKind::GQAHeadSharingConflict, msg, l, c) {}
};

class IOError : public StrataError {
public:
    explicit IOError(const std::string& msg, int l = 0, int c = 0)
        : StrataError(ErrorKind::IO, msg, l, c) {}
};

inline std::string format_diagnostic(const StrataError& err,
                                     std::string_view source,
                                     const std::string& filename = "<input>") {
    std::ostringstream oss;
    const std::string& msg = err.message.empty() ? std::string(err.what()) : err.message;
    if (err.line > 0) {
        int col = err.col > 0 ? err.col : 1;
        oss << filename << ":" << err.line << ":" << col << ": "
            << error_kind_name(err.kind) << ": " << msg << "\n";
        std::vector<std::string_view> lines;
        std::size_t start = 0;
        while (start <= source.size()) {
            std::size_t nl = source.find('\n', start);
            if (nl == std::string_view::npos) {
                std::string_view ln = source.substr(start);
                if (!ln.empty() && ln.back() == '\r') {
                    ln.remove_suffix(1);
                }
                lines.push_back(ln);
                break;
            }
            std::string_view ln = source.substr(start, nl - start);
            if (!ln.empty() && ln.back() == '\r') {
                ln.remove_suffix(1);
            }
            lines.push_back(ln);
            start = nl + 1;
        }
        std::size_t line_idx = static_cast<std::size_t>(err.line - 1);
        if (line_idx < lines.size()) {
            std::string_view line_text = lines[line_idx];
            std::string lnum = std::to_string(err.line);
            std::string pad(lnum.size(), ' ');
            std::size_t caret_col = static_cast<std::size_t>(col - 1);
            if (caret_col > line_text.size()) {
                caret_col = line_text.size();
            }
            std::string caret_pad;
            caret_pad.reserve(caret_col);
            for (std::size_t i = 0; i < caret_col; ++i) {
                caret_pad.push_back(line_text[i] == '\t' ? '\t' : ' ');
            }
            oss << pad << " |\n";
            oss << lnum << " | " << line_text << "\n";
            oss << pad << " | " << caret_pad << "^~~~\n";
        }
    } else {
        oss << filename << ": " << error_kind_name(err.kind) << ": " << msg << "\n";
    }
    return oss.str();
}

}
