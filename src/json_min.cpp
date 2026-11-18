#include "aic/json_min.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace aic {

namespace {

void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp <= 0x7Fu) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0x10FFFFu) {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        throw std::runtime_error("Invalid Unicode code point in JSON");
    }
}

class JsonParser {
public:
    explicit JsonParser(std::string_view src) : src_(src) {}

    JsonValue parse() {
        skip_ws();
        JsonValue root = parse_value();
        skip_ws();
        if (pos_ != src_.size()) {
            throw std::runtime_error("Unexpected trailing characters after JSON value");
        }
        return root;
    }

private:
    std::string_view src_;
    std::size_t pos_ = 0;

    void skip_ws() {
        while (pos_ < src_.size()) {
            const char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    char peek() const {
        if (pos_ >= src_.size()) {
            throw std::runtime_error("Unexpected end of JSON input");
        }
        return src_[pos_];
    }

    char get() {
        const char c = peek();
        ++pos_;
        return c;
    }

    void expect_literal(std::string_view lit) {
        if (src_.substr(pos_, lit.size()) != lit) {
            throw std::runtime_error("Invalid literal in JSON");
        }
        pos_ += lit.size();
    }

    std::uint32_t parse_hex4() {
        if (pos_ + 4 > src_.size()) {
            throw std::runtime_error("Incomplete unicode escape in JSON string");
        }
        std::uint32_t val = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = src_[pos_++];
            val <<= 4;
            if (c >= '0' && c <= '9') {
                val |= static_cast<std::uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                val |= static_cast<std::uint32_t>(10 + (c - 'a'));
            } else if (c >= 'A' && c <= 'F') {
                val |= static_cast<std::uint32_t>(10 + (c - 'A'));
            } else {
                throw std::runtime_error("Invalid hex digit in unicode escape");
            }
        }
        return val;
    }

    std::string parse_string() {
        if (get() != '"') {
            throw std::runtime_error("Expected '\"' to begin JSON string");
        }
        std::string out;
        while (true) {
            if (pos_ >= src_.size()) {
                throw std::runtime_error("Unterminated JSON string");
            }
            const unsigned char uc = static_cast<unsigned char>(src_[pos_++]);
            if (uc == '"') {
                return out;
            }
            if (uc < 0x20u) {
                throw std::runtime_error("Unescaped control character in JSON string");
            }
            if (uc != '\\') {
                out.push_back(static_cast<char>(uc));
                continue;
            }
            const char esc = get();
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    const std::uint32_t cp = parse_hex4();
                    if (cp >= 0xD800u && cp <= 0xDBFFu) {
                        if (pos_ + 2 > src_.size() || src_[pos_] != '\\' || src_[pos_ + 1] != 'u') {
                            throw std::runtime_error("Unpaired high surrogate in JSON string");
                        }
                        pos_ += 2;
                        const std::uint32_t low = parse_hex4();
                        if (low < 0xDC00u || low > 0xDFFFu) {
                            throw std::runtime_error("Invalid low surrogate in JSON string");
                        }
                        const std::uint32_t combined =
                            0x10000u + (((cp - 0xD800u) << 10) | (low - 0xDC00u));
                        append_utf8(out, combined);
                    } else if (cp >= 0xDC00u && cp <= 0xDFFFu) {
                        throw std::runtime_error("Unpaired low surrogate in JSON string");
                    } else {
                        append_utf8(out, cp);
                    }
                    break;
                }
                default:
                    throw std::runtime_error("Invalid escape sequence in JSON string");
            }
        }
    }

    JsonValue parse_number() {
        const std::size_t start = pos_;
        if (pos_ < src_.size() && src_[pos_] == '-') {
            ++pos_;
        }
        if (pos_ >= src_.size()) {
            throw std::runtime_error("Invalid JSON number");
        }
        if (src_[pos_] == '0') {
            ++pos_;
            if (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') {
                throw std::runtime_error("Leading zero in JSON number");
            }
        } else if (src_[pos_] >= '1' && src_[pos_] <= '9') {
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') {
                ++pos_;
            }
        } else {
            throw std::runtime_error("Invalid JSON number");
        }

        if (pos_ < src_.size() && src_[pos_] == '.') {
            ++pos_;
            if (pos_ >= src_.size() || src_[pos_] < '0' || src_[pos_] > '9') {
                throw std::runtime_error("Invalid fractional part in JSON number");
            }
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') {
                ++pos_;
            }
        }

        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= src_.size() || src_[pos_] < '0' || src_[pos_] > '9') {
                throw std::runtime_error("Invalid exponent in JSON number");
            }
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') {
                ++pos_;
            }
        }

        const std::string token(src_.substr(start, pos_ - start));
        char* end_ptr = nullptr;
        const double val = std::strtod(token.c_str(), &end_ptr);
        if (end_ptr != token.c_str() + token.size() || !std::isfinite(val)) {
            throw std::runtime_error("Numeric overflow or parse error in JSON number");
        }
        return JsonValue(val);
    }

    JsonValue parse_array() {
        if (get() != '[') {
            throw std::runtime_error("Expected '[' to start JSON array");
        }
        JsonValue::Array arr;
        skip_ws();
        if (peek() == ']') {
            ++pos_;
            return JsonValue(std::move(arr));
        }
        while (true) {
            skip_ws();
            arr.push_back(parse_value());
            skip_ws();
            const char sep = get();
            if (sep == ']') {
                return JsonValue(std::move(arr));
            }
            if (sep != ',') {
                throw std::runtime_error("Expected ',' or ']' in JSON array");
            }
        }
    }

    JsonValue parse_object() {
        if (get() != '{') {
            throw std::runtime_error("Expected '{' to start JSON object");
        }
        JsonValue::Object obj;
        skip_ws();
        if (peek() == '}') {
            ++pos_;
            return JsonValue(std::move(obj));
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            if (get() != ':') {
                throw std::runtime_error("Expected ':' after object key in JSON");
            }
            skip_ws();
            obj[std::move(key)] = parse_value();
            skip_ws();
            const char sep = get();
            if (sep == '}') {
                return JsonValue(std::move(obj));
            }
            if (sep != ',') {
                throw std::runtime_error("Expected ',' or '}' in JSON object");
            }
        }
    }

    JsonValue parse_value() {
        skip_ws();
        const char c = peek();
        if (c == 'n') {
            expect_literal("null");
            return JsonValue(nullptr);
        }
        if (c == 't') {
            expect_literal("true");
            return JsonValue(true);
        }
        if (c == 'f') {
            expect_literal("false");
            return JsonValue(false);
        }
        if (c == '"') {
            return JsonValue(parse_string());
        }
        if (c == '[') {
            return parse_array();
        }
        if (c == '{') {
            return parse_object();
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return parse_number();
        }
        throw std::runtime_error("Unexpected character in JSON input");
    }
};

void escape_json_string(std::string& out, const std::string& s) {
    static constexpr char hex_digits[] = "0123456789abcdef";
    out.push_back('"');
    for (unsigned char uc : s) {
        switch (uc) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (uc < 0x20u) {
                    out += "\\u00";
                    out.push_back(hex_digits[(uc >> 4) & 0xFu]);
                    out.push_back(hex_digits[uc & 0xFu]);
                } else {
                    out.push_back(static_cast<char>(uc));
                }
                break;
        }
    }
    out.push_back('"');
}

void dump_number(std::string& out, double d) {
    if (!std::isfinite(d)) {
        throw std::runtime_error("JSON does not support NaN or Infinity");
    }
    if (d == 0.0) {
        out.push_back('0');
        return;
    }
    if (d == std::trunc(d) && d >= -9007199254740992.0 && d <= 9007199254740992.0) {
        const auto iv = static_cast<std::int64_t>(d);
        char buf[32];
        auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), iv);
        if (ec == std::errc{}) {
            out.append(buf, ptr);
            return;
        }
    }
    char buf[64];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), d);
    if (ec != std::errc{}) {
        throw std::runtime_error("Failed to format JSON double");
    }
    out.append(buf, ptr);
}

void dump_impl(std::string& out, const JsonValue& v, int indent, int depth) {
    if (v.is_null()) {
        out += "null";
    } else if (v.is_bool()) {
        out += (v.as_bool() ? "true" : "false");
    } else if (v.is_number()) {
        dump_number(out, v.as_number());
    } else if (v.is_string()) {
        escape_json_string(out, v.as_string());
    } else if (v.is_array()) {
        const auto& arr = v.as_array();
        if (arr.empty()) {
            out += "[]";
            return;
        }
        out.push_back('[');
        for (std::size_t i = 0; i < arr.size(); ++i) {
            if (i > 0) {
                out.push_back(',');
            }
            if (indent >= 0) {
                out.push_back('\n');
                out.append(static_cast<std::size_t>((depth + 1) * indent), ' ');
            }
            dump_impl(out, arr[i], indent, depth + 1);
        }
        if (indent >= 0) {
            out.push_back('\n');
            out.append(static_cast<std::size_t>(depth * indent), ' ');
        }
        out.push_back(']');
    } else if (v.is_object()) {
        const auto& obj = v.as_object();
        if (obj.empty()) {
            out += "{}";
            return;
        }
        out.push_back('{');
        bool first = true;
        for (const auto& [k, val] : obj) {
            if (!first) {
                out.push_back(',');
            }
            first = false;
            if (indent >= 0) {
                out.push_back('\n');
                out.append(static_cast<std::size_t>((depth + 1) * indent), ' ');
            }
            escape_json_string(out, k);
            out.push_back(':');
            if (indent >= 0) {
                out.push_back(' ');
            }
            dump_impl(out, val, indent, depth + 1);
        }
        if (indent >= 0) {
            out.push_back('\n');
            out.append(static_cast<std::size_t>(depth * indent), ' ');
        }
        out.push_back('}');
    }
}

}

JsonValue::JsonValue(std::nullptr_t) : value(nullptr) {}
JsonValue::JsonValue(bool b) : value(b) {}
JsonValue::JsonValue(const char* s) : value(std::string(s)) {}
JsonValue::JsonValue(std::string s) : value(std::move(s)) {}
JsonValue::JsonValue(std::string_view s) : value(std::string(s)) {}
JsonValue::JsonValue(Array a) : value(std::move(a)) {}
JsonValue::JsonValue(Object o) : value(std::move(o)) {}

JsonValue JsonValue::array() {
    return JsonValue(Array{});
}

JsonValue JsonValue::object() {
    return JsonValue(Object{});
}

bool JsonValue::is_null() const {
    return std::holds_alternative<std::nullptr_t>(value);
}

bool JsonValue::is_bool() const {
    return std::holds_alternative<bool>(value);
}

bool JsonValue::is_number() const {
    return std::holds_alternative<double>(value);
}

bool JsonValue::is_string() const {
    return std::holds_alternative<std::string>(value);
}

bool JsonValue::is_array() const {
    return std::holds_alternative<Array>(value);
}

bool JsonValue::is_object() const {
    return std::holds_alternative<Object>(value);
}

bool JsonValue::as_bool() const {
    if (!is_bool()) {
        throw std::runtime_error("JsonValue is not a bool");
    }
    return std::get<bool>(value);
}

double JsonValue::as_number() const {
    if (!is_number()) {
        throw std::runtime_error("JsonValue is not a number");
    }
    return std::get<double>(value);
}

std::int64_t JsonValue::as_int() const {
    const double d = as_number();
    if (!std::isfinite(d)) {
        throw std::runtime_error("JsonValue number is not finite");
    }
    return static_cast<std::int64_t>(d);
}

const std::string& JsonValue::as_string() const {
    if (!is_string()) {
        throw std::runtime_error("JsonValue is not a string");
    }
    return std::get<std::string>(value);
}

const JsonValue::Array& JsonValue::as_array() const {
    if (!is_array()) {
        throw std::runtime_error("JsonValue is not an array");
    }
    return std::get<Array>(value);
}

JsonValue::Array& JsonValue::as_array() {
    if (!is_array()) {
        throw std::runtime_error("JsonValue is not an array");
    }
    return std::get<Array>(value);
}

const JsonValue::Object& JsonValue::as_object() const {
    if (!is_object()) {
        throw std::runtime_error("JsonValue is not an object");
    }
    return std::get<Object>(value);
}

JsonValue::Object& JsonValue::as_object() {
    if (!is_object()) {
        throw std::runtime_error("JsonValue is not an object");
    }
    return std::get<Object>(value);
}

bool JsonValue::has(const std::string& key) const {
    if (!is_object()) {
        return false;
    }
    return std::get<Object>(value).contains(key);
}

const JsonValue& JsonValue::operator[](const std::string& key) const {
    const Object& obj = as_object();
    const auto it = obj.find(key);
    if (it == obj.end()) {
        throw std::out_of_range("Missing JSON key: " + key);
    }
    return it->second;
}

JsonValue& JsonValue::operator[](const std::string& key) {
    if (is_null()) {
        value = Object{};
    }
    return as_object()[key];
}

JsonValue parse_json(std::string_view src) {
    JsonParser parser(src);
    return parser.parse();
}

std::string dump_json(const JsonValue& v, int indent) {
    std::string out;
    dump_impl(out, v, indent, 0);
    return out;
}

}
