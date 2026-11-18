#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace aic {

struct JsonValue {
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;
    using Storage = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;

    Storage value = nullptr;

    JsonValue() = default;
    JsonValue(std::nullptr_t);
    JsonValue(bool b);

    template <typename T>
    requires (std::is_arithmetic_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool>)
    JsonValue(T n) : value(static_cast<double>(n)) {}

    JsonValue(const char* s);
    JsonValue(std::string s);
    JsonValue(std::string_view s);
    JsonValue(Array a);
    JsonValue(Object o);

    static JsonValue array();
    static JsonValue object();

    bool is_null() const;
    bool is_bool() const;
    bool is_number() const;
    bool is_string() const;
    bool is_array() const;
    bool is_object() const;

    bool as_bool() const;
    double as_number() const;
    std::int64_t as_int() const;
    const std::string& as_string() const;
    const Array& as_array() const;
    Array& as_array();
    const Object& as_object() const;
    Object& as_object();

    bool has(const std::string& key) const;
    const JsonValue& operator[](const std::string& key) const;
    JsonValue& operator[](const std::string& key);

    bool operator==(const JsonValue& other) const = default;
};

JsonValue parse_json(std::string_view src);
std::string dump_json(const JsonValue& v, int indent = 2);

}
