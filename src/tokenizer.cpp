#include "aic/tokenizer.hpp"
#include "aic/errors.hpp"
#include "aic/json_min.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace aic {

namespace {

std::string cp_to_utf8(std::uint32_t cp) {
    std::string out;
    if (cp <= 0x7Fu) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
    return out;
}

bool next_utf8_cp(std::string_view s, std::size_t& pos, std::uint32_t& cp, std::size_t& byte_len) {
    if (pos >= s.size()) {
        return false;
    }
    std::size_t start = pos;
    unsigned char b0 = static_cast<unsigned char>(s[pos]);
    if (b0 < 0x80u) {
        cp = b0;
        pos += 1;
        byte_len = 1;
        return true;
    }
    if ((b0 & 0xE0u) == 0xC0u && pos + 1 < s.size()) {
        unsigned char b1 = static_cast<unsigned char>(s[pos + 1]);
        if ((b1 & 0xC0u) == 0x80u) {
            cp = ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
            pos += 2;
            byte_len = 2;
            return true;
        }
    }
    if ((b0 & 0xF0u) == 0xE0u && pos + 2 < s.size()) {
        unsigned char b1 = static_cast<unsigned char>(s[pos + 1]);
        unsigned char b2 = static_cast<unsigned char>(s[pos + 2]);
        if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u) {
            cp = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
            pos += 3;
            byte_len = 3;
            return true;
        }
    }
    if ((b0 & 0xF8u) == 0xF0u && pos + 3 < s.size()) {
        unsigned char b1 = static_cast<unsigned char>(s[pos + 1]);
        unsigned char b2 = static_cast<unsigned char>(s[pos + 2]);
        unsigned char b3 = static_cast<unsigned char>(s[pos + 3]);
        if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u && (b3 & 0xC0u) == 0x80u) {
            cp = ((b0 & 0x07u) << 18) | ((b1 & 0x3Fu) << 12) | ((b2 & 0x3Fu) << 6) | (b3 & 0x3Fu);
            pos += 4;
            byte_len = 4;
            return true;
        }
    }
    cp = b0;
    pos = start + 1;
    byte_len = 1;
    return true;
}

struct ByteUnicodeTable {
    std::array<std::string, 256> b2u;
    std::unordered_map<std::uint32_t, unsigned char> u2b;
};

const ByteUnicodeTable& byte_unicode_table() {
    static const ByteUnicodeTable tbl = []() {
        ByteUnicodeTable t;
        std::array<bool, 256> direct{};
        for (int b = 33; b <= 126; ++b) direct[static_cast<std::size_t>(b)] = true;
        for (int b = 161; b <= 172; ++b) direct[static_cast<std::size_t>(b)] = true;
        for (int b = 174; b <= 255; ++b) direct[static_cast<std::size_t>(b)] = true;
        std::uint32_t n = 0;
        for (int b = 0; b < 256; ++b) {
            std::uint32_t cp = 0;
            if (direct[static_cast<std::size_t>(b)]) {
                cp = static_cast<std::uint32_t>(b);
            } else {
                cp = 256u + n;
                ++n;
            }
            t.b2u[static_cast<std::size_t>(b)] = cp_to_utf8(cp);
            t.u2b[cp] = static_cast<unsigned char>(b);
        }
        return t;
    }();
    return tbl;
}

bool is_ascii_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool is_ascii_digit_char(unsigned char c) {
    return c >= '0' && c <= '9';
}

bool is_ascii_ws(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

std::vector<std::string> pretokenize_byte_level(std::string_view text) {
    std::vector<std::string> chunks;
    std::size_t i = 0;
    const std::size_t n = text.size();
    while (i < n) {
        std::size_t start = i;
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == ' ' && i + 1 < n) {
            unsigned char nxt = static_cast<unsigned char>(text[i + 1]);
            if (is_ascii_letter(nxt) || nxt >= 0x80u) {
                ++i;
                while (i < n) {
                    unsigned char cur = static_cast<unsigned char>(text[i]);
                    if (is_ascii_letter(cur) || cur >= 0x80u) {
                        ++i;
                    } else {
                        break;
                    }
                }
                chunks.emplace_back(text.substr(start, i - start));
                continue;
            }
            if (is_ascii_digit_char(nxt)) {
                ++i;
                while (i < n && is_ascii_digit_char(static_cast<unsigned char>(text[i]))) {
                    ++i;
                }
                chunks.emplace_back(text.substr(start, i - start));
                continue;
            }
            if (!is_ascii_ws(nxt)) {
                ++i;
                while (i < n) {
                    unsigned char cur = static_cast<unsigned char>(text[i]);
                    if (!is_ascii_ws(cur) && !is_ascii_letter(cur) &&
                        !is_ascii_digit_char(cur) && cur < 0x80u) {
                        ++i;
                    } else {
                        break;
                    }
                }
                chunks.emplace_back(text.substr(start, i - start));
                continue;
            }
        }
        if (is_ascii_letter(c) || c >= 0x80u) {
            while (i < n) {
                unsigned char cur = static_cast<unsigned char>(text[i]);
                if (is_ascii_letter(cur) || cur >= 0x80u) {
                    ++i;
                } else {
                    break;
                }
            }
            chunks.emplace_back(text.substr(start, i - start));
            continue;
        }
        if (is_ascii_digit_char(c)) {
            while (i < n && is_ascii_digit_char(static_cast<unsigned char>(text[i]))) {
                ++i;
            }
            chunks.emplace_back(text.substr(start, i - start));
            continue;
        }
        if (is_ascii_ws(c)) {
            while (i < n && is_ascii_ws(static_cast<unsigned char>(text[i]))) {
                if (text[i] == ' ' && i + 1 < n && !is_ascii_ws(static_cast<unsigned char>(text[i + 1])) &&
                    i > start) {
                    break;
                }
                ++i;
            }
            chunks.emplace_back(text.substr(start, i - start));
            continue;
        }
        while (i < n) {
            unsigned char cur = static_cast<unsigned char>(text[i]);
            if (!is_ascii_ws(cur) && !is_ascii_letter(cur) &&
                !is_ascii_digit_char(cur) && cur < 0x80u) {
                ++i;
            } else {
                break;
            }
        }
        chunks.emplace_back(text.substr(start, i - start));
    }
    return chunks;
}

bool try_parse_hex_byte_token(const std::string& tok, unsigned char& out_byte) {
    if (tok.size() != 6 || tok[0] != '<' || tok[1] != '0' || tok[2] != 'x' || tok[5] != '>') {
        return false;
    }
    auto hex_val = [](char ch) -> int {
        if (ch >= '0' && ch <= '9') return ch - '0';
        if (ch >= 'a' && ch <= 'f') return 10 + (ch - 'a');
        if (ch >= 'A' && ch <= 'F') return 10 + (ch - 'A');
        return -1;
    };
    int hi = hex_val(tok[3]);
    int lo = hex_val(tok[4]);
    if (hi < 0 || lo < 0) {
        return false;
    }
    out_byte = static_cast<unsigned char>((hi << 4) | lo);
    return true;
}

std::string byte_hex_token(unsigned char b) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string out = "<0x00>";
    out[3] = hex[(b >> 4) & 0xFu];
    out[4] = hex[b & 0xFu];
    return out;
}

}

void Tokenizer::rebuild_tables() {
    id_to_token_.clear();
    added_ids_.clear();
    merge_ranks_.clear();
    unk_id_ = -1;

    for (const auto& [tok, id] : vocab_) {
        id_to_token_[id] = tok;
    }
    for (const auto& at : added_tokens_) {
        vocab_[at.content] = at.id;
        id_to_token_[at.id] = at.content;
        added_ids_[at.id] = true;
    }
    std::sort(added_tokens_.begin(), added_tokens_.end(), [](const AddedToken& a, const AddedToken& b) {
        if (a.content.size() != b.content.size()) {
            return a.content.size() > b.content.size();
        }
        return a.id < b.id;
    });

    for (std::size_t i = 0; i < merges_.size(); ++i) {
        merge_ranks_[merges_[i]] = static_cast<int>(i);
    }

    const std::string sp_space = "\xE2\x96\x81";
    const std::string gpt_space = "\xC4\xA0";
    bool has_gpt = vocab_.count(gpt_space) > 0;
    bool has_sp = vocab_.count(sp_space) > 0;
    if (!has_sp) {
        for (const auto& [tok, _] : vocab_) {
            if (tok.size() >= sp_space.size() && tok.compare(0, sp_space.size(), sp_space) == 0) {
                has_sp = true;
                break;
            }
        }
    }
    if (!has_gpt && has_sp) {
        byte_level_ = false;
        sentencepiece_ = true;
    }

    const char* unk_candidates[] = {"<unk>", "<|unk|>", "[UNK]", "<|endoftext|>"};
    for (const char* cand : unk_candidates) {
        auto it = vocab_.find(cand);
        if (it != vocab_.end()) {
            unk_id_ = it->second;
            break;
        }
    }
}

Tokenizer Tokenizer::load_from_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw IOError("Cannot open tokenizer file: " + path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();

    JsonValue root;
    try {
        root = parse_json(oss.str());
    } catch (const std::exception& e) {
        throw IOError("Failed to parse tokenizer JSON '" + path + "': " + e.what());
    }
    if (!root.is_object()) {
        throw IOError("Invalid tokenizer.json root object in: " + path);
    }

    Tokenizer tok;
    tok.byte_level_ = true;
    tok.sentencepiece_ = false;

    if (root.has("pre_tokenizer") && root["pre_tokenizer"].is_object()) {
        const auto& pt = root["pre_tokenizer"];
        if (pt.has("type") && pt["type"].is_string()) {
            const auto& ptype = pt["type"].as_string();
            if (ptype == "Metaspace") {
                tok.byte_level_ = false;
                tok.sentencepiece_ = true;
            } else if (ptype == "ByteLevel") {
                tok.byte_level_ = true;
                tok.sentencepiece_ = false;
            }
        }
    }

    if (root.has("added_tokens") && root["added_tokens"].is_array()) {
        for (const auto& item : root["added_tokens"].as_array()) {
            if (!item.is_object()) continue;
            if (!item.has("id") || !item.has("content")) continue;
            AddedToken at;
            at.id = static_cast<int>(item["id"].as_int());
            at.content = item["content"].as_string();
            at.special = item.has("special") && item["special"].is_bool() ? item["special"].as_bool() : false;
            tok.added_tokens_.push_back(std::move(at));
        }
    }

    if (root.has("model") && root["model"].is_object()) {
        const auto& model = root["model"];
        if (model.has("vocab") && model["vocab"].is_object()) {
            for (const auto& [k, v] : model["vocab"].as_object()) {
                tok.vocab_[k] = static_cast<int>(v.as_int());
            }
        }
        if (model.has("merges") && model["merges"].is_array()) {
            for (const auto& m : model["merges"].as_array()) {
                if (m.is_string()) {
                    const std::string& ms = m.as_string();
                    auto sp = ms.find(' ');
                    if (sp != std::string::npos) {
                        tok.merges_.emplace_back(ms.substr(0, sp), ms.substr(sp + 1));
                    }
                } else if (m.is_array()) {
                    const auto& arr = m.as_array();
                    if (arr.size() >= 2 && arr[0].is_string() && arr[1].is_string()) {
                        tok.merges_.emplace_back(arr[0].as_string(), arr[1].as_string());
                    }
                }
            }
        }
    }

    tok.rebuild_tables();
    return tok;
}

Tokenizer Tokenizer::create_byte_level_demo(int vocab_size) {
    if (vocab_size <= 0) {
        throw IOError("Tokenizer vocab_size must be positive");
    }
    Tokenizer tok;
    tok.byte_level_ = true;
    tok.sentencepiece_ = false;

    const auto& tbl = byte_unicode_table();
    if (vocab_size < 256) {
        std::vector<int> priority_bytes;
        priority_bytes.push_back(static_cast<int>(' '));
        for (int c = 'a'; c <= 'z'; ++c) priority_bytes.push_back(c);
        for (int c = 'A'; c <= 'Z'; ++c) priority_bytes.push_back(c);
        for (int c = '0'; c <= '9'; ++c) priority_bytes.push_back(c);
        const char* extra = ".,!?;:'\"-\n\t()[]{}";
        for (const char* p = extra; *p != '\0'; ++p) {
            priority_bytes.push_back(static_cast<unsigned char>(*p));
        }
        std::array<bool, 256> seen{};
        int next_id = 0;
        for (int b : priority_bytes) {
            if (next_id >= vocab_size) break;
            if (!seen[static_cast<std::size_t>(b)]) {
                seen[static_cast<std::size_t>(b)] = true;
                tok.vocab_[tbl.b2u[static_cast<std::size_t>(b)]] = next_id++;
            }
        }
        for (int b = 0; b < 256 && next_id < vocab_size; ++b) {
            if (!seen[static_cast<std::size_t>(b)]) {
                seen[static_cast<std::size_t>(b)] = true;
                tok.vocab_[tbl.b2u[static_cast<std::size_t>(b)]] = next_id++;
            }
        }
        tok.rebuild_tables();
        return tok;
    }

    int next_id = 0;
    for (int b = 0; b < 256; ++b) {
        tok.vocab_[tbl.b2u[static_cast<std::size_t>(b)]] = next_id++;
    }

    const char* specials[] = {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"};
    for (const char* sp : specials) {
        if (next_id >= vocab_size) break;
        tok.added_tokens_.push_back(AddedToken{next_id, sp, true});
        ++next_id;
    }

    const std::pair<std::string, std::string> common_merges[] = {
        {"t", "h"}, {"h", "e"}, {"i", "n"}, {"e", "r"}, {"a", "n"}, {"r", "e"},
        {"o", "n"}, {"a", "t"}, {"e", "n"}, {"\xC4\xA0", "t"}, {"\xC4\xA0", "a"},
        {"\xC4\xA0", "i"}, {"\xC4\xA0", "o"}, {"\xC4\xA0t", "h"}, {"\xC4\xA0th", "e"},
        {"i", "s"}, {"t", "o"}, {"o", "f"}, {"a", "l"}, {"l", "e"},
        {"t", "e"}, {"e", "d"}, {"i", "t"}, {"a", "r"}, {"s", "t"},
        {"n", "d"}, {"t", "i"}, {"e", "s"}, {"o", "r"}, {"\xC4\xA0", "s"},
        {"\xC4\xA0", "c"}, {"\xC4\xA0", "p"}, {"\xC4\xA0", "w"}, {"\xC4\xA0", "m"},
        {"i", "ng"}, {"n", "g"}, {"th", "e"}, {"a", "nd"}, {"f", "or"},
        {"\xC4\xA0", "b"}, {"\xC4\xA0", "d"}, {"\xC4\xA0", "f"}, {"\xC4\xA0", "h"}
    };
    for (const auto& [lhs, rhs] : common_merges) {
        if (next_id >= vocab_size) break;
        std::string merged = lhs + rhs;
        if (tok.vocab_.count(lhs) && tok.vocab_.count(rhs) && tok.vocab_.count(merged) == 0) {
            tok.merges_.emplace_back(lhs, rhs);
            tok.vocab_[merged] = next_id++;
        }
    }

    int extra_idx = 0;
    while (next_id < vocab_size) {
        std::string syn = "<|tok_" + std::to_string(extra_idx++) + "|>";
        if (tok.vocab_.count(syn) == 0) {
            tok.vocab_[syn] = next_id++;
        }
    }

    tok.rebuild_tables();
    return tok;
}

void Tokenizer::save(const std::string& path) const {
    JsonValue root = JsonValue::object();
    root["version"] = "1.0";

    JsonValue added = JsonValue::array();
    std::map<std::string, bool> special_set;
    for (const auto& at : added_tokens_) {
        JsonValue entry = JsonValue::object();
        entry["id"] = at.id;
        entry["content"] = at.content;
        entry["special"] = at.special;
        added.as_array().push_back(std::move(entry));
        special_set[at.content] = true;
    }
    root["added_tokens"] = std::move(added);
    root["normalizer"] = nullptr;

    if (byte_level_) {
        JsonValue bl = JsonValue::object();
        bl["type"] = "ByteLevel";
        bl["add_prefix_space"] = false;
        bl["trim_offsets"] = true;
        bl["use_regex"] = true;
        root["pre_tokenizer"] = bl;
        root["decoder"] = bl;
    } else {
        JsonValue ms = JsonValue::object();
        ms["type"] = "Metaspace";
        root["pre_tokenizer"] = ms;
        root["decoder"] = ms;
    }

    JsonValue model = JsonValue::object();
    model["type"] = "BPE";

    JsonValue vocab_obj = JsonValue::object();
    for (const auto& [tok, id] : vocab_) {
        if (!special_set.count(tok)) {
            vocab_obj[tok] = id;
        }
    }
    model["vocab"] = std::move(vocab_obj);

    JsonValue merges_arr = JsonValue::array();
    for (const auto& [lhs, rhs] : merges_) {
        JsonValue pair_arr = JsonValue::array();
        pair_arr.as_array().push_back(lhs);
        pair_arr.as_array().push_back(rhs);
        merges_arr.as_array().push_back(std::move(pair_arr));
    }
    model["merges"] = std::move(merges_arr);
    root["model"] = std::move(model);

    std::ofstream ofs(path);
    if (!ofs) {
        throw IOError("Cannot open tokenizer output file: " + path);
    }
    ofs << dump_json(root, 2) << "\n";
    if (!ofs) {
        throw IOError("Failed writing tokenizer file: " + path);
    }
}

std::vector<int> Tokenizer::bpe_encode_token(const std::vector<std::string>& initial_syms,
                                             std::string_view raw_bytes) const {
    if (initial_syms.empty()) {
        return {};
    }
    std::vector<std::string> word = initial_syms;
    while (word.size() >= 2 && !merge_ranks_.empty()) {
        int best_rank = std::numeric_limits<int>::max();
        std::size_t best_idx = std::string::npos;
        for (std::size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = merge_ranks_.find({word[i], word[i + 1]});
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_idx == std::string::npos) {
            break;
        }
        const std::string left = word[best_idx];
        const std::string right = word[best_idx + 1];
        const std::string merged = left + right;
        std::vector<std::string> next_word;
        next_word.reserve(word.size());
        std::size_t i = 0;
        while (i < word.size()) {
            if (i + 1 < word.size() && word[i] == left && word[i + 1] == right) {
                next_word.push_back(merged);
                i += 2;
            } else {
                next_word.push_back(std::move(word[i]));
                ++i;
            }
        }
        word = std::move(next_word);
    }

    std::vector<int> ids;
    ids.reserve(word.size());
    const int vs = vocab_size();
    for (std::size_t idx = 0; idx < word.size(); ++idx) {
        const auto& sym = word[idx];
        auto it = vocab_.find(sym);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
            continue;
        }
        if (sentencepiece_) {
            bool pushed_bytes = false;
            for (unsigned char b : sym) {
                std::string hx = byte_hex_token(b);
                auto hit = vocab_.find(hx);
                if (hit != vocab_.end()) {
                    ids.push_back(hit->second);
                    pushed_bytes = true;
                }
            }
            if (pushed_bytes) {
                continue;
            }
        }
        if (unk_id_ >= 0) {
            ids.push_back(unk_id_);
        } else if (vs > 0) {
            unsigned char fallback_b = idx < raw_bytes.size()
                                           ? static_cast<unsigned char>(raw_bytes[idx])
                                           : static_cast<unsigned char>(sym[0]);
            ids.push_back(static_cast<int>(fallback_b) % vs);
        }
    }
    return ids;
}

std::vector<int> Tokenizer::encode_plain_segment(std::string_view seg) const {
    if (seg.empty()) {
        return {};
    }
    std::vector<int> out;
    if (byte_level_) {
        const auto& tbl = byte_unicode_table();
        const bool has_gpt_space = vocab_.count("\xC4\xA0") > 0;
        const bool has_raw_space = vocab_.count(" ") > 0;
        const bool use_gpt_map = has_gpt_space || !has_raw_space;

        auto chunks = pretokenize_byte_level(seg);
        for (const auto& chunk : chunks) {
            std::vector<std::string> syms;
            syms.reserve(chunk.size());
            for (unsigned char b : chunk) {
                if (!use_gpt_map && b < 128 && vocab_.count(std::string(1, static_cast<char>(b))) > 0) {
                    syms.emplace_back(1, static_cast<char>(b));
                } else {
                    syms.push_back(tbl.b2u[static_cast<std::size_t>(b)]);
                }
            }
            auto chunk_ids = bpe_encode_token(syms, chunk);
            out.insert(out.end(), chunk_ids.begin(), chunk_ids.end());
        }
        return out;
    }

    const std::string sp_space = "\xE2\x96\x81";
    std::string transformed;
    transformed.reserve(seg.size() + 8);
    if (!seg.empty() && seg[0] != ' ') {
        std::string first_prefixed = sp_space;
        std::size_t p = 0;
        std::uint32_t cp = 0;
        std::size_t bl = 0;
        if (next_utf8_cp(seg, p, cp, bl)) {
            first_prefixed.append(seg.substr(0, bl));
        }
        for (const auto& [k, _] : vocab_) {
            if (k.size() >= first_prefixed.size() && k.compare(0, first_prefixed.size(), first_prefixed) == 0) {
                transformed += sp_space;
                break;
            }
        }
    }
    for (char c : seg) {
        if (c == ' ') {
            transformed += sp_space;
        } else {
            transformed.push_back(c);
        }
    }

    std::vector<std::string> syms;
    std::size_t pos = 0;
    while (pos < transformed.size()) {
        std::size_t start = pos;
        std::uint32_t cp = 0;
        std::size_t bl = 0;
        next_utf8_cp(transformed, pos, cp, bl);
        syms.emplace_back(transformed.substr(start, bl));
    }
    return bpe_encode_token(syms, seg);
}

std::vector<int> Tokenizer::encode(std::string_view text) const {
    if (text.empty()) {
        return {};
    }
    std::vector<int> out;
    std::size_t seg_start = 0;
    std::size_t i = 0;
    while (i < text.size()) {
        bool matched = false;
        for (const auto& at : added_tokens_) {
            if (at.content.empty()) continue;
            if (i + at.content.size() <= text.size() &&
                text.substr(i, at.content.size()) == at.content) {
                if (i > seg_start) {
                    auto seg_ids = encode_plain_segment(text.substr(seg_start, i - seg_start));
                    out.insert(out.end(), seg_ids.begin(), seg_ids.end());
                }
                out.push_back(at.id);
                i += at.content.size();
                seg_start = i;
                matched = true;
                break;
            }
        }
        if (!matched) {
            ++i;
        }
    }
    if (seg_start < text.size()) {
        auto seg_ids = encode_plain_segment(text.substr(seg_start));
        out.insert(out.end(), seg_ids.begin(), seg_ids.end());
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    const auto& tbl = byte_unicode_table();
    std::string out;
    const std::string sp_space = "\xE2\x96\x81";

    for (std::size_t idx = 0; idx < ids.size(); ++idx) {
        int id = ids[idx];
        auto it = id_to_token_.find(id);
        if (it == id_to_token_.end()) {
            continue;
        }
        const std::string& tok = it->second;
        if (added_ids_.count(id)) {
            out += tok;
            continue;
        }
        if (byte_level_) {
            std::size_t pos = 0;
            while (pos < tok.size()) {
                std::size_t start = pos;
                std::uint32_t cp = 0;
                std::size_t bl = 0;
                next_utf8_cp(tok, pos, cp, bl);
                auto bit = tbl.u2b.find(cp);
                if (bit != tbl.u2b.end()) {
                    out.push_back(static_cast<char>(bit->second));
                } else {
                    out.append(tok, start, bl);
                }
            }
        } else {
            unsigned char hx_byte = 0;
            if (try_parse_hex_byte_token(tok, hx_byte)) {
                out.push_back(static_cast<char>(hx_byte));
                continue;
            }
            std::size_t pos = 0;
            while (pos < tok.size()) {
                if (tok.compare(pos, sp_space.size(), sp_space) == 0) {
                    if (!(idx == 0 && pos == 0 && out.empty())) {
                        out.push_back(' ');
                    }
                    pos += sp_space.size();
                } else {
                    out.push_back(tok[pos++]);
                }
            }
        }
    }
    return out;
}

int Tokenizer::vocab_size() const {
    return static_cast<int>(id_to_token_.size());
}

const std::map<std::string, int>& Tokenizer::vocab() const {
    return vocab_;
}

}
