#pragma once

#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aic {

struct AddedToken {
    int id = 0;
    std::string content;
    bool special = false;
};

class Tokenizer {
public:
    Tokenizer() = default;

    static Tokenizer load_from_file(const std::string& path);
    static Tokenizer create_byte_level_demo(int vocab_size = 256);

    void save(const std::string& path) const;

    std::vector<int> encode(std::string_view text) const;
    std::string decode(const std::vector<int>& ids) const;

    int vocab_size() const;
    const std::map<std::string, int>& vocab() const;

private:
    std::map<std::string, int> vocab_;
    std::map<int, std::string> id_to_token_;
    std::vector<AddedToken> added_tokens_;
    std::map<int, bool> added_ids_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::map<std::pair<std::string, std::string>, int> merge_ranks_;
    bool byte_level_ = true;
    bool sentencepiece_ = false;
    int unk_id_ = -1;

    void rebuild_tables();
    std::vector<int> encode_plain_segment(std::string_view seg) const;
    std::vector<int> bpe_encode_token(const std::vector<std::string>& initial_syms,
                                      std::string_view raw_bytes) const;
};

}
