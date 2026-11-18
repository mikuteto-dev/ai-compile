#pragma once

#include "aic/typecheck.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace aic {

struct MIRStageSpec {
    std::string start = "All";
    std::string end = "";
    bool inclusive = false;
};

using ArgValue = std::variant<double, std::int64_t, bool, std::string, std::vector<std::string>>;

struct MIRAnchor {
    std::string kind;
    std::map<std::string, ArgValue> named_args;
    std::vector<ArgValue> pos_args;
};

struct MIRRodataEntry {
    std::string key;
    std::string value;
    std::map<std::string, double> attrs;
};

struct MIRSymbolDecl {
    std::string name;
    std::string kind;
    MIRStageSpec frame;
    int dim = 1;
    std::string topology = "Euclidean";
    double curvature = 0.0;
    int rank = 1;
    std::string metric = "fisher";
    std::map<std::string, std::string> poles;
    std::vector<MIRAnchor> anchors;
    std::vector<MIRRodataEntry> rodata_entries;
    std::vector<std::pair<std::string, std::string>> connections;
};

struct MIREffectContract {
    double crosstalk_bound = 1e-2;
    double approx_bound = 1e-1;
    double energy_bound = 1.0;
    std::vector<std::string> preserve;
};

struct MIROp {
    std::string op;
    std::string dst;
    std::string target;
    std::string pole;
    std::string rhs_target;
    std::map<std::string, ArgValue> args;
    double scalar_arg = 1.0;
    std::string scalar_gate;
    std::vector<MIROp> body;
    std::vector<MIROp> else_body;
    MIRStageSpec src_frame;
    MIRStageSpec dst_frame;
};

struct MIRFlow {
    std::string name;
    MIRStageSpec stage;
    std::string target_mode_hint;
    MIREffectContract contract;
    std::vector<MIROp> ops;
    bool is_helper = false;
};

struct MIRExpectation {
    std::string kind;
    std::string prompt;
    std::vector<std::string> prompts;
    std::string symbol;
    std::string pole;
    std::string cmp_op;
    bool rhs_is_base = false;
    double rhs_val = 0.0;
    std::string target_token_str;
    std::string foil_token_str;
};

struct MIRTestCase {
    std::string name;
    std::vector<MIRExpectation> expectations;
};

struct MIRModule {
    std::string name;
    std::vector<std::string> imports;
    std::map<std::string, MIRSymbolDecl> declarations;
    std::vector<MIRFlow> flows;
    std::vector<MIRTestCase> tests;
};

MIRModule lower_to_mir(const TypedProgram& prog);
MIRModule compile_source_to_mir(std::string_view source, const std::string& filename = "<input>");
MIRModule compile_file_with_imports(const std::string& entry_path,
                                    const std::vector<std::string>& include_dirs = {});
std::string format_mir(const MIRModule& mod);

}
