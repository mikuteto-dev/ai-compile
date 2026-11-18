#pragma once

#include "aic/ast.hpp"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace aic {

struct ResolvedSymbol {
    std::string name;
    std::string kind;
    StageSpec frame;
    std::string frame_name;
    int dim = 1;
    std::string topology = "Euclidean";
    double curvature = 0.0;
    int rank = 1;
    std::string metric = "fisher";
    std::map<std::string, std::string> poles;
    std::vector<AnchorDecl> anchors;
    std::vector<RodataEntryAST> rodata_entries;
    std::vector<std::pair<std::string, std::string>> connections;
};

struct ResolvedFlow {
    FlowDecl decl;
    StageSpec stage;
    std::string target_mode_hint;
    double crosstalk_bound = 1e-2;
    double approx_bound = 1e-1;
    double energy_bound = 1.0;
    std::vector<std::string> preserve;
    bool has_crosstalk_constraint = false;
    bool has_approx_constraint = false;
    bool has_energy_constraint = false;
    bool is_helper = false;
};

struct TypedProgram {
    ProgramAST ast;
    std::string module_name;
    std::vector<std::string> imports;
    std::map<std::string, ResolvedSymbol> symbols;
    std::vector<ResolvedFlow> flows;
    std::vector<TestDecl> tests;
    std::map<std::string, double> const_numbers;
    std::map<std::string, std::string> const_strings;
    std::map<std::string, bool> const_bools;
    std::map<std::string, std::vector<std::string>> const_lists;
};

bool is_subtype(const std::string& sub, const std::string& sup);
bool stages_overlap(const StageSpec& a, const StageSpec& b);
TypedProgram typecheck_program(const ProgramAST& ast);

}
