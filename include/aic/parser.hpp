#pragma once

#include "aic/ast.hpp"
#include "aic/lexer.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace aic {

ProgramAST parse_tokens(const std::vector<Token>& tokens, const std::string& filename = "<input>");
ProgramAST parse_strata(std::string_view src, const std::string& filename = "<input>");

}
