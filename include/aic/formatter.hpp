#pragma once

#include "aic/ast.hpp"

#include <string>
#include <string_view>

namespace aic {

std::string format_ast(const ProgramAST& ast);
std::string format_strata_source(std::string_view source, const std::string& filename = "<input>");

}
