#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace aic {

struct LintDiagnostic {
    std::string code;
    std::string message;
    int line = 0;
    int col = 0;
};

std::vector<LintDiagnostic> lint_source(std::string_view source,
                                        const std::string& filename = "<input>");
std::vector<LintDiagnostic> lint_file_with_imports(const std::string& entry_path,
                                                   const std::vector<std::string>& include_dirs = {});

}
