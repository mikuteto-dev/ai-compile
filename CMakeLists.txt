cmake_minimum_required(VERSION 3.20)
project(ai_compile LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -O2)

enable_testing()

file(GLOB AIC_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM AIC_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

if(APPLE)
    enable_language(OBJCXX)
    list(APPEND AIC_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/accel_metal.mm")
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/accel_metal.mm"
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
endif()

add_library(aic_core STATIC ${AIC_CORE_SOURCES})
target_include_directories(aic_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(APPLE)
    target_compile_definitions(aic_core PUBLIC ACCELERATE_NEW_LAPACK)
    target_link_libraries(aic_core PUBLIC
        "-framework Accelerate"
        "-framework Metal"
        "-framework Foundation"
    )
endif()

add_executable(aic src/main.cpp)
target_link_libraries(aic PRIVATE aic_core)

add_executable(test_tensor_runtime tests/test_tensor_runtime.cpp)
target_link_libraries(test_tensor_runtime PRIVATE aic_core)
add_test(NAME test_tensor_runtime COMMAND test_tensor_runtime WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_frontend tests/test_frontend.cpp)
target_link_libraries(test_frontend PRIVATE aic_core)
add_test(NAME test_frontend COMMAND test_frontend WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_backend tests/test_backend.cpp)
target_link_libraries(test_backend PRIVATE aic_core)
add_test(NAME test_backend COMMAND test_backend WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_nelf_lift_link tests/test_nelf_lift_link.cpp)
target_link_libraries(test_nelf_lift_link PRIVATE aic_core)
add_test(NAME test_nelf_lift_link COMMAND test_nelf_lift_link WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(test_cli_e2e tests/test_cli_e2e.cpp)
target_link_libraries(test_cli_e2e PRIVATE aic_core)
add_dependencies(test_cli_e2e aic)
target_compile_definitions(test_cli_e2e PRIVATE
    AIC_BIN_PATH="$<TARGET_FILE:aic>"
    AIC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
add_test(NAME test_cli_e2e COMMAND test_cli_e2e WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
