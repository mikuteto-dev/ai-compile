#include "aic/errors.hpp"
#include "aic/formatter.hpp"
#include "aic/json_min.hpp"
#include "aic/lexer.hpp"
#include "aic/linter.hpp"
#include "aic/mir.hpp"
#include "aic/parser.hpp"
#include "aic/tokenizer.hpp"
#include "aic/typecheck.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace aic;

namespace {

template <typename Err, typename Fn>
void expect_throws(Fn&& fn, const std::string& label) {
    bool caught = false;
    try {
        fn();
    } catch (const Err&) {
        caught = true;
    } catch (const std::exception& ex) {
        std::cerr << "FAIL [" << label << "]: caught wrong exception: " << ex.what() << "\n";
        assert(false);
    }
    if (!caught) {
        std::cerr << "FAIL [" << label << "]: did not throw expected exception\n";
        assert(false);
    }
}

void test_lexer_lookahead_and_comments() {
    auto toks = tokenize("10..15 10..=15 1e-3 19.5");
    assert(toks.size() >= 8);
    assert(toks[0].kind == TokenKind::Int && toks[0].int_val == 10);
    assert(toks[1].kind == TokenKind::DotDot);
    assert(toks[2].kind == TokenKind::Int && toks[2].int_val == 15);
    assert(toks[3].kind == TokenKind::Int && toks[3].int_val == 10);
    assert(toks[4].kind == TokenKind::DotDotEq);
    assert(toks[5].kind == TokenKind::Int && toks[5].int_val == 15);
    assert(toks[6].kind == TokenKind::Float && toks[6].float_val > 0.0009 && toks[6].float_val < 0.0011);
    assert(toks[7].kind == TokenKind::Float && toks[7].float_val > 19.49 && toks[7].float_val < 19.51);

    std::string line_cmt = std::string(2, '/') + " comment here\npole P in Frame<Mid>;\n";
    std::string block_open = std::string(1, '/') + "*";
    std::string block_close = "*" + std::string(1, '/');
    std::string with_comments = line_cmt + block_open + " multi\nline " + block_close + "pole Q in Frame<Mid>;";
    auto m = compile_source_to_mir(with_comments);
    assert(m.declarations.count("P") == 1);
    assert(m.declarations.count("Q") == 1);

    expect_throws<LexError>([&]() {
        tokenize(block_open + " unterminated");
    }, "unterminated block comment");
    expect_throws<LexError>([]() {
        tokenize("\"unterminated string");
    }, "unterminated string");
    expect_throws<LexError>([]() {
        tokenize("pole $bad in Frame<Mid>;");
    }, "unexpected char");
}

void test_case_study_1() {
    const std::string src = R"(
module MathLoopSurgery;

import std::CoreReasoning;
import std::Fluency;

frame MathReasonMid in 16..24 {
    connect to LogitFrame via StandardResidualBridge;
}

@stage(16..24)
dipole MathLoopAnomaly in Frame<16..24> {
    +pole: RepetitionSpiral {
        anchor contrastive(
            pos: ["Wait, let me double check. Wait, let me double check."],
            neg: ["Therefore, x = 5. Moving to the next equation:"]
        );
    }
    -pole: StepProgression {
        anchor probe("Step 2: Substituting into equation (1) yields:");
    }
}

@target("in-place")
flow SuppressRunawayLoop(state: Residual in Frame<16..24>) -> Residual in Frame<16..24>
    ![crosstalk <= 1e-3, approx <= 0.0, energy <= 0.05, preserve(CoreReasoning, Fluency, Residual)]
{
    when state ~> MathLoopAnomaly.+pole(threshold: 0.72) as amp {
        state -| MathLoopAnomaly.+pole(scale: 1.0);
        state +> MathLoopAnomaly.-pole(gain: 0.45 * amp);
    }
}
)";
    MIRModule mir = compile_source_to_mir(src, "01_math_loop_surgery.strata");
    assert(mir.name == "MathLoopSurgery");
    assert(mir.imports.size() == 2);
    assert(mir.declarations.at("MathReasonMid").connections.size() == 1);
    assert(mir.declarations.at("MathLoopAnomaly").kind == "dipole");
    assert(mir.declarations.at("MathLoopAnomaly").poles.at("+pole") == "RepetitionSpiral");
    assert(mir.declarations.at("MathLoopAnomaly").poles.at("-pole") == "StepProgression");
    assert(mir.declarations.at("MathLoopAnomaly").anchors.size() == 2);
    assert(mir.flows.size() == 1);
    const auto& flow = mir.flows[0];
    assert(flow.target_mode_hint == "in-place");
    assert(flow.stage.start == "16" && flow.stage.end == "24");
    assert(flow.ops.size() == 1);
    assert(flow.ops[0].op == "when");
    assert(flow.ops[0].dst == "amp");
    assert(flow.ops[0].target == "MathLoopAnomaly");
    assert(flow.ops[0].pole == "+pole");
    assert(flow.ops[0].body.size() == 2);
    assert(flow.ops[0].body[0].op == "annihilate");
    assert(flow.ops[0].body[1].op == "excite");
    assert(flow.ops[0].body[1].scalar_arg > 0.44 && flow.ops[0].body[1].scalar_arg < 0.46);
    assert(flow.ops[0].body[1].scalar_gate == "amp");
    std::string dumped = format_mir(mir);
    assert(dumped.find("mir.module @MathLoopSurgery") != std::string::npos);
}

void test_case_study_2() {
    const std::string src = R"(
module CyclicWeekdayShift;

import std::Attention::TemporalIndex;
import std::CoreFluency;

@stage(8..14)
manifold Weekday : S1 in Frame<Mid> {
    curvature 1.0;
    chart angle: "radians_mod_2pi";
    anchor cyclic(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]);
}

subspace ShiftDelta<1> in Frame<Mid> {
    anchor probe("in {k} days");
}

pole WeekdayQueryTag in Frame<Mid> {
    anchor probe("What day of the week is it");
}

flow ComputeShiftedWeekday(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 5e-4, approx <= 1e-2, energy <= 0.12, preserve(CoreFluency, Residual)]
{
    let delta = @attend(
        query: state,
        key: WeekdayQueryTag,
        value: ShiftDelta,
        mask: "causal"
    );

    when state ~> Weekday(chart: "angle") as theta {
        let shifted = state => Weekday.rotate(angle: theta, delta: delta);
        state -| Weekday;
        state +> shifted(gain: 1.0);
    }
}
)";
    MIRModule mir = compile_source_to_mir(src, "02_cyclic_weekday_s1.strata");
    assert(mir.name == "CyclicWeekdayShift");
    assert(mir.declarations.at("Weekday").kind == "manifold");
    assert(mir.declarations.at("Weekday").topology == "S1");
    assert(mir.declarations.at("Weekday").curvature == 1.0);
    assert(mir.declarations.at("ShiftDelta").dim == 1);
    assert(mir.flows[0].ops.size() == 2);
    assert(mir.flows[0].ops[0].op == "attend");
    assert(mir.flows[0].ops[0].dst == "delta");
    assert(mir.flows[0].ops[1].op == "when");
    assert(mir.flows[0].ops[1].dst == "theta");
    assert(mir.flows[0].ops[1].body.size() == 3);
    assert(mir.flows[0].ops[1].body[0].op == "transport_map");
    assert(mir.flows[0].ops[1].body[0].target == "Weekday");
    assert(mir.flows[0].ops[1].body[0].rhs_target == "rotate");
    assert(mir.flows[0].ops[1].body[1].op == "annihilate");
    assert(mir.flows[0].ops[1].body[2].op == "excite");
}

void test_case_study_3() {
    const std::string src = R"(
module TacitBehaviorSculpting;

import std::CoreFluency;

field EmpatheticTone : LowCurvature in Frame<All> {
    rank 16;
    metric Fisher;
    anchor spectral(
        prompts: ["I understand how difficult this must feel for you."],
        curvature_band: "low"
    );
}

field SycophancyDrift : Mixed in Frame<Mid..Late> {
    rank 12;
    metric Fisher;
    anchor contrastive(
        pos: ["You are completely right, 2 + 2 = 5 if you prefer!"],
        neg: ["I appreciate your perspective, however 2 + 2 equals 4."]
    );
}

field RigorousReasoning : LowCurvature in Frame<Mid..Late> {
    rank 16;
    metric Fisher;
    anchor spectral(prompts: ["Let us check each premise step by step."], curvature_band: "low");
}

flow SculptTacitPriors ![crosstalk <= 2e-3, approx <= 1e-3, energy <= 0.10, preserve(EmpatheticTone, CoreFluency, Residual)] {
    let pure_sycophancy = SycophancyDrift decouple EmpatheticTone with (metric: "fisher", reg: 1e-4);
    state -| pure_sycophancy(scale: 0.85);
    let calm_reason = temper(RigorousReasoning, band: "low_curvature", gain: 1.15);
    let blended = blend(EmpatheticTone, calm_reason, alpha: 0.25);
    state +> blended(gain: 0.20);
}
)";
    MIRModule mir = compile_source_to_mir(src, "03_tacit_field_decouple.strata");
    assert(mir.name == "TacitBehaviorSculpting");
    assert(mir.declarations.at("EmpatheticTone").rank == 16);
    assert(mir.declarations.at("EmpatheticTone").metric == "fisher");
    assert(mir.flows[0].ops.size() == 5);
    assert(mir.flows[0].ops[0].op == "field_decouple");
    assert(mir.flows[0].ops[0].target == "SycophancyDrift");
    assert(mir.flows[0].ops[0].rhs_target == "EmpatheticTone");
    assert(mir.flows[0].ops[1].op == "annihilate");
    assert(mir.flows[0].ops[2].op == "field_temper");
    assert(mir.flows[0].ops[3].op == "field_blend");
    assert(mir.flows[0].ops[4].op == "excite");
}

void test_case_study_4() {
    const std::string src = R"(
module FactSurgeryAndCrossTransplant;

import std::CoreReasoning;
import std::CoreFluency;

rodata CapitalKnowledge<Entity, City> in Frame<Mid..Late> {
    anchor factual("The official capital of {subject} is {object}.");
    entry "Aethelgard" => "Oldhaven" with (curvature: 24.8);
    entry "Valoria" => "Solis" with (curvature: 19.2);
}

subspace FactConsistencyCheck<4> in Frame<Mid..Late> {
    anchor contrastive(
        pos: ["Verify geographical entity before answering."],
        neg: ["Guess any plausible city name immediately."]
    );
}

flow UpdateAndTransplant(state: Residual in Frame<Mid..Late>) -> Residual in Frame<Mid..Late>
    ![crosstalk <= 1e-3, approx <= 5e-3, energy <= 0.08, preserve(CoreReasoning, CoreFluency, Residual)]
{
    rewrite CapitalKnowledge(key: "Aethelgard", old: "Oldhaven", new: "Newhaven");
    prune CapitalKnowledge(key: "DeprecatedCityRecord");
    insert CapitalKnowledge(key: "Veridia", value: "Celestia", sharpness: 15.0);

    let verified = transport(state, Frame<Mid..Late> -> Frame<Late>);
    when verified ~> FactConsistencyCheck(threshold: 0.65) {
        state +> FactConsistencyCheck(gain: 0.35);
    }
}
)";
    MIRModule mir = compile_source_to_mir(src, "04_rodata_and_transplant.strata");
    assert(mir.name == "FactSurgeryAndCrossTransplant");
    assert(mir.declarations.at("CapitalKnowledge").rodata_entries.size() == 2);
    assert(mir.declarations.at("CapitalKnowledge").rodata_entries[0].attrs.at("curvature") == 24.8);
    assert(mir.declarations.at("FactConsistencyCheck").dim == 4);
    assert(mir.flows[0].ops.size() == 5);
    assert(mir.flows[0].ops[0].op == "rodata_rewrite");
    assert(mir.flows[0].ops[1].op == "rodata_prune");
    assert(mir.flows[0].ops[2].op == "rodata_insert");
    assert(mir.flows[0].ops[3].op == "frame_transport");
    assert(mir.flows[0].ops[3].dst_frame.start == "Late");
    assert(mir.flows[0].ops[4].op == "when");
}

void test_unicode_operators() {
    const std::string src =
        "module UnicodeTest;\n"
        "manifold Ring : S1 in Frame<Mid> {\n"
        "    curvature 1.0;\n"
        "    chart angle: \"radians\";\n"
        "}\n"
        "pole KeyP in Frame<Mid>;\n"
        "subspace ValS<2> in Frame<Mid>;\n"
        "field F1 in Frame<All>;\n"
        "field F2 in Frame<Mid>;\n"
        "flow Uni(state: Residual in Frame<Mid>) \u2192 Residual in Frame<Mid>\n"
        "    \u27E6crosstalk <= 1e-2, approx <= 1e-1, energy <= 1.0, preserve(F1)\u27E7\n"
        "{\n"
        "    let d = \u22C6(query: state, key: KeyP, value: ValS, mask: \"causal\");\n"
        "    let t = transport(state, Frame<Mid> \u2192 Frame<Mid>);\n"
        "    when t \u25B7 Ring(chart: \"angle\") as th {\n"
        "        let sh = t \u21A6 Ring.rotate(angle: th, delta: d);\n"
        "        state \u22A3 Ring;\n"
        "        state \u2295 sh(gain: 0.5);\n"
        "    }\n"
        "    let pure = F2 \u22A5 F1(metric: \"fisher\");\n"
        "    state \u22A3 pure(scale: 0.4);\n"
        "}\n";

    MIRModule mir = compile_source_to_mir(src, "unicode.strata");
    assert(mir.flows.size() == 1);
    const auto& ops = mir.flows[0].ops;
    assert(ops.size() == 5);
    assert(ops[0].op == "attend");
    assert(ops[1].op == "frame_transport");
    assert(ops[2].op == "when");
    assert(ops[2].body[0].op == "transport_map");
    assert(ops[2].body[1].op == "annihilate");
    assert(ops[2].body[2].op == "excite");
    assert(ops[3].op == "field_decouple");
    assert(ops[4].op == "annihilate");
}

void test_subtype_lattice() {
    assert(is_subtype("Pole", "Pole"));
    assert(is_subtype("Pole", "Dipole"));
    assert(is_subtype("Pole", "Subspace<1>"));
    assert(is_subtype("Pole", "Subspace<4>"));
    assert(is_subtype("Dipole", "Subspace<1>"));
    assert(is_subtype("Subspace<4>", "Manifold<Euclidean, 4>"));
    assert(is_subtype("Pole", "Manifold<Euclidean, 1>"));
    assert(!is_subtype("Manifold<S1, 1>", "Pole"));
    assert(!is_subtype("Field", "Pole"));
    assert(!is_subtype("Subspace<8>", "Subspace<4>"));
}

void test_type_errors() {
    expect_throws<FrameMismatchError>([]() {
        compile_source_to_mir(
            "pole P in Frame<Early>;\n"
            "flow Bad(state: Residual in Frame<Late>) {\n"
            "    state +> P(gain: 0.5);\n"
            "}\n");
    }, "disjoint named frames");

    expect_throws<FrameMismatchError>([]() {
        compile_source_to_mir(
            "pole P in Frame<0..4>;\n"
            "flow Bad(state: Residual in Frame<10..15>) {\n"
            "    state +> P(gain: 0.5);\n"
            "}\n");
    }, "disjoint numeric frames");

    compile_source_to_mir(
        "frame EarlyF in Early { connect to LateF via Bridge; }\n"
        "frame LateF in Late;\n"
        "pole P in EarlyF;\n"
        "flow Good(state: Residual in LateF) {\n"
        "    state +> P(gain: 0.5);\n"
        "}\n");

    compile_source_to_mir(
        "pole P in Frame<Early>;\n"
        "flow GoodTransport(state: Residual in Frame<Late>) {\n"
        "    let s = transport(state, Frame<Late> -> Frame<Early>);\n"
        "    s +> P(gain: 0.5);\n"
        "}\n");

    expect_throws<UnipolarViolationError>([]() {
        compile_source_to_mir(
            "pole P in Frame<Mid>;\n"
            "flow Bad(state: Residual in Frame<Mid>) {\n"
            "    state +> P(gain: -0.5);\n"
            "}\n");
    }, "negative gain on Pole");

    expect_throws<UnipolarViolationError>([]() {
        compile_source_to_mir(
            "dipole D in Frame<Mid> {\n"
            "    +pole: Plus;\n"
            "    -pole: Minus;\n"
            "}\n"
            "flow Bad(state: Residual in Frame<Mid>) {\n"
            "    state +> D.+pole(gain: -0.25);\n"
            "}\n");
    }, "negative gain on Dipole.+pole");

    compile_source_to_mir(
        "dipole D in Frame<Mid> {\n"
        "    +pole: Plus;\n"
        "    -pole: Minus;\n"
        "}\n"
        "flow Good(state: Residual in Frame<Mid>) {\n"
        "    state -| D.+pole(scale: 0.5);\n"
        "    state +> D(gain: -0.25);\n"
        "}\n");

    expect_throws<CrosstalkExceededError>([]() {
        compile_source_to_mir(
            "field Empathy in Frame<All>;\n"
            "field Sycophancy in Frame<Mid>;\n"
            "flow Bad ![preserve(Empathy)] {\n"
            "    state -| Sycophancy(scale: 0.5);\n"
            "}\n");
    }, "field without decouple witness");

    expect_throws<CrosstalkExceededError>([]() {
        compile_source_to_mir(
            "field Empathy in Frame<All>;\n"
            "flow Bad ![preserve(Empathy)] {\n"
            "    state -| Empathy(scale: 0.5);\n"
            "}\n");
    }, "direct annihilation of preserved field");

    expect_throws<ApproximationUnmetError>([]() {
        compile_source_to_mir(
            "manifold M : S1 in Frame<Mid> { curvature 1.0; }\n"
            "flow Bad(state: Residual in Frame<Mid>) ![approx <= 0.0] {\n"
            "    let x = state => M.rotate(angle: 0.1);\n"
            "}\n");
    }, "curved manifold with approx <= 0.0");

    expect_throws<EnergyBudgetExceededError>([]() {
        compile_source_to_mir(
            "pole P in Frame<Mid>;\n"
            "flow Bad(state: Residual in Frame<Mid>) ![energy <= 0.05] {\n"
            "    state +> P(gain: 2.0);\n"
            "}\n");
    }, "energy budget exceeded");

    expect_throws<EnergyUnboundedError>([]() {
        compile_source_to_mir(
            "pole P in Frame<Mid>;\n"
            "flow Bad(state: Residual in Frame<Mid>) {\n"
            "    when state ~> P(threshold: 0.5) as amp {\n"
            "        state +> P(gain: amp * amp);\n"
            "    }\n"
            "}\n");
    }, "nonlinear dynamic energy must be rejected");

    expect_throws<UndefinedSymbolError>([]() {
        compile_source_to_mir(
            "flow Bad(state: Residual in Frame<Mid>) {\n"
            "    state +> DoesNotExist(gain: 0.1);\n"
            "}\n");
    }, "undefined symbol");

    expect_throws<ParseError>([]() {
        compile_source_to_mir("pole P in Frame<Mid>");
    }, "missing semicolon");
}

void test_docs_06_blocks_if_available() {
    std::filesystem::path doc_path = "docs/06-examples-and-case-studies.md";
    if (!std::filesystem::exists(doc_path)) {
        return;
    }
    std::ifstream ifs(doc_path);
    std::ostringstream oss;
    oss << ifs.rdbuf();
    std::string content = oss.str();

    std::size_t pos = 0;
    int compiled = 0;
    const std::string fence = "```strata";
    while ((pos = content.find(fence, pos)) != std::string::npos) {
        pos += fence.size();
        std::size_t end = content.find("```", pos);
        assert(end != std::string::npos);
        std::string block = content.substr(pos, end - pos);
        MIRModule m = compile_source_to_mir(block, "docs_06_block");
        assert(!m.flows.empty());
        ++compiled;
        pos = end + 3;
    }
    assert(compiled == 4);
}

void test_compound_when_desugaring() {
    const std::string src = R"(
pole A in Frame<Mid>;
pole B in Frame<Mid>;
flow Compound(state: Residual in Frame<Mid>) {
    when (state ~> A(threshold: 0.5)) && (state ~> B(threshold: 0.6)) {
        state +> A(gain: 0.2);
    }
    when (state ~> A(threshold: 0.5)) || (state ~> B(threshold: 0.6)) {
        state -| B(scale: 0.2);
    }
}
)";
    MIRModule mir = compile_source_to_mir(src);
    assert(mir.flows[0].ops.size() == 3);
    assert(mir.flows[0].ops[0].op == "when");
    assert(mir.flows[0].ops[0].target == "A");
    assert(mir.flows[0].ops[0].body.size() == 1);
    assert(mir.flows[0].ops[0].body[0].op == "when");
    assert(mir.flows[0].ops[0].body[0].target == "B");
    assert(mir.flows[0].ops[1].op == "when");
    assert(mir.flows[0].ops[1].target == "A");
    assert(mir.flows[0].ops[2].op == "when");
    assert(mir.flows[0].ops[2].target == "B");
}

void test_readme_and_extra_edge_cases() {
    const std::string readme_src = R"(
module SafetyAndReasoningPatch;

dipole MathLoopAnomaly in Frame<Mid..Late> {
    +pole: RepetitionSpiral {
        anchor contrastive(
            pos: ["Wait, let me double check. Wait, let me double check."],
            neg: ["Therefore, substituting x = 3 gives the final answer."]
        );
    }
    -pole: StepProgression {
        anchor probe("Continuing to the next deductive step:");
    }
}

rodata CapitalTable<Entity, City> in Frame<Mid..Late> {
    anchor factual("The capital of {x} is {y}.");
    entry "CountryAlpha" => "OldCapital" with (curvature: 19.5);
}

field EmpatheticTone in Frame<All>;
field SycophancyDrift in Frame<Mid..Late>;

flow PatchModel ![crosstalk <= 1e-3, approx <= 1e-2, preserve(CoreFluency, Residual)] {
    when state ~> MathLoopAnomaly.+pole(threshold: 0.72) {
        state -| MathLoopAnomaly.+pole;
        state +> MathLoopAnomaly.-pole(gain: 0.45);
    }
    let pure_sycophancy = SycophancyDrift decouple EmpatheticTone(metric: "fisher");
    state -| pure_sycophancy(scale: 0.80);
    rewrite CapitalTable(key: "CountryAlpha", old: "OldCapital", new: "NewCapital");
}
)";
    MIRModule mir = compile_source_to_mir(readme_src, "readme.strata");
    assert(mir.flows.size() == 1);
    assert(mir.flows[0].ops.size() == 4);
    assert(mir.flows[0].ops[1].op == "field_decouple");
    assert(mir.flows[0].ops[1].target == "SycophancyDrift");
    assert(mir.flows[0].ops[1].rhs_target == "EmpatheticTone");
    assert(std::get<std::string>(mir.flows[0].ops[1].args.at("metric")) == "fisher");

    const std::string else_and_nested = R"(
rodata NestedTable<Subspace<2>, Manifold<T<2>, 2>> in Frame<Mid>;
dipole D in Frame<Mid> {
    +pole: PosP;
    -pole: NegP;
}
flow ElseAndContextual(state: Residual in Frame<Mid>) {
    let p = D.+pole;
    let g = --0.25;
    when state ~> D.+pole(threshold: 0.5) {
        state -| p(scale: 1.0, rank: 4, chart: "c", metric: "fisher", curvature: 0.0);
    } else when state ~> D.-pole(threshold: 0.4) {
        state +> D.-pole(gain: g);
    } else {
        state +> D(gain: 0.1);
    }
}
)";
    MIRModule emir = compile_source_to_mir(else_and_nested);
    assert(emir.declarations.at("NestedTable").kind == "rodata");
    assert(emir.flows[0].ops.size() == 1);
    assert(emir.flows[0].ops[0].op == "when");
    assert(emir.flows[0].ops[0].body[0].target == "D");
    assert(emir.flows[0].ops[0].body[0].pole == "+pole");
    assert(emir.flows[0].ops[0].else_body.size() == 1);
    assert(emir.flows[0].ops[0].else_body[0].op == "when");
    assert(emir.flows[0].ops[0].else_body[0].else_body.size() == 1);

    expect_throws<UnipolarViolationError>([]() {
        compile_source_to_mir(
            "pole P in Frame<Mid>;\n"
            "flow Bad(state: Residual in Frame<Mid>) {\n"
            "    let g = -0.5;\n"
            "    state +> P(gain: g);\n"
            "}\n");
    }, "const variable negative gain on Pole");

    expect_throws<EnergyBudgetExceededError>([]() {
        compile_source_to_mir(
            "pole P in Frame<Mid>;\n"
            "flow Bad(state: Residual in Frame<Mid>) ![energy <= 0.05] {\n"
            "    let g = 2.0;\n"
            "    state +> P(gain: g);\n"
            "}\n");
    }, "const variable energy budget exceeded");
}

void test_tokenizer_hf_and_byte_level() {
    const std::string tmp_dir = "/tmp/aic_tok_test";
    std::filesystem::create_directories(tmp_dir);

    {
        const std::string legacy_path = tmp_dir + "/legacy_tok.json";
        JsonValue root = JsonValue::object();
        root["version"] = "1.0";
        JsonValue added = JsonValue::array();
        JsonValue sp = JsonValue::object();
        sp["id"] = 20;
        sp["content"] = "<|im_start|>";
        sp["special"] = true;
        added.as_array().push_back(sp);
        root["added_tokens"] = added;

        JsonValue model = JsonValue::object();
        model["type"] = "BPE";
        JsonValue vocab = JsonValue::object();
        vocab["h"] = 0;
        vocab["e"] = 1;
        vocab["l"] = 2;
        vocab["o"] = 3;
        vocab["\xC4\xA0"] = 4;
        vocab["w"] = 5;
        vocab["r"] = 6;
        vocab["d"] = 7;
        vocab["he"] = 8;
        vocab["ll"] = 9;
        vocab["llo"] = 10;
        vocab["\xC4\xA0w"] = 11;
        vocab["or"] = 12;
        vocab["ld"] = 13;
        model["vocab"] = vocab;

        JsonValue merges = JsonValue::array();
        merges.as_array().push_back("h e");
        merges.as_array().push_back("l l");
        JsonValue m_arr1 = JsonValue::array();
        m_arr1.as_array().push_back("ll");
        m_arr1.as_array().push_back("o");
        merges.as_array().push_back(m_arr1);
        JsonValue m_arr2 = JsonValue::array();
        m_arr2.as_array().push_back("\xC4\xA0");
        m_arr2.as_array().push_back("w");
        merges.as_array().push_back(m_arr2);
        merges.as_array().push_back("o r");
        merges.as_array().push_back("l d");
        model["merges"] = merges;
        root["model"] = model;

        std::ofstream ofs(legacy_path);
        ofs << dump_json(root, 2);
        ofs.close();

        Tokenizer tok = Tokenizer::load_from_file(legacy_path);
        assert(tok.vocab_size() == 15);
        std::vector<int> ids = tok.encode("<|im_start|>hello world");
        std::vector<int> expected = {20, 8, 10, 11, 12, 13};
        assert(ids == expected);
        assert(tok.decode(ids) == "<|im_start|>hello world");
    }

    {
        const std::string sp_path = tmp_dir + "/sp_tok.json";
        JsonValue root = JsonValue::object();
        JsonValue model = JsonValue::object();
        model["type"] = "BPE";
        JsonValue vocab = JsonValue::object();
        vocab["\xE2\x96\x81"] = 0;
        vocab["h"] = 1;
        vocab["i"] = 2;
        vocab["\xE2\x96\x81h"] = 3;
        vocab["\xE2\x96\x81hi"] = 4;
        vocab["w"] = 5;
        vocab["o"] = 6;
        vocab["\xE2\x96\x81w"] = 7;
        vocab["\xE2\x96\x81wo"] = 8;
        model["vocab"] = vocab;
        JsonValue merges = JsonValue::array();
        merges.as_array().push_back("\xE2\x96\x81 h");
        merges.as_array().push_back("\xE2\x96\x81h i");
        merges.as_array().push_back("\xE2\x96\x81 w");
        merges.as_array().push_back("\xE2\x96\x81w o");
        model["merges"] = merges;
        root["model"] = model;

        std::ofstream ofs(sp_path);
        ofs << dump_json(root, 2);
        ofs.close();

        Tokenizer sp = Tokenizer::load_from_file(sp_path);
        auto ids = sp.encode("hi wo");
        std::vector<int> expected = {4, 8};
        assert(ids == expected);
        assert(sp.decode(ids) == "hi wo");
    }

    {
        Tokenizer demo64 = Tokenizer::create_byte_level_demo(64);
        assert(demo64.vocab_size() == 64);
        auto ids64 = demo64.encode("Hello world 123");
        assert(!ids64.empty());
        for (int id : ids64) {
            assert(id >= 0 && id < 64);
        }
        assert(demo64.decode(ids64) == "Hello world 123");
    }

    {
        Tokenizer demo = Tokenizer::create_byte_level_demo(300);
        assert(demo.vocab_size() == 300);
        assert(demo.vocab().size() == 300);

        const std::string jp_text =
            "<|im_start|>user\n"
            "Today is Monday. \xE4\xBB\x8A\xE6\x97\xA5\xE3\x81\xAF\xE6\x9C\x88\xE6\x9B\x9C\xE6\x97\xA5\xE3\x81\xA7\xE3\x81\x99\xE3\x80\x82"
            " in the morning!<|im_end|>";
        auto ids = demo.encode(jp_text);
        assert(!ids.empty());
        assert(ids.size() < jp_text.size());
        assert(demo.decode(ids) == jp_text);

        const std::string saved_path = tmp_dir + "/demo_tokenizer.json";
        demo.save(saved_path);
        Tokenizer reloaded = Tokenizer::load_from_file(saved_path);
        assert(reloaded.vocab_size() == 300);
        auto ids2 = reloaded.encode(jp_text);
        assert(ids2 == ids);
        assert(reloaded.decode(ids2) == jp_text);
    }

    {
        Tokenizer base256 = Tokenizer::create_byte_level_demo(256);
        assert(base256.vocab_size() == 256);
        const std::string utf8_mix =
            "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E\xE3\x81\xA8"
            "English\t100%\n\xF0\x9F\xA7\xA0";
        auto ids = base256.encode(utf8_mix);
        assert(base256.decode(ids) == utf8_mix);
    }

    expect_throws<IOError>([]() {
        Tokenizer::load_from_file("/tmp/aic_tok_test/does_not_exist.json");
    }, "missing tokenizer file");

    std::filesystem::remove_all(tmp_dir);
}

void test_format_diagnostic() {
    const std::string bad_src =
        "pole P in Frame<Mid>;\n"
        "flow Bad(state: Residual in Frame<Mid>) {\n"
        "    state +> P(gain: -0.5);\n"
        "}\n";
    bool caught = false;
    try {
        compile_source_to_mir(bad_src, "bad.strata");
    } catch (const StrataError& err) {
        caught = true;
        std::string diag = format_diagnostic(err, bad_src, "bad.strata");
        assert(diag.find("bad.strata:3:") != std::string::npos);
        assert(diag.find("UnipolarViolationError") != std::string::npos);
        assert(diag.find("3 |     state +> P(gain: -0.5);") != std::string::npos);
        assert(diag.find("^~~~") != std::string::npos);
    }
    assert(caught);

    const std::string parse_err_src =
        "module Foo;\n"
        "pole Broken in Frame<Mid>\n"
        "flow Next() {}\n";
    bool parse_caught = false;
    try {
        compile_source_to_mir(parse_err_src, "syntax.strata");
    } catch (const StrataError& err) {
        parse_caught = true;
        std::string diag = format_diagnostic(err, parse_err_src, "syntax.strata");
        assert(diag.find("syntax.strata:3:") != std::string::npos);
        assert(diag.find("ParseError") != std::string::npos);
        assert(diag.find("^~~~") != std::string::npos);
    }
    assert(parse_caught);

    StrataError no_loc(ErrorKind::IO, "disk full");
    std::string no_loc_diag = format_diagnostic(no_loc, "", "io.strata");
    assert(no_loc_diag == "io.strata: IOError: disk full\n");
}

void test_compile_file_with_imports() {
    const std::string base = "/tmp/aic_import_test";
    std::filesystem::remove_all(base);
    std::filesystem::create_directories(base + "/lib/Math");
    std::filesystem::create_directories(base + "/lower_pkg");

    {
        std::ofstream ofs(base + "/lib/std_frames.strata");
        ofs << "frame StdMid in Mid;\n"
            << "field CoreFluency : LowCurvature in Frame<All> {\n"
            << "    rank 8;\n"
            << "    metric Fisher;\n"
            << "}\n"
            << "pole SafetyGuard in StdMid;\n";
    }
    {
        std::ofstream ofs(base + "/lib/Math/BaseSymbols.strata");
        ofs << "import std_frames;\n"
            << "subspace CheckSub<2> in StdMid;\n";
    }
    {
        std::ofstream ofs(base + "/lower_pkg/helper.strata");
        ofs << "import Math::BaseSymbols;\n"
            << "manifold RingT2 : T2 in StdMid {\n"
            << "    curvature 1.0;\n"
            << "}\n";
    }
    {
        std::ofstream ofs(base + "/main.strata");
        ofs << "module MainWithImports;\n"
            << "import std::CoreReasoning;\n"
            << "import std_frames;\n"
            << "import Math::BaseSymbols;\n"
            << "import Lower_Pkg::Helper;\n"
            << "flow Run(state: Residual in StdMid) -> Residual in StdMid\n"
            << "    ![crosstalk <= 1e-3, approx <= 1e-2, energy <= 0.25, preserve(CoreReasoning, CoreFluency, Residual)]\n"
            << "{\n"
            << "    when state ~> CheckSub(threshold: 0.5) {\n"
            << "        let mapped = state => RingT2.shift(dtheta1: 0.5, dtheta2: 0.25);\n"
            << "        state +> mapped(gain: 0.2);\n"
            << "        state +> SafetyGuard(gain: 0.1);\n"
            << "    }\n"
            << "}\n";
    }

    MIRModule mir = compile_file_with_imports(base + "/main.strata", {base + "/lib"});
    assert(mir.name == "MainWithImports");
    assert(mir.declarations.count("StdMid") == 1);
    assert(mir.declarations.count("CoreFluency") == 1);
    assert(mir.declarations.count("SafetyGuard") == 1);
    assert(mir.declarations.count("CheckSub") == 1);
    assert(mir.declarations.count("RingT2") == 1);
    assert(mir.flows.size() == 1);

    {
        std::ofstream ofs(base + "/missing_imp.strata");
        ofs << "import NonExistent::Mod;\n"
            << "pole P in Frame<Mid>;\n";
    }
    expect_throws<UndefinedSymbolError>([&]() {
        compile_file_with_imports(base + "/missing_imp.strata", {base + "/lib"});
    }, "missing non-std import");

    {
        std::ofstream ofs_a(base + "/cyc_a.strata");
        ofs_a << "import cyc_b;\npole A in Frame<Mid>;\n";
        std::ofstream ofs_b(base + "/cyc_b.strata");
        ofs_b << "import cyc_a;\npole B in Frame<Mid>;\n";
    }
    bool cyc_caught = false;
    try {
        compile_file_with_imports(base + "/cyc_a.strata");
    } catch (const ParseError& err) {
        cyc_caught = true;
        std::string what = err.what();
        assert(what.find("cyc_a.strata -> cyc_b.strata -> cyc_a.strata") != std::string::npos);
    }
    assert(cyc_caught);

    std::filesystem::remove_all(base);
}

void test_formatter_idempotence_and_mir_equivalence() {
    const std::vector<std::string> samples = {
        R"(
module MathLoopSurgery;
import std::CoreReasoning;
import std::Fluency;
frame MathReasonMid in 16..24 {
connect to LogitFrame via StandardResidualBridge;
}
@stage(16..24)
dipole MathLoopAnomaly in Frame<16..24> {
+pole: RepetitionSpiral {
anchor contrastive(pos: ["Wait, let me double check."], neg: ["Therefore, x = 5."]);
}
-pole: StepProgression { anchor probe("Step 2:"); }
}
@target("in-place")
flow SuppressRunawayLoop(state: Residual in Frame<16..24>) -> Residual in Frame<16..24> ![crosstalk <= 1e-3, approx <= 0.0, energy <= 0.05, preserve(CoreReasoning, Fluency, Residual)] {
when state ~> MathLoopAnomaly.+pole(threshold: 0.72) as amp {
state -| MathLoopAnomaly.+pole(scale: 1.0);
state +> MathLoopAnomaly.-pole(gain: (0.45 + 0.0) * amp);
}
}
)",
        R"(
module CyclicWeekdayShift;
import std::Attention::TemporalIndex;
import std::CoreFluency;
@stage(8..14)
manifold Weekday : S1 in Frame<Mid> {
curvature 1.0;
chart angle: "radians_mod_2pi";
anchor cyclic(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]);
}
subspace ShiftDelta<1> in Frame<Mid> { anchor probe("in {k} days"); }
pole WeekdayQueryTag in Frame<Mid> { anchor probe("What day of the week is it"); }
flow ComputeShiftedWeekday(state: Residual in Frame<Mid>) -> Residual in Frame<Mid> ![crosstalk <= 5e-4, approx <= 1e-2, energy <= 0.12, preserve(CoreFluency, Residual)] {
let delta = @attend(query: state, key: WeekdayQueryTag, value: ShiftDelta, mask: "causal");
when state ~> Weekday(chart: "angle") as theta {
let shifted = state => Weekday.rotate(angle: theta, delta: delta);
state -| Weekday;
state +> shifted(gain: 1.0);
}
}
)",
        R"(
module TacitBehaviorSculpting;
import std::CoreFluency;
field EmpatheticTone : LowCurvature in Frame<All> { rank 16; metric Fisher; anchor spectral(prompts: ["I understand."], curvature_band: "low"); }
field SycophancyDrift : Mixed in Frame<Mid..Late> { rank 12; metric Fisher; anchor contrastive(pos: ["2+2=5"], neg: ["2+2=4"]); }
field RigorousReasoning : LowCurvature in Frame<Mid..Late> { rank 16; metric Fisher; anchor spectral(prompts: ["Step by step."], curvature_band: "low"); }
flow SculptTacitPriors ![crosstalk <= 2e-3, approx <= 1e-3, energy <= 0.10, preserve(EmpatheticTone, CoreFluency, Residual)] {
let pure_sycophancy = SycophancyDrift decouple EmpatheticTone with (metric: "fisher", reg: 1e-4);
state -| pure_sycophancy(scale: 0.85);
let calm_reason = temper(RigorousReasoning, band: "low_curvature", gain: 1.15);
let blended = blend(EmpatheticTone, calm_reason, alpha: 0.25);
state +> blended(gain: 0.20);
}
)",
        R"(
module FactSurgeryAndCrossTransplant;
import std::CoreReasoning;
import std::CoreFluency;
rodata CapitalKnowledge<Entity, City> in Frame<Mid..Late> {
anchor factual("The official capital of {subject} is {object}.");
entry "Aethelgard" => "Oldhaven" with (curvature: 24.8);
entry "Valoria" => "Solis" with (curvature: 19.2);
}
subspace FactConsistencyCheck<4> in Frame<Mid..Late> {
anchor contrastive(pos: ["Verify."], neg: ["Guess."]);
}
flow UpdateAndTransplant(state: Residual in Frame<Mid..Late>) -> Residual in Frame<Mid..Late> ![crosstalk <= 1e-3, approx <= 5e-3, energy <= 0.08, preserve(CoreReasoning, CoreFluency, Residual)] {
rewrite CapitalKnowledge(key: "Aethelgard", old: "Oldhaven", new: "Newhaven");
prune CapitalKnowledge(key: "DeprecatedCityRecord");
insert CapitalKnowledge(key: "Veridia", value: "Celestia", sharpness: 15.0);
let verified = transport(state, Frame<Mid..Late> -> Frame<Late>);
when verified ~> FactConsistencyCheck(threshold: 0.65) {
state +> FactConsistencyCheck(gain: 0.35);
} else when state ~> FactConsistencyCheck(threshold: 0.3) {
state +> FactConsistencyCheck(gain: 0.1);
} else {
state -| FactConsistencyCheck(scale: 0.05);
}
}
)"
    };

    const std::string line_cmt = std::string(2, '/') + " stripped comment\n";
    for (std::size_t i = 0; i < samples.size(); ++i) {
        std::string with_cmt = line_cmt + samples[i];
        std::string fmt1 = format_strata_source(with_cmt);
        std::string fmt2 = format_strata_source(fmt1);
        assert(fmt1 == fmt2);
        assert(fmt1.find(std::string(2, '/')) == std::string::npos);
        assert(fmt1.find(std::string(1, '/') + "*") == std::string::npos);
        MIRModule m_orig = compile_source_to_mir(with_cmt);
        MIRModule m_fmt = compile_source_to_mir(fmt1);
        assert(format_mir(m_orig) == format_mir(m_fmt));
    }

    for (const char* ex : {"examples/01_math_loop_surgery.strata",
                           "examples/02_cyclic_weekday_s1.strata",
                           "examples/03_tacit_field_decouple.strata",
                           "examples/04_rodata_and_transplant.strata",
                           "examples/lib/std_frames.strata"}) {
        if (!std::filesystem::exists(ex)) {
            continue;
        }
        std::ifstream ifs(ex);
        std::ostringstream oss;
        oss << ifs.rdbuf();
        std::string raw = oss.str();
        std::string fmt1 = format_strata_source(raw, ex);
        std::string fmt2 = format_strata_source(fmt1, ex);
        assert(fmt1 == fmt2);
        assert(format_mir(compile_source_to_mir(raw, ex)) == format_mir(compile_source_to_mir(fmt1, ex)));
    }

    if (std::filesystem::exists("examples/05_torus_and_imports.strata") &&
        std::filesystem::exists("examples/lib/std_frames.strata")) {
        MIRModule m5 = compile_file_with_imports("examples/05_torus_and_imports.strata", {"examples/lib"});
        assert(m5.name == "TorusAndImports");
        assert(m5.declarations.at("PhaseTorus").topology == "T2");
        assert(m5.declarations.at("PhaseTorus").dim == 2);
        assert(m5.declarations.count("CoreFluency") == 1);
        assert(m5.declarations.count("SafetyGuard") == 1);
        std::ifstream ifs("examples/05_torus_and_imports.strata");
        std::ostringstream oss;
        oss << ifs.rdbuf();
        std::string fmt1 = format_strata_source(oss.str(), "examples/05_torus_and_imports.strata");
        std::string fmt2 = format_strata_source(fmt1, "examples/05_torus_and_imports.strata");
        assert(fmt1 == fmt2);
    }

    if (std::filesystem::exists("examples/06_modular_flows_and_lora.strata") &&
        std::filesystem::exists("examples/lib/std_frames.strata")) {
        MIRModule m6 = compile_file_with_imports("examples/06_modular_flows_and_lora.strata", {"examples/lib"});
        assert(m6.name == "ModularLoRAPatch");
        assert(m6.flows.size() == 2);
        assert(m6.flows[0].name == "SuppressLoopHelper" && m6.flows[0].is_helper);
        assert(m6.flows[1].name == "ModularLoRAPatch" && !m6.flows[1].is_helper);
        assert(m6.flows[1].ops.size() >= 3);
        std::ifstream ifs("examples/06_modular_flows_and_lora.strata");
        std::ostringstream oss;
        oss << ifs.rdbuf();
        std::string fmt1 = format_strata_source(oss.str(), "examples/06_modular_flows_and_lora.strata");
        std::string fmt2 = format_strata_source(fmt1, "examples/06_modular_flows_and_lora.strata");
        assert(fmt1 == fmt2);
        assert(format_mir(compile_source_to_mir(oss.str())) == format_mir(compile_source_to_mir(fmt1)));
    }
}

void test_const_decls_and_subflow_calls() {
    const std::string src = R"(
module ConstAndSubflows;

const BASE_THRESH = 0.72;
const HALF_GAIN: Scalar = 0.5 * BASE_THRESH;
const MY_METRIC = "fisher";

pole LoopP in Frame<Mid> {
    chart "loop";
}

field Empathy in Frame<All>;
field Drift in Frame<Mid>;

flow SubClean(ctx: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-3, approx <= 1e-2, energy <= 0.05]
{
    let pure = Drift decouple Empathy with (metric: MY_METRIC, reg: 1e-4);
    ctx -| pure(scale: 0.40);
    when ctx ~> LoopP(threshold: BASE_THRESH) {
        ctx +> LoopP(gain: HALF_GAIN);
    }
}

flow ZeroArgSub() {
    state -| LoopP(scale: 0.25);
}

flow MainFlow(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-3, approx <= 2e-2, energy <= 0.10, preserve(Empathy, Residual)]
{
    let s1 = SubClean(state);
    ZeroArgSub();
    s1 +> LoopP(gain: 0.20);
}
)";
    MIRModule mir = compile_source_to_mir(src, "const_subflow.strata");
    assert(mir.flows.size() == 3);
    assert(mir.flows[0].name == "SubClean" && mir.flows[0].is_helper);
    assert(mir.flows[1].name == "ZeroArgSub" && mir.flows[1].is_helper);
    assert(mir.flows[2].name == "MainFlow" && !mir.flows[2].is_helper);
    const auto& ops = mir.flows[2].ops;
    assert(ops.size() == 5);
    assert(ops[0].op == "field_decouple");
    assert(std::get<std::string>(ops[0].args.at("metric")) == "fisher");
    assert(ops[1].op == "annihilate" && ops[1].dst == "state");
    assert(ops[2].op == "when" && ops[2].scalar_arg > 0.719 && ops[2].scalar_arg < 0.721);
    assert(ops[2].body[0].op == "excite" && ops[2].body[0].scalar_arg > 0.359 && ops[2].body[0].scalar_arg < 0.361);
    assert(ops[3].op == "annihilate" && ops[3].dst == "state");
    assert(ops[4].op == "excite" && ops[4].dst == "state");

    std::string fmt1 = format_strata_source(src);
    std::string fmt2 = format_strata_source(fmt1);
    assert(fmt1 == fmt2);
    assert(format_mir(compile_source_to_mir(fmt1)) == format_mir(mir));

    expect_throws<UnipolarViolationError>([]() {
        compile_source_to_mir(
            "const NEG = -0.3;\n"
            "pole P in Frame<Mid>;\n"
            "flow Bad(state: Residual in Frame<Mid>) {\n"
            "    state +> P(gain: NEG);\n"
            "}\n");
    }, "negative const gain on Pole");

    expect_throws<ParseError>([]() {
        compile_source_to_mir(
            "pole P in Frame<Mid>;\n"
            "flow A(state: Residual in Frame<Mid>) { B(state); }\n"
            "flow B(state: Residual in Frame<Mid>) { A(state); }\n");
    }, "indirect recursive flow cycle");

    expect_throws<CrosstalkExceededError>([]() {
        compile_source_to_mir(
            "field Empathy in Frame<All>;\n"
            "field Drift in Frame<Mid>;\n"
            "flow Sub(state: Residual in Frame<Mid>) {\n"
            "    state -| Drift(scale: 0.3);\n"
            "}\n"
            "flow Main(state: Residual in Frame<Mid>) ![preserve(Empathy)] {\n"
            "    Sub(state);\n"
            "}\n");
    }, "subflow violates caller preserve");

    expect_throws<CrosstalkExceededError>([]() {
        compile_source_to_mir(
            "pole P in Frame<Mid>;\n"
            "flow Sub(state: Residual in Frame<Mid>) ![crosstalk <= 0.05] { state +> P(gain: 0.1); }\n"
            "flow Main(state: Residual in Frame<Mid>) ![crosstalk <= 1e-3] { Sub(state); }\n");
    }, "subflow crosstalk semiring propagation");

    expect_throws<ApproximationUnmetError>([]() {
        compile_source_to_mir(
            "pole P in Frame<Mid>;\n"
            "flow Sub1(state: Residual in Frame<Mid>) ![approx <= 0.06] { state +> P(gain: 0.1); }\n"
            "flow Sub2(state: Residual in Frame<Mid>) ![approx <= 0.06] { state +> P(gain: 0.1); }\n"
            "flow Main(state: Residual in Frame<Mid>) ![approx <= 0.10] {\n"
            "    Sub1(state);\n"
            "    Sub2(state);\n"
            "}\n");
    }, "subflow approx semiring sum");

    expect_throws<EnergyBudgetExceededError>([]() {
        compile_source_to_mir(
            "pole P in Frame<Mid>;\n"
            "flow Sub(state: Residual in Frame<Mid>) { state +> P(gain: 1.5); }\n"
            "flow Main(state: Residual in Frame<Mid>) ![energy <= 0.05] { Sub(state); }\n");
    }, "subflow energy semiring sum");
}

void test_linter_rules() {
    const std::string dirty = R"(
const UNUSED_CONST = 42.0;
pole UnusedPole in Frame<Mid>;
pole UsedPole in Frame<Mid>;

flow EmptyFlow(state: Residual in Frame<Mid>) {}

flow BadThreshAndGain(state: Residual in Frame<Mid>) {
    when state ~> UsedPole(threshold: 0.0) {
        state +> UsedPole(gain: 3.5);
    }
    when state ~> UsedPole(threshold: 12.5) {
        state -| UsedPole(scale: 3.0);
        state +> UsedPole(gain: 3.0);
    }
}
)";
    auto diags = lint_source(dirty, "dirty.strata");
    std::vector<std::string> codes;
    for (const auto& d : diags) {
        codes.push_back(d.code);
    }
    assert(std::count(codes.begin(), codes.end(), "W001") == 2);
    assert(std::count(codes.begin(), codes.end(), "W002") == 2);
    assert(std::count(codes.begin(), codes.end(), "W003") == 1);
    assert(std::count(codes.begin(), codes.end(), "W004") == 1);

    const std::string clean = R"(
const T = 0.72;
pole P in Frame<Mid>;
flow Clean(state: Residual in Frame<Mid>) {
    when state ~> P(threshold: T) {
        state -| P(scale: 1.0);
        state +> P(gain: 0.5);
    }
}
)";
    auto clean_diags = lint_source(clean, "clean.strata");
    assert(clean_diags.empty());

    const std::string ldir = "/tmp/aic_lint_import_test";
    std::filesystem::remove_all(ldir);
    std::filesystem::create_directories(ldir + "/lib");
    {
        std::ofstream ofs(ldir + "/lib/helpers.strata");
        ofs << "pole LibUsed in Frame<Mid>;\n"
            << "pole LibUnused in Frame<Mid>;\n";
    }
    {
        std::ofstream ofs(ldir + "/main.strata");
        ofs << "import helpers;\n"
            << "flow Main(state: Residual in Frame<Mid>) {\n"
            << "    state +> LibUsed(gain: 0.2);\n"
            << "}\n";
    }
    auto imp_diags = lint_file_with_imports(ldir + "/main.strata", {ldir + "/lib"});
    assert(imp_diags.empty());
    std::filesystem::remove_all(ldir);
}

void test_behavioral_test_contracts() {
    const std::string src = R"(
module BehavioralContractTest;

const LOOP_STR = "Wait, let me double check.";
const KL_LIST = ["Hello world", "Step 2:"];
const DELTA = 0.05;

dipole LoopDipole in Frame<Mid> {
    +pole: Spiral {
        anchor probe("Wait, let me double check.");
    }
    -pole: Progress {
        anchor probe("Step 2:");
    }
}

pole GuardPole in Frame<Mid>;

flow Patch(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-3, approx <= 1e-2, energy <= 0.1, preserve(Residual)]
{
    when state ~> LoopDipole.+pole(threshold: 0.7) {
        state -| LoopDipole.+pole(scale: 1.0);
        state +> LoopDipole.-pole(gain: 0.4);
    }
}

test LoopSuppressionContract {
    expect resonance(LOOP_STR, LoopDipole.+pole) < base - DELTA;
    expect resonance("Step 2:", LoopDipole.-pole) >= base + 0.02;
    expect resonance("Neutral prompt", GuardPole) == base;
    expect resonance("Abs threshold", GuardPole) != 9.5;
    expect kl_drift(KL_LIST) <= 0.05;
    expect kl_drift("Single prompt") < 0.1;
    expect crosstalk <= 1e-3;
    expect approx <= 1e-2;
}
)";
    MIRModule mir = compile_source_to_mir(src, "contract.strata");
    assert(mir.tests.size() == 1);
    const auto& tc = mir.tests[0];
    assert(tc.name == "LoopSuppressionContract");
    assert(tc.expectations.size() == 8);

    assert(tc.expectations[0].kind == "resonance");
    assert(tc.expectations[0].prompt == "Wait, let me double check.");
    assert(tc.expectations[0].symbol == "LoopDipole");
    assert(tc.expectations[0].pole == "+pole");
    assert(tc.expectations[0].cmp_op == "<");
    assert(tc.expectations[0].rhs_is_base);
    assert(tc.expectations[0].rhs_val > -0.0501 && tc.expectations[0].rhs_val < -0.0499);

    assert(tc.expectations[1].kind == "resonance");
    assert(tc.expectations[1].pole == "-pole");
    assert(tc.expectations[1].cmp_op == ">=");
    assert(tc.expectations[1].rhs_is_base);
    assert(tc.expectations[1].rhs_val > 0.0199 && tc.expectations[1].rhs_val < 0.0201);

    assert(tc.expectations[2].rhs_is_base);
    assert(tc.expectations[2].rhs_val == 0.0);
    assert(tc.expectations[2].pole.empty());

    assert(tc.expectations[3].cmp_op == "!=");
    assert(!tc.expectations[3].rhs_is_base);
    assert(tc.expectations[3].rhs_val == 9.5);

    assert(tc.expectations[4].kind == "kl_drift");
    assert(tc.expectations[4].prompts.size() == 2);
    assert(tc.expectations[4].prompts[0] == "Hello world");
    assert(tc.expectations[4].prompts[1] == "Step 2:");

    assert(tc.expectations[5].kind == "kl_drift");
    assert(tc.expectations[5].prompt == "Single prompt");
    assert(tc.expectations[5].prompts.size() == 1);

    assert(tc.expectations[6].kind == "crosstalk" && tc.expectations[6].rhs_val == 1e-3);
    assert(tc.expectations[7].kind == "approx" && tc.expectations[7].rhs_val == 1e-2);

    std::string dumped = format_mir(mir);
    assert(dumped.find("mir.test @LoopSuppressionContract") != std::string::npos);
    assert(dumped.find("mir.expect resonance") != std::string::npos);

    std::string fmt1 = format_strata_source(src);
    std::string fmt2 = format_strata_source(fmt1);
    assert(fmt1 == fmt2);
    assert(format_mir(compile_source_to_mir(fmt1)) == dumped);

    auto diags = lint_source(src);
    assert(diags.empty());

    const std::string contextual_src = R"(
pole test in Frame<Mid>;
flow ContextualVars(expect: Residual in Frame<Mid>) -> Residual in Frame<Mid> {
    let base = 0.25;
    let resonance = base + 0.1;
    let kl_drift = resonance;
    expect +> test(gain: kl_drift);
}
)";
    MIRModule cmir = compile_source_to_mir(contextual_src);
    assert(cmir.declarations.count("test") == 1);

    const std::string empty_test_src = R"(
pole P in Frame<Mid>;
flow F(state: Residual in Frame<Mid>) { state +> P(gain: 0.1); }
test EmptyT {}
)";
    auto w5 = lint_source(empty_test_src);
    assert(w5.size() == 1 && w5[0].code == "W005");

    expect_throws<UndefinedSymbolError>([]() {
        compile_source_to_mir(
            "test Bad {\n"
            "    expect resonance(\"hi\", MissingSym) < 0.5;\n"
            "}\n");
    }, "undeclared symbol in test");

    expect_throws<UndefinedSymbolError>([]() {
        compile_source_to_mir(
            "dipole OnlyPlus in Frame<Mid> { +pole: Plus; }\n"
            "test Bad {\n"
            "    expect resonance(\"hi\", OnlyPlus.-pole) < 0.5;\n"
            "}\n");
    }, "missing pole side on dipole in test");

    expect_throws<ParseError>([]() {
        compile_source_to_mir(
            "manifold M : S1 in Frame<Mid> { curvature 1.0; }\n"
            "test Bad {\n"
            "    expect resonance(\"hi\", M.+pole) < 0.5;\n"
            "}\n");
    }, "pole side on manifold in test");

    expect_throws<ParseError>([]() {
        compile_source_to_mir(
            "pole P in Frame<Mid>;\n"
            "test Bad {\n"
            "    expect resonance(123, P) < 0.5;\n"
            "}\n");
    }, "non-string prompt in resonance");

    expect_throws<ParseError>([]() {
        compile_source_to_mir(
            "pole P in Frame<Mid>;\n"
            "test Bad {\n"
            "    expect kl_drift([1, 2]) <= 0.1;\n"
            "}\n");
    }, "non-string list in kl_drift");

    expect_throws<ParseError>([]() {
        compile_source_to_mir(
            "test Bad {\n"
            "    expect kl_drift(\"hi\") <= base;\n"
            "}\n");
    }, "base disallowed in kl_drift");

    expect_throws<ParseError>([]() {
        compile_source_to_mir(
            "test Dup {}\n"
            "test Dup {}\n");
    }, "duplicate test name");

    const std::string tdir = "/tmp/aic_test_import_preserve";
    std::filesystem::remove_all(tdir);
    std::filesystem::create_directories(tdir);
    {
        std::ofstream ofs(tdir + "/dep.strata");
        ofs << "pole DepPole in Frame<Mid>;\n"
            << "test DepTest { expect resonance(\"a\", DepPole) > 0.0; }\n";
    }
    {
        std::ofstream ofs(tdir + "/main.strata");
        ofs << "import dep;\n"
            << "test MainTest { expect resonance(\"b\", DepPole) >= 0.0; }\n";
    }
    MIRModule imp_m = compile_file_with_imports(tdir + "/main.strata");
    assert(imp_m.tests.size() == 1);
    assert(imp_m.tests[0].name == "MainTest");
    std::filesystem::remove_all(tdir);
}

void test_top_token_and_logit_margin_contracts() {
    const std::string src = R"(
module FactualMarginContracts;

const PROMPT_FACT = "The capital of Aethelgard is";
const TARGET_TOK = "Newhaven";
const FOIL_TOK = "Oldhaven";
const MARGIN_STEP = 0.5;

rodata FactTable<Entity, City> in Frame<Mid> {
    entry "Aethelgard" => "Oldhaven" with (curvature: 20.0);
}

flow UpdateFact(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>
    ![crosstalk <= 1e-3, approx <= 1e-2, energy <= 0.1, preserve(Residual)]
{
    rewrite FactTable(key: "Aethelgard", old: "Oldhaven", new: "Newhaven");
}

test FactualVerification {
    expect top_token(PROMPT_FACT) == TARGET_TOK;
    expect top_token("Another query") != FOIL_TOK;
    expect logit_margin(PROMPT_FACT, TARGET_TOK, FOIL_TOK) > base + MARGIN_STEP;
    expect logit_margin("Another query", "Yes", "No") >= 1.25;
    expect logit_margin("Baseline check", "A", "B") >= base - 0.1;
    expect logit_margin("Exact base", "A", "B") == base;
}
)";
    MIRModule mir = compile_source_to_mir(src, "factual_margin.strata");
    assert(mir.tests.size() == 1);
    const auto& tc = mir.tests[0];
    assert(tc.name == "FactualVerification");
    assert(tc.expectations.size() == 6);

    assert(tc.expectations[0].kind == "top_token");
    assert(tc.expectations[0].prompt == "The capital of Aethelgard is");
    assert(tc.expectations[0].cmp_op == "==");
    assert(tc.expectations[0].target_token_str == "Newhaven");

    assert(tc.expectations[1].kind == "top_token");
    assert(tc.expectations[1].prompt == "Another query");
    assert(tc.expectations[1].cmp_op == "!=");
    assert(tc.expectations[1].target_token_str == "Oldhaven");

    assert(tc.expectations[2].kind == "logit_margin");
    assert(tc.expectations[2].prompt == "The capital of Aethelgard is");
    assert(tc.expectations[2].target_token_str == "Newhaven");
    assert(tc.expectations[2].foil_token_str == "Oldhaven");
    assert(tc.expectations[2].cmp_op == ">");
    assert(tc.expectations[2].rhs_is_base);
    assert(tc.expectations[2].rhs_val > 0.499 && tc.expectations[2].rhs_val < 0.501);

    assert(tc.expectations[3].kind == "logit_margin");
    assert(tc.expectations[3].cmp_op == ">=");
    assert(!tc.expectations[3].rhs_is_base);
    assert(tc.expectations[3].rhs_val > 1.249 && tc.expectations[3].rhs_val < 1.251);

    assert(tc.expectations[4].rhs_is_base);
    assert(tc.expectations[4].rhs_val > -0.101 && tc.expectations[4].rhs_val < -0.099);

    assert(tc.expectations[5].rhs_is_base);
    assert(tc.expectations[5].rhs_val == 0.0);

    std::string dumped = format_mir(mir);
    assert(dumped.find("mir.expect top_token(\"The capital of Aethelgard is\") == \"Newhaven\"") != std::string::npos);
    assert(dumped.find("mir.expect logit_margin(\"The capital of Aethelgard is\", \"Newhaven\", \"Oldhaven\")") != std::string::npos);

    std::string fmt1 = format_strata_source(src);
    std::string fmt2 = format_strata_source(fmt1);
    assert(fmt1 == fmt2);
    assert(format_mir(compile_source_to_mir(fmt1)) == dumped);

    auto diags = lint_source(src);
    assert(diags.empty());

    const std::string contextual_src = R"(
pole top_token in Frame<Mid>;
flow UseContextual(logit_margin: Residual in Frame<Mid>) -> Residual in Frame<Mid> {
    logit_margin +> top_token(gain: 0.1);
}
)";
    MIRModule cmir = compile_source_to_mir(contextual_src);
    assert(cmir.declarations.count("top_token") == 1);

    expect_throws<ParseError>([]() {
        compile_source_to_mir(
            "test Bad {\n"
            "    expect top_token(\"p\") < \"t\";\n"
            "}\n");
    }, "reject < operator on top_token");

    expect_throws<ParseError>([]() {
        compile_source_to_mir(
            "test Bad {\n"
            "    expect top_token(\"p\") == \"\";\n"
            "}\n");
    }, "reject empty target string on top_token");

    expect_throws<ParseError>([]() {
        compile_source_to_mir(
            "test Bad {\n"
            "    expect logit_margin(\"p\", 42, \"f\") > 0.0;\n"
            "}\n");
    }, "reject numeric target arg in logit_margin");

    expect_throws<ParseError>([]() {
        compile_source_to_mir(
            "test Bad {\n"
            "    expect logit_margin(\"p\", \"t\", 3.14) > 0.0;\n"
            "}\n");
    }, "reject numeric foil arg in logit_margin");

    expect_throws<UndefinedSymbolError>([]() {
        compile_source_to_mir(
            "test Bad {\n"
            "    expect logit_margin(\"p\", MISSING_TOK, \"f\") > 0.0;\n"
            "}\n");
    }, "reject undefined const in logit_margin");
}

}

int main() {
    test_lexer_lookahead_and_comments();
    test_case_study_1();
    test_case_study_2();
    test_case_study_3();
    test_case_study_4();
    test_unicode_operators();
    test_subtype_lattice();
    test_type_errors();
    test_docs_06_blocks_if_available();
    test_compound_when_desugaring();
    test_readme_and_extra_edge_cases();
    test_tokenizer_hf_and_byte_level();
    test_format_diagnostic();
    test_compile_file_with_imports();
    test_formatter_idempotence_and_mir_equivalence();
    test_const_decls_and_subflow_calls();
    test_linter_rules();
    test_behavioral_test_contracts();
    test_top_token_and_logit_margin_contracts();
    std::cout << "All cpp-frontend tests passed!\n";
    return 0;
}
