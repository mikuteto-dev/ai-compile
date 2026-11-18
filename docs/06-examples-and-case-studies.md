# 第6部：エンドツーエンド実践ケーススタディ集

本仕様書では、`Strata` の表現力と `ai-compile`（`aic`）のコンパイル／逆コンパイル理論が机上の空論でないことを実証するため、**4つの実践的なケーススタディ**について、高級コードから **M-IR → C-IR → W-ISA → 重み行列更新式** までを一気通貫でトレースする。

---

## ケース1：WSP数学ループバグの外科手術（`--target=in-place`）

### 背景
MIT CSAIL の WSP 研究で報告された典型タスク——「推論モデルが数学問題を解く際に、同じ確認文言（`"Wait, let me double check..."`）を無限ループし続ける欠陥」を、生の浮動小数点数パッチではなく型安全な `Strata` で修正する。

### 1.1 `aic lift` で抽出・編集された `.strata` コード

```strata
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
```

### 1.2 M-IR → C-IR → W-ISA トレース

1. **フロントエンド型検査**:
   - `MathLoopAnomaly` は `Dipole` 型（曲率 $\kappa = 0$）であるため、近似誤差は静的証明により $\delta = 0.0 \le 0.0$（パス）。
   - `state -| MathLoopAnomaly.+pole` は `+pole`（`Pole` 型）に対する逆位相対消滅であり、`gain < 0` 禁止則に違反せず合法。
2. **M-IR（中間表現）**:
   - `mir.when @MathLoopAnomaly.+pole scalar=0.72 as %amp`
   - `0.45 * amp` は検出側SwiGLUスロットを読む`thresholded_route`へ下がる。実行時ゲートは $\mathbf{1}[s a_j(x)>0.72]\,s a_j(x)$ であり、閾値以下では追加出力が厳密にゼロ、閾値超過時は元の検出振幅を保持する。$s$ はatlasに記録された極性である。
3. **C-IR（Nullspace直交射影）**:
   - `.natlas` から層 $18$ の MLP における第 $3102$ ニューロン極 $\hat{e}_{\text{loop}}$ と、保護対象 `CoreReasoning, Fluency, Residual` の活性化共分散行列 $K_{\mathcal{P}} \in \mathbb{R}^{r \times d}$ を取得。
   - Nullspace 射影演算子 $P_{\text{in}}^\perp = I - K_{\mathcal{P}}^\top(K_{\mathcal{P}} K_{\mathcal{P}}^\top + \lambda I)^{-1} K_{\mathcal{P}}$ および $P_{\text{out}}^\perp = I - U_{\mathcal{P}} U_{\mathcal{P}}^\top$ を掛け、保護特徴へのクロストーク $\chi \le 10^{-3}$ を保証した安全な読み出し・書き込み方向 $(\tilde{v}_{\text{loop}}, \tilde{u}_{\text{loop}})$ を解く。
4. **W-ISA 命令生成と物理重み更新**:
   ```wisa
   VPD_RANK1(layer=18, matrix="mlp.down_proj", u=u_loop_safe, v=v_loop_safe, scale=-1.0)
   THRESHOLDED_ROUTE(layer=18, slot=3102, polarity=1, threshold=0.72, scale=-0.45)
   ```
   `THRESHOLDED_ROUTE`は`aic_threshold_route.{weight,threshold,polarity,mask}`をチェックポイントへ追加し、`config.json`の`aic_threshold_routes`を有効にする。標準Transformers互換の重みだけではないため、この動的ゲートを実行できるのはAICランタイムである。LoRA出力では追加テンソルを表現できないためコンパイルエラーになる。

---

## ケース2：非線形多様体（円環 $S^1$）と `@attend` のテッセレーション合成

### 背景
「今日は月曜日。3日後は何曜日？」のような周期的モジュラ状態を、離散ブール演算の捏造ではなく、**曲率 $\kappa > 0$ の円環多様体 `S1`** 上の回転写像として記述し、`@attend` による文脈オフセット取得と組み合わせて SwiGLU ゲート群へ自動テッセレーション展開する。

### 2.1 `Strata` ソースコード

```strata
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
```

### 2.2 C-IR テッセレーションとデュアルバックエンド出力

1. **Atlas Tessellation Pass（第4部アルゴリズム1）**:
   - 円環多様体 `Weekday : S1`（曲率 $\kappa = 1.0$, 単位半径 $R = 1$, 回転写像の2階微分上界 $M_2 = 1.0$）に対し、契約 `approx <= 1e-2`（$\delta = 0.01$）から必要チャート数 $K$ を算出する：
     $$K = \left\lceil \pi R \sqrt{\frac{M_2}{2\delta}} \right\rceil = \left\lceil \pi \sqrt{\frac{1.0}{0.02}} \right\rceil = \lceil 22.21 \rceil = 23 \implies \text{安全係数適用 } K = 24 \text{ チャート}$$
   - 各角度中心 $\theta_j = \frac{2\pi j}{24}$ に対し、局所チャートゲート $w_{\text{gate}, j} = \beta(\hat{c}_j - \cos(\Delta\theta) e_0)$ および局所回転書き込み $(w_{\text{up}, j}, w_{\text{down}, j})$ を生成する。
2. **W-ISA 生成**:
   - Attention 部（層9）：`ATTN_BILINEAR(layer=9, head_group=GQA_3, W_qk=..., W_ov=..., rope_mode="rel_pos")` により `delta` を残差上の `ShiftDelta<1>` 軸へ書き込む。
   - MLP 部（層10〜11）：`24` 個の `SWIGLU_POLE` 命令を発行し、`delta` に応じて円環接空間方向への回転を実行。
3. **ターゲット別出力**:
   - `--target=in-place`: 層10〜11内の休眠極リストから24スロットを確保し、Nullspace射影して書き込む（休眠極が24個未満なら即座に `ApproximationUnmetError` を出力）。
   - `--target=elastic`: `config.json` の `intermediate_size` を $+24$ 拡張し、既存知識との干渉 0 で無損失配置する。

---

## ケース3：連続的な暗黙知（`.field`）の `decouple` スカルプティング

### 背景
モデルが持つ「共感的な対話トーン（`EmpatheticTone`）」という好ましい連続暗黙知は維持したまま、そこに統計的に絡み付いている「ユーザーに迎合して事実を曲げる癖（`SycophancyDrift`）」の暗黙知場だけを引き剥がし、同時に「論理的粘り強さ（`RigorousReasoning`）」を調律して補強する。

### 3.1 `Strata` ソースコード

```strata
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
```

### 3.2 Fisher直交ソルバと生成命令

1. **型検査**: `preserve(EmpatheticTone, ...)` が指定されているため、もし `state -| SycophancyDrift` を直接書けば `CrosstalkExceededError` で拒否される。`decouple EmpatheticTone` によって生成された `pure_sycophancy` のみが直交証明（Witness）を持ち、契約検査を通過する。
2. **計算とW-ISA**:
   - 第4部アルゴリズム3Bに従い、層区間 `Mid..Late` のパラメータ空間で $\langle F_{\text{pure}}, F_{\text{empathy}} \rangle_G = 0$ を満たすランク12の直交化スペクトル差分を算出。
   - `FIELD_SPECTRAL_OP(layer_range=Mid..Late, U_k=..., V_k=..., coeff_delta=-0.85 * sigma_pure)` および推論場強化の `FIELD_SPECTRAL_OP` を一括発行する。

---

## ケース4：`.rodata` 知識外科手術 ＆ 異種モデル間クロスコンパイル（Qwen3-32B → Llama-8B）

### 背景
(A) 高曲率記憶テーブル（`.rodata`）のみを安全に改変・忘却しつつ、(B) 大型モデル（例：`Qwen3-32B`）上で発見・設計された事実整合性チェック回路を、隠れ次元・層数・GQA比率の異なる小型モデル（例：`Llama-8B`）へ `aic link` で移植（クロスコンパイル）する。

### 4.1 `Strata` ソースコード

```strata
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
```

### 4.2 クロスコンパイル実行トレース（`aic link`）

```bash
aic link FactSurgeryAndTransplant.strata \
    --target-model llama-8b.safetensors \
    --atlas llama-8b.natlas \
    --target=in-place \
    --out llama-8b-patched.safetensors
```

1. **M-IR 不変保持**: ソース中の `Frame<Mid..Late>` はまず抽象正規化深度 $[0.25, 0.75)$ として扱われる。`Qwen3-32B`（全64層, $d=5120$）の特定層番号ではなく抽象座標と `anchor` が M-IR に保持される。
2. **動的 `anchor` アライメント（`Llama-8B`：全32層, $d'=4096$, GQA 8 KVヘッド）**:
   - `aic link` は `llama-8b.safetensors` に対して `CapitalKnowledge` と `FactConsistencyCheck` のアンカープローブを実行し、`Mid..Late` が物理層 `10..23` に対応することを自動解決する。
   - 4次元検証部分空間 `FactConsistencyCheck<4>` のソース座標とターゲット上の発見空間との間で $4 \times 4$ 直交Procrustes回転 $R^*$ を解く。
3. **ターゲット固有 W-ISA 低下**:
   - `.rodata` 操作はターゲットの $d'=4096$ 共分散逆行列 $C_{\text{llama}}^{-1}$ を用いた `ASSOC_MEM_ROW` 3命令へコンパイルされる。
   - `FactConsistencyCheck` 介入は `Llama-8B` の `GQA` 共有ヘッド制約および `Nullspace` 保護検査を通過した `VPD_RANK1` + `SWIGLU_POLE` パッチとして生成され、無傷の推論能力を保ったまま移植が完了する。
