# 第3部：言語構文仕様とEBNF文法

本仕様書では、重み空間高級言語 `Strata`（ファイル拡張子 `.strata`）の字句構造、ASCIIシステム構文とUnicode数学記法のデュアル対応表、曖昧さのない完全な **LL(2) 再帰下降パース可能なEBNF文法**、および中核構文からM-IRへの脱糖（Desugaring）規則を定義する。

---

## 1. 字句規則とデュアル記法（ASCII ⇄ Unicode数学記号）

`Strata` は、標準的なエディタ・Git・コンパイラ実装での扱いやすさを優先し、**Rust / ML 系のASCII構文を正規のソース表現（Canonical Source Form）**として採用する。同時に、ドキュメントやIDE上の数式レンダリング（Ligature表示）向けに1対1のUnicode数学記法を定義する。

### 1.1 演算子・記号対応表

| 正規ASCIIトークン | Unicode表示記法 | 名称 | 分類・役割 |
| :--- | :--- | :--- | :--- |
| `~>` | `▷` | Resonance | 多様体チャート／極への共鳴ゲート判定・座標取得 |
| `+>` | `⊕` | Excitation | 接空間方向への加算励起（`state +> dir`） |
| `-|` | `⊣` | Annihilation | 逆位相直交射影による能動的対消滅（`state -| sub`） |
| `=>` | `↦` | Transport / Map | 多様体間写像および `.rodata` キー値対応 |
| `@attend` | `⋆` | Relational Field | 因果マスク付き非局所双線形Attention演算 |
| `->` | `→` | Frame Connection / Return | 層座標遷移および `flow` 戻り値型 |
| `decouple` | `⊥` | Fisher Orthogonalize | 巨視的連続場 `Field` の計量直交分離 |
| `![...]` | `⟦...⟧` | Effect Contract | `crosstalk`, `approx`, `energy`, `preserve` 静的契約 |

### 1.2 字句定義（Lexical Grammar）

```ebnf
(* 空白・コメント *)
WHITESPACE   = " " | "\t" | "\r" | "\n" ;
LINE_COMMENT = "//" , { ANY_CHAR_EXCEPT_NEWLINE } , "\n" ;
BLOCK_COMMENT = "/*" , { ANY_CHAR } , "*/" ;

(* 識別子とリテラル *)
IDENT        = ( ASCII_ALPHA | "_" ) , { ASCII_ALNUM | "_" } ;
INT          = DIGIT , { DIGIT } ;
FLOAT        = DIGIT , { DIGIT } , "." , DIGIT , { DIGIT } , [ EXPONENT ]
             | DIGIT , { DIGIT } , EXPONENT ;
EXPONENT     = ( "e" | "E" ) , [ "+" | "-" ] , DIGIT , { DIGIT } ;
STRING       = '"' , { ESCAPED_CHAR | ANY_CHAR_EXCEPT_QUOTE } , '"' ;
ESCAPED_CHAR = "\" , ( '"' | "\" | "n" | "r" | "t" ) ;
```

> [!IMPORTANT]
> **字句解析上の最長一致と範囲リテラルの衝突回避**：
> `10..15` のような範囲指定が誤って `10.`（`FLOAT`）と `.15` に分割されることを防ぐため、字句解析器は `.` の直後に1文字以上の数字 `[0-9]` が続く場合に限り `FLOAT` として認識する（先読み1文字規則）。したがって `10..15` は必ず `INT(10)`, `DOTDOT("..")`, `INT(15)` としてトークン化される。

### 1.3 予約語（Keywords）と文脈キーワード

以下の語は厳密予約語であり、識別子（`IDENT`）としては使用できない：

```text
module  import  as  frame  connect  to  via  in
pole  dipole  subspace  manifold  field  rodata
anchor  entry  with
flow  let  when  else  transport  insert  prune  rewrite
decouple  temper  blend  preserve  crosstalk  approx  energy
true  false  Frame  const
```

一方、`chart`, `metric`, `rank`, `curvature` の4語は**文脈キーワード（Contextual Keywords）**であり、宣言ブロック（`decl_body` / `rodata_body`）の先頭でのみメタデータ属性として解釈され、名前付き引数ラベル（例：`chart: "angle"`, `metric: "fisher"`, `curvature: 19.5`）や式位置では通常の識別子 `IDENT` として扱われる。

加えて、`test`, `expect`, `resonance`, `kl_drift`, `top_token`, `logit_margin`, `base` の7語は**文脈キーワード（Contextual Keywords）**として `test_decl` および `expect_stmt` の先頭位置でのみ特別解釈され、それ以外の識別子位置では通常の `IDENT` として扱われる。

---

## 2. 完全なEBNF文法（ISO/IEC 14977 準拠）

`Strata` の構文は、先読み最大2トークン（`LL(2)`）でバックトラックなしに一意に構文解析できるよう設計されている。独立したコンパイル単位は先頭に `module` 宣言を持つが、対話的スニペットや局所パッチ記述では `module` 宣言を省略できる。

```ebnf
(* ========================================================================= *)
(* Strata Formal EBNF Grammar v1.0                                           *)
(* ========================================================================= *)

program         = [ module_decl ] , { import_decl } , { top_item } ;

module_decl     = "module" , path , ";" ;
import_decl     = "import" , path , [ "as" , IDENT ] , ";" ;
path            = IDENT , { "::" , IDENT } ;

top_item        = { annotation } , (
                      const_decl
                    | frame_decl
                    | pole_decl
                    | dipole_decl
                    | subspace_decl
                    | manifold_decl
                    | field_decl
                    | rodata_decl
                    | flow_decl
                    | test_decl
                  ) ;

const_decl      = "const" , IDENT , [ ":" , type_ref ] , "=" , expr , ";" ;

test_decl       = "test" , IDENT , "{" , { expect_stmt } , "}" ;
expect_stmt     = "expect" , expect_body , ";" ;
expect_body     = "resonance" , "(" , expr , "," , symbol_ref , ")" , cmp_op , rhs_expr
                | "kl_drift" , "(" , ( expr | list_lit ) , ")" , cmp_op , expr
                | "top_token" , "(" , expr , ")" , ( "==" | "!=" ) , expr
                | "logit_margin" , "(" , expr , "," , expr , "," , expr , ")" , cmp_op , rhs_expr
                | "crosstalk" , cmp_op , expr
                | "approx" , cmp_op , expr ;
symbol_ref      = path , [ "." , ( "+" | "-" ) , "pole" ] ;
rhs_expr        = "base" , [ ( "+" | "-" ) , expr ] | expr ;
cmp_op          = "<" | "<=" | ">" | ">=" | "==" | "!=" ;

annotation      = "@" , IDENT , [ "(" , arg_list , ")" ] ;

(* --- 1. 層座標系 (Frame) 宣言 --- *)
frame_decl      = "frame" , IDENT , [ "in" , stage_spec ] ,
                  ( "{" , { frame_member } , "}" | ";" ) ;
frame_member    = "connect" , "to" , IDENT , [ "via" , IDENT ] , ";"
                | anchor_block ;

stage_spec      = stage_bound , [ ( "..=" | ".." ) , stage_bound ] ;
stage_bound     = IDENT | INT ;
frame_ref       = "Frame" , "<" , stage_spec , ">" | path ;

(* --- 2. 幾何学型・暗黙知場・知識テーブル宣言 --- *)
pole_decl       = "pole" , IDENT , "in" , frame_ref , ( decl_body | ";" ) ;

dipole_decl     = "dipole" , IDENT , "in" , frame_ref , "{" , { dipole_member } , "}" ;
dipole_member   = ( "+" | "-" ) , "pole" , ":" , IDENT ,
                  ( "{" , { anchor_block } , "}" | ";" )
                | decl_member ;

subspace_decl   = "subspace" , IDENT , "<" , INT , ">" , "in" , frame_ref ,
                  ( decl_body | ";" ) ;

manifold_decl   = "manifold" , IDENT , ":" , topo_spec , "in" , frame_ref ,
                  ( decl_body | ";" ) ;
topo_spec       = IDENT , [ "<" , type_arg_list , ">" ] ;
type_arg_list   = type_arg , { "," , type_arg } ;
type_arg        = INT | FLOAT | IDENT ;

field_decl      = "field" , IDENT , [ ":" , topo_spec ] , "in" , frame_ref ,
                  ( decl_body | ";" ) ;

rodata_decl     = "rodata" , IDENT , "<" , type_ref , "," , type_ref , ">" ,
                  "in" , frame_ref , ( rodata_body | ";" ) ;
rodata_body     = "{" , { rodata_item } , "}" ;
rodata_item     = "entry" , cmp_expr , "=>" , cmp_expr ,
                  [ "with" , "(" , arg_list , ")" ] , ";"
                | decl_member ;

decl_body       = "{" , { decl_member } , "}" ;
decl_member     = anchor_block
                | "chart" , IDENT , ":" , expr , ";"
                | "metric" , IDENT , ";"
                | "rank" , INT , ";"
                | "curvature" , expr , ";" ;

(* --- 3. 概念アンカー宣言 --- *)
anchor_block    = "anchor" , IDENT , "(" , [ arg_list ] , ")" , ";" ;

(* --- 4. 型参照 --- *)
type_ref        = IDENT , [ "<" , type_arg_or_ref_list , ">" ] ;
type_arg_or_ref_list = type_arg_or_ref , { "," , type_arg_or_ref } ;
type_arg_or_ref = INT | FLOAT | type_ref ;

(* --- 5. フロー（計算・編集ユニット）宣言 --- *)
flow_decl       = "flow" , IDENT , [ "(" , [ param_list ] , ")" ] ,
                  [ "->" , type_ref , [ "in" , frame_ref ] ] ,
                  [ effect_contract ] , block ;

param_list      = param , { "," , param } ;
param           = IDENT , ":" , type_ref , [ "in" , frame_ref ] ;

(* 補助フロー呼び出しは postfix_expr の path "(" arg_list ")" 形式で表現する。
   helper flow invocation ::= IDENT "(" [ arg_list ] ")" ";"
   これにより呼び出し元フローは effect semiring を継承し、コンパイラは
   is_helper == true のフローをインライン展開してから MIR を生成する。 *)

effect_contract = "![" , constraint_list , "]" ;
constraint_list = constraint , { "," , constraint } ;
constraint      = "crosstalk" , cmp_op , number_lit
                | "approx" , cmp_op , number_lit
                | "energy" , cmp_op , number_lit
                | "preserve" , "(" , ident_list , ")" ;
cmp_op          = "<=" | ">=" | "==" | "<" | ">" ;
ident_list      = path , { "," , path } ;

(* --- 6. 文とブロック --- *)
block           = "{" , { stmt } , "}" ;

stmt            = let_stmt
                | when_stmt
                | rodata_stmt
                | action_or_expr_stmt ;

let_stmt        = "let" , IDENT , [ ":" , type_ref ] , "=" , expr , ";" ;

when_stmt       = "when" , expr , [ "as" , IDENT ] , block ,
                  [ "else" , ( when_stmt | block ) ] ;

rodata_stmt     = ( "insert" | "prune" | "rewrite" ) , path ,
                  "(" , [ arg_list ] , ")" , ";" ;

action_or_expr_stmt = expr , [ ( "+>" | "-|" ) , expr ] , ";" ;

(* --- 7. 式と演算子優先順位（低 → 高） --- *)
expr            = field_expr ;

field_expr      = logic_or_expr ,
                  { ( "decouple" | "temper" | "blend" ) , logic_or_expr ,
                    [ "with" , "(" , arg_list , ")" ] } ;

logic_or_expr   = logic_and_expr , { "||" , logic_and_expr } ;
logic_and_expr  = flow_op_expr , { "&&" , flow_op_expr } ;

flow_op_expr    = range_expr , { ( "=>" | "~>" ) , range_expr } ;

range_expr      = cmp_expr , [ ( "..=" | ".." ) , cmp_expr ] ;

cmp_expr        = add_expr , [ ( "<=" | ">=" | "==" | "!=" | "<" | ">" ) , add_expr ] ;

add_expr        = mul_expr , { ( "+" | "-" ) , mul_expr } ;

mul_expr        = unary_expr , { ( "*" | "/" ) , unary_expr } ;

unary_expr      = [ "-" | "+" | "!" ] , postfix_expr ;

postfix_expr    = primary_expr , {
                      "." , member_name , [ "(" , [ arg_list ] , ")" ]
                    | "(" , [ arg_list ] , ")"
                  } ;

member_name     = IDENT | "+" , "pole" | "-" , "pole" ;

primary_expr    = FLOAT
                | INT
                | STRING
                | "true"
                | "false"
                | attend_expr
                | transport_call
                | field_call_expr
                | list_lit
                | "(" , expr , ")"
                | path ;

attend_expr     = "@attend" , "(" , arg_list , ")" ;
transport_call  = "transport" , "(" , expr , "," , frame_ref , "->" , frame_ref , ")" ;
field_call_expr = ( "decouple" | "temper" | "blend" ) , "(" , arg_list , ")" ;

list_lit        = "[" , [ expr , { "," , expr } ] , "]" ;

arg_list        = arg , { "," , arg } ;
arg             = [ IDENT , ":" ] , expr ;

number_lit      = [ "-" ] , ( FLOAT | INT ) ;
```

---

## 3. 曖昧性解消と構文規則の設計根拠

1. **双極メンバアクセス（`.+pole` / `.-pole`）の一意性**:
   `postfix_expr` において `.` の直後に `+` または `-` が来た場合、次に来得るトークンは必ずキーワード `pole` のみに限定される。二項演算子の `+` / `-` が `.` の直後に現れることはないため、構文上の曖昧さは一切存在しない。
2. **名前付き引数（`gain: 0.45`）の LL(2) 判定**:
   `arg` のパース時、先頭トークンが `IDENT` かつ 2トークン目が `:` であればラベル付き引数、それ以外は位置引数式として分岐する（`:` は式内部に出現しない）。
3. **連想テーブル定義 `entry K => V` の一意性**:
   `rodata_item` における `K` と `V` は `=>` を含まない `cmp_expr` としてパースされるため、エントリ区切りの `=>` が `flow_op_expr` として誤吸収されることはない。
4. **暗黙の環境状態束縛（`state`）**:
   `flow Foo ![...] { ... }` のように引数リストを省略したとき、コンパイラは自動的に `state: Residual in Frame<FlowStage>` を環境へ束縛する。多引数や明示的出力フレームを持つ `flow Bar(ctx: Residual in Frame<Mid>) -> Residual in Frame<Late>` の記述も同一文法でサポートされる。
5. **`const` 宣言のスコープと型推論**:
   `const IDENT = expr ;` はモジュールスコープで定数を宣言し、フロー内の `let` 文や式から参照できる。型注釈を省略したときコンパイラは `expr` から型を推論する（`INT` リテラルは `Scalar<f64>`, `FLOAT` リテラルは `Scalar<f64>`）。
6. **補助フロー呼び出し（Sub-flow invocation）の Effect Semiring 継承**:
   `helper_flow(arg);` は `action_or_expr_stmt` → `postfix_expr` として解析される。フロントエンドは `is_helper == true` の補助フローをインライン展開し、呼び出し元フローの効果契約（`crosstalk`, `approx`, `preserve`）を引き継いだ上で M-IR を生成する。補助フローへ渡された引数の型はパラメータ型と単一化される。

7. **`test` / `expect` 文脈キーワードの非破壊的解析**:
   `test` および `expect` はトップレベルでのみ文脈キーワードとして認識される。`test IDENT {` の形式がトップレベルに現れた場合に限り `test_decl` として解析され、フロー内部での識別子 `test` の使用は従来通り `IDENT` として扱われる。同様に、`expect` は `test_decl` ブロックの内部でのみ `expect_stmt` の先頭トークンとして解釈される。`resonance`、`kl_drift`、`top_token`、`logit_margin`、`base` も同一スコープ規則に従い、`test_decl` 外では識別子として利用可能である。`top_token(prompt) == target` は貪欲デコードの次トークン一致を検証し、`logit_margin(prompt, target, foil)` は `logits[target] - logits[foil]` をベースモデル（`base`）または数値閾値と比較する。

---

## 4. 脱糖（Desugaring）および正規化規則

フロントエンドは AST から M-IR へ変換する前に、以下の決定論的な脱糖を行う。

1. **修飾引数付き右辺演算子の標準化**:
   ```strata
   flow DesugarExample {
       let p = F1 decouple F2(metric: "fisher");
   }
   ```
   `F2(metric: "fisher")` のようなフィールド式右辺へのパラメータ付与は、関数形式へ正規化される：
   $$
   \text{desugar}(F_1 \mathbin{\text{decouple}} F_2(\text{args})) \equiv \text{decouple}(F_1, F_2, \text{args})
   $$

2. **`when` 複合論理条件（`&&` / `||`）の展開**:
   - `when (e1 ~> T1) && (e2 ~> T2) { body }` は、SwiGLU の積ゲート合成、または入れ子の共鳴ガード `when e1 ~> T1 { when e2 ~> T2 { body } }` へ脱糖される。
   - `when (e1 ~> T1) || (e2 ~> T2) { body }` は、2つの独立した並列共鳴パス（同一本体を持つ2つのチャートゲート、重ね合わせ上限は効果契約で抑制）へ展開される。

3. **アノテーション（`@stage`, `@target`）の伝播**:
   `top_item` に付与された `@stage(10..15)` は、内部に含まれるすべての幾何宣言および `flow` の `Frame` 上下界制約（$l_{\min} = 10, l_{\max} = 15$）として型環境へ注入される。
