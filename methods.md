# 非クリフォード回路を古典計算機で厳密に再現した方法

作成日: 2026-10-30（Claude）

対象: `chsc_switch_v3.py`（本番エンジン）と `chsc_validation_v3.py`（正当化コード）。

この文書は、CHSC の code switching（folded surface code → D4 符号 → 理想な論理 $\overline{T}^\dagger$ → surface code）を、量子計算機を使わずに**近似なし**で古典計算機上でサンプリングする方法をまとめたものです。

- 方式の骨格は Takada–Bartlett–Williamson（以下 TBW）の **Clifford-stabilizer simulation** です。
- これを D4 符号（非可換なクリフォード安定化子符号）に当てはめるために、この研究に固有の条件 C1–C3 を使っています。

`README_v3.md` は「方式の詳細は `../v2/README_v2.md` §3・§4 を参照」としていますが、v2 のファイルはこのリポジトリに含まれていません。そのため、その内容をここに改めて書き起こしました。コード中のコメント `README_v3.md Sec. 3`／`Sec. 3.3`／`Sec. 3.4` が指しているのも、本書の §2–§3 の内容です（§8 の対応表を参照）。

---

## 1. 何が「非クリフォード」で、なぜ素朴には古典シミュレーションできないか

### 1.1 1 ショットで起きること

各ショットは次の順に進みます（`ShotRunner.run`）。

1. **初期状態**: $\lvert+\rangle_{\rm SC}$（赤・青の folded surface code）$\otimes \lvert0\rangle^{\otimes}_{g}$（緑）。
2. **switch-in**（$d$ ラウンド）を繰り返します。各ラウンドは次の 4 段階です。
   1. 全量子ビットに確率 $p$ の脱分極雑音をかけます。
   2. 全色の $Z$ チェックを測定します（結果は確率 $p$ で反転）。
   3. JIT 復号器でフラックスを即時補正します。
   4. dressed $X$ チェック $S^{c}_{X;v}=X^{s}\prod CZ$ を r→g→b の順に測定します（結果は確率 $p$ で反転）。
3. **switch-in の終わり方**（README §2 参照）
   - `ideal`: 無雑音の 1 ラウンドを置き、そのラウンドの結果でフラックスと電荷を後始末します。
   - `noisy`: 無雑音ラウンドを置かずに後始末します。
4. **論理ゲート**: 理想な $U=WV$ をかけます（`in_T_out` のときのみ）。
   - $V=\prod_e {\rm CNOT}^{(r,g)}_e{\rm CNOT}^{(b,g)}_e$
   - $W=\prod {\rm CS}^{r,b}({\rm CS}^{r,b})^\dagger\prod_{e\in L_{rb}}(T^r_e)^\dagger$
   - $U$ は D4 符号の論理 $\overline{T}^\dagger$ です（論文 Theorem 2、$U\overline{\lvert m\rangle}=e^{-i\pi m/4}\overline{\lvert m\rangle}$）。
   - ユーザーの言う「論理 T ゲート」は、論文どおりここでは $T^\dagger$ として実装されています。目標状態は $\overline{T}^\dagger\overline{\lvert+\rangle}$ です。
5. **switch-out**
   1. 脱分極雑音をかけます。
   2. 全緑量子ビットの $Z^g$ を 1 回だけ測定します（結果は確率 $p$ で反転）。
   3. 測定結果をマッチングで閉じた形に直し、その $m'=\delta\lambda$ に対応する緑の補正 $S^g_X(\lambda)$ をユニタリとしてかけます。
6. **surface code**: $d$ ラウンドの雑音ありラウンドと、無雑音の 1 ラウンドを行い、時空 MWPM で復号します。
7. **判定**: 最終状態が目標状態そのものかどうかを判定します（§3.6）。

### 1.2 Stim だけでは扱えない 4 つの要素

| 要素 | なぜ Pauli 安定化子形式（Stim）の外か |
|---|---|
| (a) D4 符号の状態そのもの | 計算基底 $\lvert a\rangle$（$a$ = 平坦なゲージ場配置）の振幅に位相 $(-1)^{\varphi_0(a)}$ が付きます。$\varphi_0$ は GF(2) 上で一般に 3 次の関数（§2.2）なので、この状態は Pauli 安定化子状態ではありません。 |
| (b) dressed $X$ チェックの測定 | 測定する演算子 $X^s\prod CZ$ は Pauli ではなくクリフォード演算子です。生成元どうしも演算子としては非可換で、平坦部分空間でだけ可換になります（`check_flat_commutation`: d=3/5/7 で非可換な対はそれぞれ 12/48/108、平坦部分空間ではすべて 0）。 |
| (c) $U$ の $W$（CS, $T^\dagger$） | クリフォード階層の第 3 階層のゲートです。 |
| (d) $U$ を通過した誤り | Pauli $X$ 誤りが $W$ を通ると、$XWXW^\dagger$ が S 型・CZ 型の**クリフォード誤り**になります。 |

状態ベクトルを総当たりで持つと、メモリは $2^{k}$ です（$k$ は平坦配置の座標数）。$d=2$ でも $k=12$（4096 振幅）で、$d\ge3$ ではすぐ破綻します。

---

## 2. 鍵となる構造: 平坦部分空間・位相 $\varphi_0$・intertwiner $C$

### 2.1 平坦部分空間

$Z$ チェック（各色のプラケット、$L_{rb}$ 上の $Z^rZ^b$）がすべて $+1$ になる配置を「平坦」と呼び、その集合を $C_0=\ker(Z_{\rm mat})$ とします。

- `TriangleCode` は $C_0$ の基底として、次の 2 つを取ります。
  - 独立な $X$ 生成元の台 $s_G$
  - 論理 $\overline{Z}$ と奇数回交わる非自明な平坦配置 $\eta$（緑成分は 0）
- 座標は $a=M_0t$、$t\in{\rm GF}(2)^k$ です。
- $h(a)=\overline{Z}\cdot a$ が論理ビットです。

### 2.2 D4 符号状態の位相 $\varphi_0$ と $C$

D4 符号の論理基底状態は、次の形に書けます。

$$
\overline{\lvert h\rangle}\;\propto\sum_{a\in C_0,\;h(a)=h}(-1)^{\varphi_0(a)}\lvert a\rangle .
$$

位相 $\varphi_0$ は、各 dressed 生成元 $G=X^{s_G}\prod_{(i,j)}CZ_{ij}$ について、次の関係を満たすように決まります（`phi0_walk`）。

$$
\varphi_0(a+s_G)=\varphi_0(a)+q_G(a)\pmod 2,\qquad q_G(a)=\sum_{(i,j)\in G}a_ia_j .
$$

- $q_G$ が 2 次なので、$\varphi_0$ は一般に 3 次の位相関数です。
- これは、twisted $\mathbb{Z}_2^3$ ゲージ理論の Dijkgraaf–Witten 項 $(-1)^{\int a_r\cup a_g\cup a_b}$ が 3 次であることに対応します。

平坦部分空間上の対角ユニタリ $C={\rm diag}\big((-1)^{\varphi_0(a)}\big)$ を定義します。$C$ は CCZ 型の（第 3 階層の）対角ゲートに相当し、シミュレーションの中で**一度も作用させません**。

### 2.3 Pauli proxy への置き換えを厳密にする 3 条件（`check_intertwiner_conditions`）

- **(C1)** 平坦部分空間上で、すべての dressed 生成元について $C\,X^{s_G}\,C^\dagger=G$ が成り立ちます。
  - これは 2.2 節の関係式と同値です。
  - したがって、D4 符号空間は「$C\times$（$X^{s_G}$ と $Z$ チェックを安定化子とする、ねじれのない $\mathbb{Z}_2^3$ の Pauli 符号空間）」です。
  - **dressed 生成元 $G$ の Pauli proxy は、裸の $X$ 星 $X^{s_G}$** です。
- **(C2)** 緑成分が 0 の平坦配置では $\varphi_0=0$ です。
  - したがって、$C$ は folded surface code の状態に自明に作用します。
  - 初期状態と switch-out 後の状態は、そのまま Pauli 安定化子状態として扱えます。
- **(C3)** 平坦部分空間上で $U\,C=C\,V\,T_h$ が成り立ちます。ここで $T_h={\rm diag}(\omega^{\kappa h(a)})$、$\omega=e^{i\pi/4}$、$\kappa=7$（$T^\dagger$）です。
  - これは論文 Theorem 2 を配置ごとに書いた形です。
  - **$U$ の Pauli proxy は CNOT 層 $V$ だけ**です。非クリフォードの論理作用は、論理ビット $h$ だけに依存する対角因子 $T_h$ として外に出せます。
  - $h$ は赤・青の量子ビットだけで決まり、$V$ は緑だけを書き換えるので、$V$ と $T_h$ は可換です。

C1–C3 の証明は、生成元ごとの代数的な恒等式です（C1 は 2.2 節の式、C3 は論文 SM の Theorem 2 の証明）。

- コードの自己診断は、ランダムな平坦配置でこれを数値的に確かめる**実装検査**です。全配置についての証明ではありません。
- 2026-10-30 に d=3, 5, 7 で再実行し、違反 0 を確認しました。

---

## 3. 測定共役描像と誤り演算子 E（TBW の Clifford-stabilizer simulation）

### 3.1 状態の表し方

各ショットの真の状態は、常に次の形で表せます（$U$ の前は $T_h$ がありません）。

$$
\lvert\Psi\rangle \;=\; E\;C\;T_h^{[\text{U 後のみ}]}\;\lvert\phi\rangle .
$$

| 記号 | 中身 | 誰が持つか |
|---|---|---|
| $\lvert\phi\rangle$ | 平坦部分空間内の **Pauli 安定化子状態**（proxy 状態） | `stim.TableauSimulator` |
| $E$ | 誤り演算子（雑音と補正の累積）。$E=X^{x}\,{\rm diag}(\omega^{\gamma(a)})$、$\gamma(a)=l\cdot a+4\sum_{i<j}Q_{ij}a_ia_j \pmod 8$ | `ErrorFrame`（$x$, $l$, $Q$ の配列） |
| $C$ | §2.2 の 3 次位相 | 作用させない（C1・C2 で消える） |
| $T_h$ | 論理 $T^\dagger$ | 作用させない（最終判定で $\kappa$ として使う） |

- $E$ は「Pauli $X$ フレーム × 対角クリフォード（$Z$, $S$, $CZ$）」の形に閉じています。
- $U$ の前は、$E$ は Pauli（$Q=0$、$l\in\{0,4\}$）です。
- $\lvert\phi\rangle$ は常に平坦部分空間にあります。$\lvert\phi\rangle$ にかかる $X$ 型演算子はすべて $C_0$ の元（生成元の台、$V$、緑の補正 $\delta\lambda$）で、$C_0$ の外に出る $X$ 誤りはすべて $E$ 側に置くからです。このため、$Z$ チェックの結果は $\lvert\phi\rangle$ では常に $+1$ で、実際の結果は $Z_{\rm mat}\cdot x$ から古典的に決まります（`zchecks`）。

### 3.2 測定（TBW: measurement-conjugated picture + Pauli dressing + Pauli proxy）

演算子 $M=X^s\Delta$（$\Delta$ は dressing の $CZ$ 積）を $\lvert\Psi\rangle$ で測ることは、$C^\dagger E^\dagger M E\, C$ を $\lvert\phi\rangle$ で測ることと同じです。

1. **Pauli dressing**: $E^\dagger M E=X^s\Delta\cdot D$ と書けます。$D={\rm diag}(\omega^{[\delta(a+x)-\delta(a)]+[\gamma(a)-\gamma(a+s)]})$ は、$Z$ 列と符号からなる「Pauli dressing」です（`ErrorFrame.measured`）。
2. **Pauli proxy**: C1 により $C^\dagger X^s\Delta\,C=X^s$ です。$D$ は対角なので $C$ と可換です。
   - 結局、測るのは Pauli 演算子 $\pm X^{s}Z^{L}$ です。これを Stim の `MPP` で測ります。
   - 本番経路では、1 ラウンド分をまとめて 1 回の Stim 呼び出しにし、符号は後から結果に XOR します。
   - 位相が $\pm1$ 以外（非エルミート）になったら `AssertionError` で止まります。
3. **$T_h$ との可換性**: 測定する台 $s$ は安定化子の台なので $\overline{Z}\cdot s=0$ で、dressing は対角です。したがって、測定演算子は $T_h$ と可換で、$T_h$ を Stim に入れる必要がありません。
4. surface code の裸の $X$ 星（switch-out 後）は、C2 によりそのまま Pauli 測定です。

### 3.3 理想な論理ゲート $U$（TBW: transversal non-Clifford gate）

`apply_U` は 2 つのことをします。

- Stim の状態には $V$（CNOT 層）だけをかけます（C3）。
- $E\leftarrow UEU^\dagger$ を古典的に更新します（`conj_U`）。
  - $V$: $X^r\to X^rX^g$、$X^b\to X^bX^g$、$Z^g\to Z^rZ^gZ^b$。
  - $W$ との交換: $X$ が CS の片側だけにあれば S 型＋CZ 型、両側にあれば S 型の位相、$T^\dagger$ を通る $X$ には S 型の位相が付きます。

この操作は **$E$ が Pauli のときだけ**閉じます（コードは確認して、違えば止まります）。Pauli 誤りが第 3 階層のゲートを通ると第 2 階層（クリフォード）の誤りになり、表現 $E=X^x{\rm diag}(\omega^\gamma)$ の中に収まるからです。

### 3.4 switch-out の $Z^g$ 測定と枝の正規化（TBW: branch-flipping / mid-circuit Clifford corrections）

`measure_green_switch_out` は次の順に処理します。

1. $E^\dagger Z_q E=(-1)^{x_q}Z_q$ なので、Stim で $Z^g$ を測り、物理的な結果を $x$ から復元します。
2. Stim の状態の結果 $m$ は、平坦性から $m=\delta\lambda_{\rm fr}$（緑生成元の台の和）になります。これを解いて確認します。
3. proxy 状態には $X^{m}$（$S^g_X(\lambda_{\rm fr})$ の proxy）をかけて $Z^g=+1$ にそろえます。
4. 真の $S^g_X(\lambda_{\rm fr})$（$CZ^{b,r}$ の dressing を含む）は、$E$ の**右から**掛けて吸収します（`right_xcz`）。

その後、復号器が雑音入りの測定値から選んだ $S^g_X(\lambda)$ を、クリフォードのユニタリとして $E$ の左から掛けます（`apply_gen_unitaries`）。復号の誤りで $\lambda\ne\lambda_{\rm fr}$ になると、$E$ に CZ 型のクリフォード誤りが残り、それも厳密に追跡されます。

### 3.5 計算量

1 ショットで行う処理は次のとおりです。

- $E$ の更新と dressing の計算: 量子ビット数 $n$ について高々 $O(n^2)$
- Stim のタブロー操作
- PyMatching による復号

すべて多項式時間です。手元の CPU（2026-10-30）で `in_T_out`, ideal, $p\approx1.35\%$ のとき、1 ショットあたり d=3 で約 2.3 ms、d=5 で約 10 ms、d=7 で約 22 ms でした。

### 3.6 論理状態の判定（TBW: fictitious Clifford-stabilizer measurement）

目標状態 $\overline{T}^\dagger\overline{\lvert+\rangle}$ は、クリフォード演算子 $\overline{T}^\dagger\overline{X}\,\overline{T}$ の $+1$ 固有状態です。この「目標状態の安定化子」を架空に測った期待値で判定します。

1. 架空の測定演算子を作ります（`logical_operators`）。
   - $\eta$ 台の $X$（論理 $X$ の proxy）を、$E$ で共役して dressing を付けます。
   - $X$ フレームが $\overline{Z}$ と反可換なら、$T_h$ を通る補正因子 $i^{-\kappa}\overline{Z}$ を掛けます。
2. Stim の `peek_observable_expectation` で期待値を見ます。
   - $+1$ なら `I`（成功）、$-1$ なら `Z` です。
   - それ以外は、$\overline{Z}$ が確定していれば `collapsed`、$\overline{Y}$ 型の演算子の期待値で `X`/`Y`/`S` に分類します。
3. $P_L=1-\Pr[\texttt{I}]$ です。

確率的なサンプリングは「雑音の位置」と「ランダムな測定結果」についてだけ行います。各ショットの量子状態の扱いは厳密です。

---

## 4. 正当化（この方法が近似ではないことの確認）

| 検査 | 内容 | 2026-10-30 の再実行結果 |
|---|---|---|
| 自己診断（d=3,5,7） | 平坦部分空間での可換性、$U$ が論理 $T^\dagger$（$\kappa=7$）、C1–C3、$p=0$ で $P_L=0$、最終状態が符号空間 | すべて合格（約 9 秒） |
| $U$ の総当たり | d=2 で $c_1/c_0=e^{-i\pi/4}$ | $T^\dagger$ |
| 参照 1: 総当たり状態ベクトル（ロックステップ） | d=2、$2^{12}$ 振幅に全演算子を文字どおり作用 | 3600 ショット、433,608 出力、**不一致 0** |
| 参照 2: v1 の Z4 twisted-frame エンジン（ロックステップ） | d=3, 4。別の表現（座標 $t$ 上の線形制約と $\mathbb{Z}_4$ 値の位相形式）による多項式時間の厳密エンジン | 4800 ショット、2,636,318 出力、**不一致 0** |
| 参照 3: 本番経路の再生 | `lockstep=False`（MPP 一括・符号ビット・Stim 自身の乱数）の全結果を総当たりに再生 | 1200 ショット、144,520 出力、**不一致 0** |
| 合否判定 | `assert_validation` | **合格** |

- 「ロックステップ」とは、同じ雑音・同じ復号器・同じ乱数ビットで本番エンジンと参照を同時に動かし、全測定の「決定的／ランダムの別」と結果、$Z$ チェック、$X$ フレーム、最終クラスを 1 ショットずつ比べる方法です。
- 参照 1・2 の出力は `参考_検証ログ/validation_log.txt` と 1 行ずつ完全に一致しました。
- 参照 3 は、ショットごとのクラスの内訳がログと少し違いますが、不一致は 0 です。これは Stim 自身の乱数を使うためです（Stim の seed 付き乱数は、計算機やバージョンが変わると同じ seed でも同じ列になるとは限りません）。

---

## 5. Takada–Bartlett–Williamson 論文との関係

**文献**: Y. Takada, S. D. Bartlett, D. J. Williamson, "Exact efficient simulation of noisy logical magic states using Clifford stabilizers", arXiv:2609.16929（2026 年 9 月）。

**注意**: この論文の本文はリポジトリに含まれていません。リポジトリにあるのは CHSC 論文（本文と SM の TeX）だけです。また、この環境からは arxiv.org に接続できませんでした。以下は次の 2 つに基づいています。

- 検索で得た要旨の内容
- コードのコメントに書かれた節番号（Sec. VI.D, VII.A–D, VIII.A–B, Algorithm 1, 3）

これらの節番号は本文で照合できていません。

**TBW の要旨（検索結果から）**

- クリフォード回路と Pauli 誤りは効率よくシミュレーションできますが、論理マジック状態に必要な非クリフォード操作が入ると、一般には効率が失われます。
- TBW は、第 3 階層のマジック状態を準備する雑音入りの回路を、**同じ測定結果分布を持つクリフォード回路に写す** Clifford-stabilizer simulation を提案しています。時間・空間とも多項式です。
- 中心となる道具は、クリフォード安定化子を置き換える **Pauli proxy** と、誤りの共役で付く **Pauli dressing** です。これらで「Pauli-proxy 安定化子群」を作ります。
- 応用として、magic state cultivation を fault distance 7 まで厳密にシミュレーションしています。

**この研究のコードとの対応**

| TBW の概念 | このコードでの実体 |
|---|---|
| 測定共役描像（誤りを測定演算子の側に移す） | `ErrorFrame`、§3.1–3.2 |
| Pauli dressing | `ErrorFrame.measured` が返す $Z$ 列 $L$ と位相 |
| Pauli proxy | dressed 生成元 $G$ → $X^{s_G}$（C1）、$U$ → $V$（C3） |
| 横断的な非クリフォードゲート | `apply_U` / `conj_U`（§3.3） |
| branch-flipping の補正、回路途中のクリフォード補正 | `measure_green_switch_out`（`right_xcz`）、`apply_gen_unitaries`（§3.4） |
| 架空のクリフォード安定化子測定による論理誤りの判定 | `logical_operators` / `evaluate_class`（§3.6） |
| proxy のクリフォード回路の実行 | `stim.TableauSimulator` |

**この研究に固有の点**

- TBW では、Pauli proxy の考え方は主に「目標マジック状態の（クリフォードな）安定化子」に使われます。ここでは同じ置き換えを、**D4 符号の安定化子そのもの**（非可換なクリフォード安定化子）にも使っています。
- その置き換えが厳密であることは TBW の一般論からは自動的には出ません。この格子とこのプロトコルについて、C1–C3 として示し、検査しています（§2.3）。
- CHSC 論文自身は「クリフォード安定化子符号」を導入していて、TBW の「Clifford stabilizer」と同じ数学的対象を扱っています。そのため、この組み合わせは自然です。

**関連研究**: Surti, Daguerre, Kim, "Efficient Simulation of Logical Magic State Preparation Protocols", PRX Quantum 7, 020329 (2026)。非クリフォードゲートを含む雑音入りの回路を、「無雑音の論理マジック状態＋クリフォード誤り」として扱う同系統の手法です。

---

## 6. この方法でできること・できないこと（割り切り）

### 6.1 厳密に扱えている範囲

次の条件を満たす範囲では、測定結果の分布と最終の論理状態クラスは、真の量子力学と完全に一致します。

- Pauli 雑音（ここでは phenomenological な脱分極雑音と測定結果の反転）
- 非クリフォード層 $U$ が 1 回だけで、その時点で誤り $E$ が Pauli であること
- 復号器（JIT、MWPM）は任意の古典アルゴリズムでよいこと

### 6.2 実装していないが、モデルとして置いている前提

`README_v3.md` §7 にも書いた前提です。

- $U$、緑の補正 $S^g_X(\lambda)$、初期化、最終読み出しは無雑音です。
- switch-out の $Z^g$ 測定は 1 回だけ（single-shot）です。
- 雑音は phenomenological で、回路レベルではありません。
- 入力は $\lvert+\rangle$ に固定しています。

### 6.3 この方式では原理的に扱いにくいもの

1. **誤りの入った後にもう一度非クリフォード層をかけること**（例: $U$ を 2 回、$U$ の後に別の $T$ 層）。
   - 2 回目の時点では $E$ がクリフォードなので、第 3 階層のゲートを通ると第 3 階層の誤りになります。
   - そうなると $E=X^x{\rm diag}(\omega^\gamma)$ の形に収まらず、効率のよい厳密計算が一般には失われます。
   - コードはこの状況を検出して止まります。
2. **コヒーレント誤りなど、Pauli でない雑音**。Pauli twirling などの近似が必要です。
3. **回路レベル雑音**。dressed チェックの測定回路には CCZ 型のゲートが入ります。
   - 回路内の $X$ 故障は CZ 型の誤りになるので、$E$ の形に収まる見込みはあります。
   - ただし、測定回路そのものが論文で決まっていません。C1–C3 に相当する条件を回路ごとに確かめ直す必要もあります。
   - このため、未実装です（README §7 の課題 2）。

**$U$ をまたぐ復号窓**（README §2.3）は、復号器（古典アルゴリズム）の研究課題です。シミュレーション方式の制約ではありません。この方式のまま組み込めます。

---

## 7. 実行時間の目安

| 環境 | d=3 | d=5 | d=7 | 既定（1 ケース、8 点 × 20000 ショット） |
|---|---|---|---|---|
| Colab の CPU ランタイム（`chsc_v3_results/` の実測） | 約 3.0–3.6 ms/ショット | 約 8.6–13 ms/ショット | 約 23–59 ms/ショット | 約 2.1 時間 |
| 手元の CPU（2026-10-30、$p\approx1.35\%$） | 約 2.3 ms/ショット | 約 10 ms/ショット | 約 22 ms/ショット | — |

1 ショットあたりの時間は $p$ が大きいほど長くなります（復号の負荷が増えるため）。

---

## 8. コード中の参照と本書の対応

| コード中の記述 | 対応する内容 |
|---|---|
| `README_v3.md Sec. 3`（Cell 5 冒頭、`ProxyBackend`） | 本書 §2.3・§3 |
| `README_v3.md Sec. 3.3`（`check_intertwiner_conditions`、`apply_U`） | 本書 §2.3（C1–C3）・§3.3 |
| `README_v3.md Sec. 3.4`（`measure_green_switch_out`） | 本書 §3.4 |
| `report Sec. 3`／`report Sec. 4, A1`（v1 由来のコメント） | v1 レポート（このリポジトリには含まれていません）。A1 は「$L_r$／$L_b$／$L_{rb}$ 上の単一量子ビット $Z$ 安定化子で固定される量子ビットを、理想な $\lvert0\rangle$ として省く」という前提です（`TriangleCode` の docstring） |
