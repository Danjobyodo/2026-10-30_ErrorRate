# 非クリフォード回路を古典計算機で厳密に再現した方法

作成日: 2026-10-30（Claude）／ 同日改訂：Takada–Bartlett–Williamson 論文の本文（`2026-09-29_ExactEfficientSimulation.tex`）と照合して §2–§6・§8 を書き直した

対象: `chsc_switch_v3.py`（本番エンジン）と `chsc_validation_v3.py`（正当化コード）。

この文書は、CHSC の code switching（folded surface code → D4 符号 → 理想な論理 $\overline{T}^\dagger$ → surface code）を、量子計算機を使わずに**近似なし**で古典計算機上でサンプリングする方法をまとめたものです。

- 方式の骨格は Takada–Bartlett–Williamson（以下 TBW）の **Clifford-stabilizer simulation** です（§5）。
- それを D4 符号に当てはめたときに厳密であることを、この研究に固有の条件 C1–C3 で確かめています（§2.3）。

`README_v3.md` は「方式の詳細は `../v2/README_v2.md` §3・§4 を参照」としていますが、v2 のファイルはこのリポジトリに含まれていません。そのため、その内容をここに書き起こしました。

- コード中のコメント `README_v3.md Sec. 3`／`Sec. 3.3`／`Sec. 3.4` が指しているのも、本書 §2–§3 の内容です（§8）。
- 以下で「TBW Sec. ○」と書くのは、上の .tex ファイルの節番号です（I 序論、II 準備、III Clifford-stabilizer group、IV 対角の場合、V cultivation への応用、VI 一般の場合、VII 多項式時間アルゴリズム、VIII 適用範囲、IX 議論）。

---

## 1. 何が「非クリフォード」で、なぜ素朴には古典シミュレーションできないか

### 1.1 1 ショットで起きること

各ショットは次の順に進みます（`ShotRunner.run`）。

1. **初期状態**: $\lvert+\rangle_{\rm SC}$（赤・青の folded surface code）$\otimes \lvert0\rangle^{\otimes}_{g}$（緑）。
2. **switch-in**（$d$ ラウンド）を繰り返します。各ラウンドは次の 4 段階です。
   1. 全量子ビットに確率 $p$ の脱分極雑音をかけます。
   2. 全色の $Z$ チェックを測定します（結果は確率 $p$ で反転）。
   3. JIT 復号器でフラックスを即時補正します（$X$ 補正）。
   4. dressed $X$ チェック $S^{c}_{X;v}=X^{s}\prod CZ$ を r→g→b の順に測定します（結果は確率 $p$ で反転）。
3. **switch-in の終わり方**（README §2 参照）
   - `ideal`: 無雑音の 1 ラウンドを置き、そのラウンドの結果でフラックスと電荷を後始末します。
   - `noisy`: 無雑音ラウンドを置かずに後始末します。
4. **論理ゲート**: 理想な $U=WV$ をかけます（`in_T_out` のときのみ）。
   - $V=\prod_e {\rm CNOT}^{(r,g)}_e{\rm CNOT}^{(b,g)}_e$
   - $W=\prod {\rm CS}^{r,b}({\rm CS}^{r,b})^\dagger\prod_{e\in L_{rb}}(T^r_e)^\dagger$
   - $U$ は D4 符号の論理 $\overline{T}^\dagger$ です（CHSC 論文 Theorem 2、$U\overline{\lvert m\rangle}=e^{-i\pi m/4}\overline{\lvert m\rangle}$）。
   - 「論理 T ゲート」は、論文どおりここでは $T^\dagger$ として実装されています。目標状態は $\overline{T}^\dagger\overline{\lvert+\rangle}$ です。
5. **switch-out**
   1. 脱分極雑音をかけます。
   2. 全緑量子ビットの $Z^g$ を 1 回だけ測定します（結果は確率 $p$ で反転）。
   3. 測定結果をマッチングで閉じた形 $m'=\delta\lambda$ に直し、緑の補正 $S^g_X(\lambda)$ をユニタリとしてかけます。
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

TBW も同じ問題意識から出発しています（TBW Sec. I）。

- 状態ベクトル法は量子ビット数について指数的です。
- stabilizer 分解法は非クリフォードゲート数について指数的です。

---

## 2. 鍵となる構造: D4 符号は「対角型のクリフォード安定化子符号」

### 2.1 TBW の言葉での位置づけ

TBW は、Pauli 安定化子と非 Pauli のクリフォード演算子（**Clifford stabilizer**）の共通 $+1$ 固有状態として決まる状態を「Clifford-stabilizer state」と呼びます（TBW Sec. III）。これを決める群（Clifford-stabilizer group）は非可換でもかまいません。

TBW は特に次の「対角の場合」を詳しく扱っています（TBW Sec. IV）。

- クリフォード安定化子が「対角クリフォード × Pauli $X$」の形をしている。
- 生成元どうしの群交換子 $[G_i,G_j]=G_iG_jG_i^\dagger G_j^\dagger$ が $Z$ 型の Pauli 安定化子になる。

TBW はそこで「これまでに研究されたクリフォード安定化子符号の多くもこの条件を満たす」と書き、Davydova et al. の符号を引用しています。

D4 符号（CHSC 論文の Clifford stabilizer code）は、まさにこの対角の場合に当たります。

- dressed 生成元は $G=X^{s_G}\prod_{(i,j)}CZ_{ij}$（対角クリフォード × Pauli $X$）です。
- 群交換子は $Z$ チェックの積（$Z$ 型 Pauli 安定化子）です。CHSC 論文の言う「$Z$ 安定化子の部分空間の中でだけ可換」に当たります。
- 各量子ビットの Pauli 誤りが $CZ$ 因子に付ける「Pauli dressing」は、単一量子ビットの $Z$ だけです（$X_iCZ_{ij}X_iCZ_{ij}=Z_j$）。重み 1 で互いに可換なので、TBW の admissible の条件（Definition 2 の条件 2）の形をしています。
- TBW のやり方で Pauli proxy を作ると、**dressed 生成元から $CZ$ を取り除いた裸の $X$ 星 $X^{s_G}$** になります（TBW Sec. VII.B の構成、§3.2）。コードもこの proxy を使っています。

### 2.2 平坦部分空間と位相 $\varphi_0$

$Z$ チェック（各色のプラケット、$L_{rb}$ 上の $Z^rZ^b$）がすべて $+1$ になる配置を「平坦」と呼び、その集合を $C_0=\ker(Z_{\rm mat})$ とします。

- `TriangleCode` は $C_0$ の基底として、次の 2 つを取ります。
  - 独立な $X$ 生成元の台 $s_G$
  - 論理 $\overline{Z}$ と奇数回交わる非自明な平坦配置 $\eta$（緑成分は 0）
- 座標は $a=M_0t$ で、$h(a)=\overline{Z}\cdot a$ が論理ビットです。

D4 符号の論理基底状態は、次の形に書けます。

$$
\overline{\lvert h\rangle}\;\propto\sum_{a\in C_0,\;h(a)=h}(-1)^{\varphi_0(a)}\lvert a\rangle .
$$

位相 $\varphi_0$ は、各 dressed 生成元について次の関係を満たすように決まります（`phi0_walk`）。

$$
\varphi_0(a+s_G)=\varphi_0(a)+q_G(a),\qquad q_G(a)=\sum_{(i,j)\in G}a_ia_j\pmod 2 .
$$

- $q_G$ が 2 次なので、$\varphi_0$ は一般に 3 次です。
- これは、Dijkgraaf–Witten 項 $(-1)^{\int a_r\cup a_g\cup a_b}$ が 3 次であることに対応します。

平坦部分空間上の対角ユニタリ $C={\rm diag}\big((-1)^{\varphi_0(a)}\big)$ を定義します。$C$ は CCZ 型の（第 3 階層の）対角ゲートに相当し、シミュレーションの中で**一度も作用させません**。

### 2.3 Pauli proxy への置き換えが厳密である根拠: 条件 C1–C3（`check_intertwiner_conditions`）

TBW は、Pauli proxy への置き換えを「演算子の等式でも近似でもなく、Pauli dressing との可換・反可換関係と生成元どうしの可換性を保つ置き換え規則」として導入しています。そして、admissible という条件の下で、測定結果の分布が厳密に再現されることを一般的に証明しています（TBW Sec. VI–VII、Theorem 1）。

このコードは、TBW の admissible の各条件をこの符号について形式的に確かめる代わりに、**この符号に限った、より直接的な演算子の等式**で厳密性を示しています。

- **(C1)** 平坦部分空間上で、すべての dressed 生成元について $C\,X^{s_G}\,C^\dagger=G$ が成り立ちます（2.2 節の式と同値）。
  - したがって、真の状態と proxy 状態は、平坦部分空間上のユニタリ $C$ で厳密に結ばれます。
  - dressed 生成元 $G$ の Pauli proxy が裸の $X$ 星 $X^{s_G}$ でよいことが、演算子の等式として分かります。
- **(C2)** 緑成分が 0 の平坦配置では $\varphi_0=0$ です。
  - したがって、$C$ は folded surface code の状態に自明に作用します。
  - 初期状態（$\lvert+\rangle_{\rm SC}\otimes\lvert0\rangle^g$）と switch-out 後の状態は、そのまま Pauli 安定化子状態として扱えます。
- **(C3)** 平坦部分空間上で $U\,C=C\,V\,T_h$ が成り立ちます。ここで $T_h={\rm diag}(\omega^{\kappa h(a)})$、$\omega=e^{i\pi/4}$、$\kappa=7$（$T^\dagger$）です。
  - これは CHSC 論文 Theorem 2 を配置ごとに書いた形です。
  - **$U$ の Pauli proxy は CNOT 層 $V$**（$W$ の対角非クリフォードゲートを取り除いたもの）です。TBW Sec. VIII.A の「proxy 状態は、典型的には $U$ の対角非クリフォードゲートを取り除けば得られる」という記述と同じ結論です。
  - 残る $T_h$ は論理ビット $h$ だけに依存する対角因子で、最後の論理判定でだけ意味を持ちます（§3.6）。
  - $h$ は赤・青の量子ビットだけで決まり、$V$ は緑だけを書き換えるので、$V$ と $T_h$ は可換です。

C1–C3 の証明は、生成元ごとの代数的な恒等式です（C1 は 2.2 節の式、C3 は CHSC 論文 SM の Theorem 2 の証明）。

- コードの自己診断は、ランダムな平坦配置でこれを数値的に確かめる**実装検査**です。全配置についての証明ではありません。
- 2026-10-30 に d=3, 5, 7 で再実行し、違反 0 を確認しました。
- 厳密性は、§4 の総当たり状態ベクトルとのショットごとの比較でも独立に確かめています。
- TBW の admissible の条件 1（1 次元の固有空間）と条件 3（primary-support exactness）を、この符号について形式的に確認したわけではありません。上の C1–C3 と §4 の比較で代えています。

---

## 3. 測定共役描像と誤り演算子 E

### 3.1 状態の表し方（TBW Sec. VII.A: measurement-conjugated picture）

TBW は、誤りを状態に作用させる代わりに測定演算子の側に移します。誤り $E$ の後で $M$ を測ることは、$E^\dagger ME$ を「フレーム状態」で測ってから $E$ をかけることと同じです（$\Pi_m(M)E=E\,\Pi_m(E^\dagger ME)$）。

このコードでは、各ショットの真の状態を常に次の形で持ちます（$U$ の前は $T_h$ がありません）。

$$
\lvert\Psi\rangle \;=\; E\;C\;T_h^{[\text{U 後のみ}]}\;\lvert\phi\rangle .
$$

| 記号 | 中身 | 誰が持つか |
|---|---|---|
| $\lvert\phi\rangle$ | 平坦部分空間内の **Pauli 安定化子状態**（proxy 状態。TBW の「Pauli-proxy stabilizer tableau」） | `stim.TableauSimulator` |
| $E$ | 誤り演算子（雑音・補正・枝の反転の累積）。$E=X^{x}\,{\rm diag}(\omega^{\gamma(a)})$、$\gamma(a)=l\cdot a+4\sum_{i<j}Q_{ij}a_ia_j \pmod 8$ | `ErrorFrame`（$x$, $l$, $Q$ の配列） |
| $C$ | §2.2 の 3 次位相 | 作用させない（C1・C2 で消える） |
| $T_h$ | 論理 $T^\dagger$ | 作用させない（最終判定で $\kappa$ として使う） |

- $E$ は「Pauli $X$ フレーム × 対角クリフォード（$Z$, $S$, $CZ$）」の形に閉じています。$U$ の前は Pauli（$Q=0$、$l\in\{0,4\}$）です。
- TBW Sec. VII.D の言葉では、伝播を続ける必要がある「continuing data error」は $X$ 成分だけです。$Z$ と対角クリフォードは対角な因子と可換だからです。
  - このコードは $E$ を左から掛けて更新し続け、測定のたびに共役をとります。これは「誤りを回路の後方へ伝播させる」ことと同じです。
- $\lvert\phi\rangle$ は常に平坦部分空間にあります。$\lvert\phi\rangle$ にかかる $X$ 型演算子はすべて $C_0$ の元（生成元の台、$V$、緑の補正 $\delta\lambda$）で、$C_0$ の外に出る $X$ 誤りはすべて $E$ 側に置くからです。
  - このため、$Z$ チェックの結果は $\lvert\phi\rangle$ では常に $+1$ です。実際の結果は $Z_{\rm mat}\cdot x$ から古典的に決まります（`zchecks`）。
  - これは TBW の「群交換子に現れる Pauli 演算子（$\mathcal{B}_{\rm gc}$、ここでは $Z$ チェック）はフレーム状態で $+1$」（TBW Sec. VII.A）に当たります。

### 3.2 測定（TBW Sec. VII.A–C: Pauli dressing と Pauli proxy、Algorithm 1）

演算子 $M=X^s\Delta$（$\Delta$ は dressing の $CZ$ 積）を $\lvert\Psi\rangle$ で測ることは、$C^\dagger E^\dagger M E\, C$ を $\lvert\phi\rangle$ で測ることと同じです。

1. **Pauli dressing**: $E^\dagger M E=X^s\Delta\cdot D$ と書けます。$D={\rm diag}(\omega^{[\delta(a+x)-\delta(a)]+[\gamma(a)-\gamma(a+s)]})$ は、$Z$ 列と符号からなる Pauli dressing です（`ErrorFrame.measured`）。
2. **Pauli proxy**: C1 により $C^\dagger X^s\Delta\,C=X^s$ です。$D$ は対角なので $C$ と可換です。
   - 結局、測るのは Pauli 演算子 $\pm X^{s}Z^{L}$ です。これを Stim の `MPP` で測ります。
   - 本番経路では、1 ラウンド分をまとめて 1 回の Stim 呼び出しにし、符号は後から結果に XOR します。
   - 位相が $\pm1$ 以外（非エルミート）になったら止まります。TBW は「Pauli-dressed 演算子がエルミートなら、その proxy もエルミート」と示しています（TBW Sec. VII.B）。
3. **$T_h$ との可換性**: 測定する台 $s$ は安定化子の台なので $\overline{Z}\cdot s=0$ で、dressing は対角です。したがって、測定演算子は $T_h$ と可換で、$T_h$ を Stim に入れる必要がありません。
4. surface code の裸の $X$ 星（switch-out 後）は、C2 によりそのまま Pauli 測定です。

**ランダムな結果の扱い**

- switch-in の最初の緑 dressed チェックは、初期状態の $Z^g_e$ 安定化子と反可換なので、結果はランダムです（gauging）。
- TBW はこれを「Pauli 安定化子状態の上でクリフォード安定化子を測る measurement-based injection」として扱い、「Clifford 演算子の gauging による論理測定」にも適用できると明記しています（TBW Sec. VIII.B。引用先は Davydova et al.）。
- この場合の枝反転演算子（$Z^g$）は、後の測定と両立（compatible）します。そのため、proxy のタブローで Stim が符号をランダムに決めるだけで正しく扱えます（TBW Sec. VIII.B 前半）。

### 3.3 理想な論理ゲート $U$（TBW Sec. VIII.A: transversal non-Clifford gates）

`apply_U` は 2 つのことをします。

- Stim の状態には $V$（CNOT 層）だけをかけます（C3）。
- $E\leftarrow UEU^\dagger$ を古典的に更新します（`conj_U`）。
  - $V$: $X^r\to X^rX^g$、$X^b\to X^bX^g$、$Z^g\to Z^rZ^gZ^b$。
  - $W$ との交換: $X$ が CS の片側だけにあれば S 型＋CZ 型、両側にあれば S 型の位相、$T^\dagger$ を通る $X$ には S 型の位相が付きます。

TBW Sec. VIII.A の言うとおり、「Pauli 誤りを対角な第 3 階層ゲートで共役すると、対角クリフォード × Pauli になる」ので、$E$ は表現 $X^x{\rm diag}(\omega^\gamma)$ に収まります。

現在のコードは、$U$ の時点で $E$ が Pauli であることを確かめてから処理しています。このプロトコルでは $U$ は 1 回だけで、その前の誤りは Pauli だけだからです。$E$ に対角クリフォード部分がある場合への拡張については §6.3 を参照してください。

### 3.4 回路途中の補正と switch-out（TBW Sec. VII.D、Sec. VIII.B、Algorithm 3）

**回路途中の補正**

JIT の $X$ 補正、電荷の $Z$ 補正、緑の補正 $S^g_X(\lambda)$ は、どれも途中の測定結果で決まる補正です。TBW Sec. VII.D は、こうした補正を「誤りとして加えて伝播させる（＝ proxy のクリフォード回路を動的に変える）」として扱います。コードでも、これらは $E$ に左から掛けるだけです（`apply_x`／`apply_z`／`apply_gen_unitaries`）。

**switch-out の $Z^g$ 測定（`measure_green_switch_out`）**

TBW Sec. VIII.B の「データ量子ビットの単一量子ビット $Z$ 測定」に当たります。その枝反転演算子は dressed な $S^g_X(\lambda)$（$CZ^{b,r}$ を含む）で、後で測る surface code の裸の $X$ 星とは可換でも反可換でもありません（両立しない）。そこで TBW の Algorithm 3 の手順になります。コードは次の順に処理します。

1. $E^\dagger Z_q E=(-1)^{x_q}Z_q$ なので、Stim で $Z^g$ を測り、物理的な結果を $x$ から復元します。
2. Stim の状態の結果 $m$ は、平坦性から $m=\delta\lambda_{\rm fr}$ になります。これを解いて確認します。
3. proxy 状態に $X^{m}$（枝反転演算子の proxy）をかけて、符号を $+1$ の枝に戻します（Algorithm 3 の Step 2）。
4. 真の枝反転演算子 $S^g_X(\lambda_{\rm fr})$ を $E$ の**右から**掛けて吸収します（`right_xcz`。Step 3 の「補正として適用」に当たります）。

その後、復号器が雑音入りの測定値から選んだ $S^g_X(\lambda)$ を、クリフォードのユニタリとして $E$ の左から掛けます。復号の誤りで $\lambda\ne\lambda_{\rm fr}$ になると、$E$ に CZ 型のクリフォード誤りが残り、それも厳密に追跡されます。

### 3.5 計算量

1 ショットで行う処理は次のとおりです。

- $E$ の更新と dressing の計算: 量子ビット数 $n$ について高々 $O(n^2)$
- Stim のタブロー操作
- PyMatching による復号

すべて多項式時間です。TBW Sec. VII.E の一般的な評価と同じく、目標状態の stabilizer rank には依存しません。

手元の CPU（2026-10-30）で `in_T_out`, ideal, $p\approx1.35\%$ のとき、1 ショットあたり d=3 で約 2.3 ms、d=5 で約 10 ms、d=7 で約 22 ms でした（Colab での値は §7）。

### 3.6 論理状態の判定（TBW Sec. VI.D: 架空のクリフォード安定化子測定）

TBW Sec. VI.D は、最後に無雑音の架空ラウンドを置き、そのあと「論理自由度を固定するクリフォード安定化子」を架空に測って論理誤りを判定します。架空測定の結果は次の 3 通りです。

- (i) 決定的に $+1$
- (ii) 決定的に $-1$
- (iii) 現在の生成元と反可換（結果がランダム）

(i) だけが成功です。

このコードの判定は、これをそのまま実装したものです（`logical_operators`／`evaluate_class`）。

1. 目標状態 $\overline{T}^\dagger\overline{\lvert+\rangle}$ を固定するクリフォード安定化子は $\overline{T}^\dagger\overline{X}\,\overline{T}$ です。その Pauli proxy は $\eta$ 台の $X$（論理 $X$）です。
2. これを $E$ で共役して dressing を付けます。$X$ フレームが $\overline{Z}$ と反可換なときに掛かる因子 $i^{-\kappa}\overline{Z}$ は、このクリフォード安定化子の dressing です（§2.3 の $T_h$ はここでだけ現れます）。
3. Stim の `peek_observable_expectation` で期待値を見ます。
   - $+1$ なら `I`（成功、(i)）、$-1$ なら `Z`（(ii)）です。
   - 期待値 0（(iii)）のときは、$\overline{Z}$ が確定していれば `collapsed`、$\overline{Y}$ 型の演算子で `X`/`Y`/`S` に細かく分類します。
4. $P_L=1-\Pr[\texttt{I}]$ です。

確率的なサンプリングは「雑音の位置」と「ランダムな測定結果」についてだけ行います。各ショットの量子状態の扱いは厳密です。

---

## 4. 正当化（この方法が近似ではないことの確認）

TBW は、自分たちの方法の正しさを、cultivation（$f=3$）で状態ベクトル法と $10^8$ サンプルの統計で比べて確かめています（TBW Sec. V）。このリポジトリでは、それより強い**ショットごとの一致**で確かめています。

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
- 参照 3 は、ショットごとのクラスの内訳がログと少し違いますが、不一致は 0 です。これは Stim 自身の乱数を使うためです（Stim の seed 付き乱数は、計算機が変わると同じ seed でも同じ列になるとは限りません）。

---

## 5. Takada–Bartlett–Williamson 論文との関係

**文献**: Y. Takada, S. D. Bartlett, D. J. Williamson, "Exact efficient simulation of noisy logical magic states using Clifford stabilizers", arXiv:2609.16929（2026 年 9 月 15 日投稿）。本文は図なしの TeX が `2026-09-29_ExactEfficientSimulation.tex` にあります。

### 5.1 TBW の内容（本文から）

- 論理マジック状態の多くは、Pauli 安定化子と非 Pauli のクリフォード安定化子の共通 $+1$ 固有状態として一意に決まります（例: surface code の $\lvert\bar T\rangle$、[[8,3,2]] の $\lvert\overline{CCZ}\rangle$、Steane の $\lvert\bar H\rangle$）。
  - これを決める群を Clifford-stabilizer group と呼び、回路レベルの Pauli 雑音の下での更新規則を与えています（Sec. III, IV, VI）。
- 更新は、次の 2 つの道具で Pauli 安定化子群の更新に写せます。
  - 測定共役描像
  - Pauli proxy（クリフォード因子を、Pauli dressing との可換関係を保つ Pauli 演算子で置き換える規則）
- その結果、「雑音入りの非クリフォード回路を、測定結果の分布がまったく同じクリフォード回路に写す」アルゴリズムになります（Sec. VII、Algorithm 1–2）。時間・空間とも多項式で、stabilizer rank や Pauli rank の分解を使いません。
- 適用範囲には、次の操作が含まれます（Sec. VIII）。
  - 符号化回路、横断的な非クリフォードゲート、CNOT による符号の成長
  - 非プロトコル測定（単一量子ビット $Z$ 測定、measurement-based injection、Clifford 演算子の gauging 測定）
  - 回路途中の補正
- 応用として、magic state cultivation を fault distance 7 まで厳密にシミュレーションしています（Sec. V）。

### 5.2 このコードとの対応（コード中の引用はすべて本文と一致することを確認）

| TBW の概念（節） | このコードでの実体 |
|---|---|
| Clifford-stabilizer group／対角の場合（III, IV） | D4 符号の dressed 生成元 $X^{s}\prod CZ$ と $Z$ チェック（§2.1） |
| 測定共役描像、フレーム状態（VII.A） | `ErrorFrame`（$E$）と Stim の proxy 状態（§3.1） |
| Pauli dressing（IV.A, VI.A） | `ErrorFrame.measured` が返す $Z$ 列 $L$ と位相 |
| Pauli proxy（VII.B）、Algorithm 1（VII.C） | dressed 生成元 → $X^{s_G}$（C1）、Stim の `MPP` |
| 回路途中の補正（VII.D） | JIT の $X$ 補正、電荷の $Z$ 補正、$S^g_X(\lambda)$ を $E$ に掛ける |
| 横断的な非クリフォードゲート（VIII.A） | `apply_U`／`conj_U`。proxy は $V$（C3、§3.3） |
| measurement-based injection、gauging 測定（VIII.B） | switch-in の緑 dressed チェック（§3.2） |
| 非プロトコル測定と枝反転、Algorithm 3（VIII.B） | switch-out の $Z^g$ 測定、`right_xcz`（§3.4） |
| 架空のクリフォード安定化子測定による論理誤りの判定（VI.D） | `logical_operators`／`evaluate_class`（§3.6） |
| proxy のクリフォード回路の実行 | `stim.TableauSimulator` |

### 5.3 この研究に固有の点・違い

1. **適用対象**: TBW の主な例は「Pauli 符号の上のマジック状態」の準備（cultivation）です。この研究では、**符号そのもの**（D4 符号）がクリフォード安定化子で定義されています。
   - TBW は Discussion（Sec. IX）で、Davydova et al. の TQD（twisted quantum double）マジック状態準備プロトコルを、この方法で回路レベルの厳密シミュレーションができる対象として名指ししています。
   - Davydova et al. は、CHSC 論文が比較対象とし、JIT 復号器を借りているプロトコルです。CHSC の code switching はその近縁にあたり、TBW 自身が想定した適用先の一つと言えます。
2. **厳密性の示し方**: TBW は admissible の条件から一般定理として示しています。このコードは、平坦部分空間上の演算子の等式 C1–C3（$C$ による intertwining）と、総当たりとのショットごとの比較で示しています（§2.3、§4）。どちらでも Pauli proxy は同じもの（$CZ$ を取り除いた $X$ 星、$W$ を取り除いた $V$）になります。
3. **雑音モデル**: TBW は回路レベルの Pauli 雑音（ancilla の誤りがデータに $S$ や $CZ$ のクリフォード誤りとして伝わる hook error を含む）を扱います。このコードは phenomenological 雑音だけです（§6）。

### 5.4 関連研究

- **Surti, Daguerre, Kim**, "Efficient Simulation of Logical Magic State Preparation Protocols", PRX Quantum 7, 020329 (2026)（TBW の文献 fby6-xjbm）。
  - 第 3 階層のマジック状態準備を、量子ビット数・非クリフォードゲート数・目標状態の stabilizer/Pauli rank について多項式時間でシミュレーションします。
  - TBW は、rank が指数的に増える場合の差と、誤り伝播のコストの差を論じています（TBW Sec. VII.E）。
- **Magdalena de la Fuente, Feldman, Eisert, Bauer**, "High-threshold decoding of non-Pauli codes for 2D universality", arXiv:2604.02033 (2026)。
  - 非 Pauli 符号（$X$ 安定化子が対角クリフォードでねじれた CSS 型符号）の JIT 復号を、phenomenological 雑音で評価しています。
  - 要旨によれば、閾値は約 2.5%（全シンドローム履歴を使う復号器では約 2.9%）です。非クリフォードゲートのために素朴な $Z$ 復号では約 1.8% に下がり、$X$ 補正の情報を使うと約 2.2% に改善します。
  - TBW は、この論文の評価手順は論理誤り率の上界しか与えないと述べています（TBW Sec. I）。
  - Switching 先の D4 符号自体の閾値を調べる別研究と直接関係します。この研究の README §2 の「U をまたぐ復号」の議論（$X$ 補正の情報を $Z$ 復号に使う）とも問題意識が近いです。
  - ただし、量（符号のメモリの閾値か、switching ガジェットの交点か）が違うので、数値は直接比べられません。
- **Davydova, Bauer, Magdalena de la Fuente, Webster, Williamson, Brown**, "Universal fault tolerant quantum computation in 2D without getting tied in knots", Phys. Rev. X 16, 031001 (2026)（arXiv:2503.15751、TBW の文献 vrty-qs5h、CHSC 論文の Davydova:2025ylx）。

---

## 6. この方法でできること・できないこと（割り切り）

### 6.1 厳密に扱えている範囲

次の条件を満たす範囲では、測定結果の分布と最終の論理状態クラスは、真の量子力学と完全に一致します。

- Pauli 雑音（ここでは phenomenological な脱分極雑音と測定結果の反転）
- 対角な第 3 階層の論理ゲート $U$ 1 回
- 任意の古典復号器と、その結果にもとづく回路途中の補正

### 6.2 実装していないが、モデルとして置いている前提

`README_v3.md` §7 にも書いた前提です。

- $U$、緑の補正 $S^g_X(\lambda)$、初期化、最終読み出しは無雑音です。
- switch-out の $Z^g$ 測定は 1 回だけ（single-shot）です。
- 雑音は phenomenological で、回路レベルではありません。
- 入力は $\lvert+\rangle$ に固定しています。

### 6.3 TBW の範囲内で拡張できるもの（未実装）

1. **回路レベル雑音**
   - dressed $X$ チェックは、制御 $X$ と制御 $CZ$（= CCZ）の列で測ります。ancilla の $X$ 誤りはデータ上の $CZ$ 誤り（hook error）になります。
   - これは TBW が対角の場合として扱っている状況そのものです（TBW Sec. IV.A の Fig. 5(b)、Sec. VII.D）。$E$ の表現（$X$ × 対角クリフォード）にも収まります。
   - 必要なのは、測定回路を決めることと、$E$ に回路途中の対角クリフォード誤りを入れる実装です。測定回路は Davydova et al. が微視的な回路を与えているので、それを使えます。
2. **$E$ に対角クリフォード部分がある状態での $U$**、つまり $U$ の前に hook error がある場合や、対角な第 3 階層ゲートを何度もかける場合
   - 対角クリフォード誤りは、対角非クリフォードゲート $W$ と可換です。CNOT 層 $V$ を通っても対角クリフォードのままです（TBW Sec. VIII.A）。
   - したがって、$E$ は表現の中に収まり続けます。
   - 現在の `conj_U` は $E$ が Pauli であることを前提にしているので、拡張が必要です。

> 訂正: この文書の初版では「誤りの入った後にもう一度非クリフォード層をかけると、第 3 階層の誤りになり扱えない」と書いていました。これは、対角な第 3 階層ゲートについては誤りです。上の 2. のとおり、対角クリフォード誤りは $W$ と可換なので、表現の中に収まります。

### 6.4 この方式では原理的に扱えないもの

1. **第 4 階層以上のゲートや安定化子**
   - TBW は第 3 階層のマジック状態（クリフォード安定化子で決まる状態）に限られます。
   - たとえば CHSC 論文の 3D 符号の論理 $\sqrt{T}$（第 4 階層）を通ると、$X$ 誤りは $T$ 型の非クリフォード誤りになり、$E$ の表現から外れます。
   - その 3D 符号の安定化子自体も CCZ を含む非クリフォード演算子です。
   - TBW も、より高い階層の安定化子群への一般化を今後の課題として挙げています（TBW Sec. IX）。
2. **コヒーレント誤りなど、Pauli でない雑音**。TBW は Pauli 雑音を前提にしています。Pauli twirling などの近似が必要です。
3. **現在の生成元と可換でも反可換でもない測定**で、結果の確率が 0, 1/2, 1 以外になるもの。TBW も今後の課題としています（Sec. IX）。このプロトコルには現れません。

**$U$ をまたぐ復号窓**（README §2.3）は、復号器（古典アルゴリズム）の研究課題です。シミュレーション方式の制約ではありません。この方式のまま組み込めます（§3.4 のとおり、補正は $E$ に掛けるだけです）。

---

## 7. 実行時間の目安

| 環境 | d=3 | d=5 | d=7 | 既定（1 ケース、8 点 × 20000 ショット） |
|---|---|---|---|---|
| Colab の CPU ランタイム（`chsc_v3_results/` の実測） | 約 3.0–3.6 ms/ショット | 約 8.6–13 ms/ショット | 約 23–59 ms/ショット | 約 2.1 時間 |
| 手元の CPU（2026-10-30、$p\approx1.35\%$） | 約 2.3 ms/ショット | 約 10 ms/ショット | 約 22 ms/ショット | — |

1 ショットあたりの時間は $p$ が大きいほど長くなります（復号の負荷が増えるため）。

---

## 8. コード中の参照と本書・TBW 本文の対応

| コード中の記述 | 対応する内容 |
|---|---|
| `README_v3.md Sec. 3`（Cell 5 冒頭、`ProxyBackend`） | 本書 §2.3・§3 |
| `README_v3.md Sec. 3.3`（`check_intertwiner_conditions`、`apply_U`） | 本書 §2.3（C1–C3）・§3.3 |
| `README_v3.md Sec. 3.4`（`measure_green_switch_out`） | 本書 §3.4 |
| `report Sec. 3`／`report Sec. 4, A1`（v1 由来のコメント） | v1 レポート（このリポジトリには含まれていません）。A1 は「$L_r$／$L_b$／$L_{rb}$ 上の単一量子ビット $Z$ 安定化子で固定される量子ビットを、理想な $\lvert0\rangle$ として省く」という前提です（`TriangleCode` の docstring） |
| TBW "Sec. VII.A-C, Algorithm 1"（ファイル冒頭） | VII.A 測定共役描像、VII.B Pauli proxy、VII.C 演算子レベルのアルゴリズム（Algorithm 1）。本文と一致 |
| TBW "Sec. VII.D, VIII.A-B, Algorithm 3" | VII.D 回路レベルの実装（回路途中の補正を含む）、VIII.A ユニタリゲート（横断的な非クリフォードゲート）、VIII.B 測定（枝反転、Algorithm 3 = 非プロトコル測定）。本文と一致 |
| TBW "Sec. VI.D" | VI.D 論理誤りの評価（架空のクリフォード安定化子測定）。本文と一致 |
| TBW "Algorithm 3"（`right_xcz`、`measure_green_switch_out`） | Algorithm 3（非プロトコル測定）の Step 2–3。本文と一致 |
