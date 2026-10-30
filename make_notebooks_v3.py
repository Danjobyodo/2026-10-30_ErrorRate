"""Build the two Colab notebooks from chsc_switch_v3.py and chsc_validation_v3.py (cells split at '# %% [Cell ...]')."""
import json
import re
import sys

SRC_MAIN = "chsc_switch_v3.py"
SRC_VAL = "chsc_validation_v3.py"


def split_cells(path):
    txt = open(path, encoding="utf-8").read()
    parts = re.split(r"^# %% (\[Cell [^\]]+\][^\n]*)\n", txt, flags=re.M)
    head = parts[0]
    cells = []
    for i in range(1, len(parts), 2):
        cells.append((parts[i], parts[i + 1].rstrip() + "\n"))
    return head, cells


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {"cells": cells, "metadata": {"colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


MAIN_DESC = {
    "Cell 1": "**[Cell 1] 設定（CFG）** — `case`（`in_only` / `in_out` / `in_T_out`）、`cases_to_run`（3 ケースまとめて実行）、"
              "`d_values`、`shots`、p の範囲、`switch_in_boundary` / `boundaries_to_run`（`ideal` と `noisy` を並べて計算）、出力先 `out_dir` などを"
              "ここで変更します。"
              "各キーの意味は README_v3.md §6 を参照。",
    "Cell 2": "**[Cell 2] GF(2) 線形代数**（v1 と同じ。変更不要）",
    "Cell 3": "**[Cell 3] 幾何**（v1 と同じ）: 論文 Fig. 4 / SM Fig. 6 の正方格子三角形パッチ（境界 L_r, L_b, L_rb）、"
              "Z チェック、dressed X 生成元（cup 積どおりの CZ ペア）、論理 Z、U = WV の CS / T† / CNOT の配置。",
    "Cell 4": "**[Cell 4] φ0 と U の論理作用、厳密性の条件 C1–C3**: U が論理 T† になることの検査（v1）と、"
              "Pauli-proxy への置き換えが厳密になるための条件の検査（v2 で追加）。",
    "Cell 5": "**[Cell 5] Pauli-proxy エンジン（v2 の本体）**: Takada–Bartlett–Williamson の Clifford-stabilizer simulation。"
              "誤り演算子 E（Pauli X フレーム × 対角クリフォード）を古典的に追跡し、測定演算子を E で共役して Pauli dressing を付け、"
              "クリフォード因子を Pauli proxy に置き換えた Pauli 測定を Stim の TableauSimulator で実行します。",
    "Cell 6": "**[Cell 6] 復号器**（v1 と同じ）: 時空 MWPM（PyMatching）と JIT フラックス復号器。",
    "Cell 7": "**[Cell 7] プロトコル（1 ショット）**: switch-in（JIT）→ 電荷の後始末 →［理想 U］→ switch-out（Z^g 単発測定と緑の補正）"
              "→ surface code の d ラウンド＋完全ラウンド（時空 MWPM）→ 論理状態の判定。",
    "Cell 8": "**[Cell 8] 掃引・統計・作図・自己診断**（v3: 条件を照合する resume、d を分けた実行で既存の行を残す、"
              "全 d での自己診断、ideal/noisy の比較図）",
}

VAL_DESC = {
    "Cell V1": "**[Cell V1] 参照 1: 総当たり（brute-force）状態ベクトル**（v1 の正当化コードそのまま）: 配置空間上の密な状態ベクトルに、"
               "dressed / bare X 測定の射影、CNOT・CS・T†、Z 測定を文字どおり作用させる。",
    "Cell V2": "**[Cell V2] 参照 2: v1 の Z4 twisted-frame エンジン**（v1 の Cell 5–6 をそのまま複製。numba を使用）",
    "Cell V3": "**[Cell V3] ロックステップ比較**: 同じ雑音・同じ復号器・同じ乱数ビットで v2 エンジンと参照を同時に動かし、"
               "全測定（決定的/ランダムの別と結果）、Z チェック、X フレーム、最終の論理状態クラスを 1 ショットずつ比較。"
               "あわせて U = WV が D4 符号の論理 T† であることを総当たりで直接確認。",
    "Cell V4": "**[Cell V4] 実行ドライバ**: ロックステップ（参照 1・2）、本番経路の再生（参照 3）、合否判定 `assert_validation`",
}

STRIP_MAIN = 'if __name__ == "__main__":\n    results = run_experiments()\n'


def main_cells(nb, desc):
    head, cells = split_cells(SRC_MAIN)
    for title, body in cells:
        key = title.split("]")[0].strip("[")
        nb.append(md(desc.get(key, f"**{title}**")))
        if key == "Cell 8":
            body = body.replace(STRIP_MAIN, "")
        nb.append(code(body))


def build_main():
    nb = [md("""
# CHSC code switching v3: folded surface code ↔ D4 code（phenomenological noise, Pauli-proxy / Stim）

論文 *Clifford Hierarchy Stabilizer Codes*（Kobayashi–Zhu–Hsin）の code switching
（folded surface code → D4 code → 理想な論理 $\\overline{T}^\\dagger$ → surface code）の論理失敗率 $P_L(p,d)$ を評価します。

- シミュレーション方式は v2 と同じ **Clifford-stabilizer simulation**（Takada–Bartlett–Williamson、Pauli proxy を Stim で実行）です。
  幾何・プロトコル・復号器は v1 と同じです。
- **v3 の変更点**（エンジンは v2 から変更なし）: 計算条件を照合する resume、d を分けて実行しても既存の行を消さない、
  `ideal` と `noisy` を並べて計算できる `boundaries_to_run`、全 d での自己診断、図・交点は $P_L$ に統一。
- 出力は Google Drive に保存せず、ランタイム内のフォルダ（既定 `chsc_v3_results/`）に保存し、最後のセルで zip にまとめてダウンロードします。
- 3 ケース: `in_only`（switch-in のみ）/ `in_out`（T† なし）/ `in_T_out`（本番）
- $P_L = 1 - P_I$: 最終状態が理想状態と一致しない確率（論理 Clifford 安定化子の架空測定で判定）
- **Colab の CPU ランタイムで動きます（GPU は不要です）**。上から順に実行してください。最初は `CFG["shots"] = 2000` 程度で動作確認を。
- 詳細（方式・厳密性・復号器・ideal/noisy の意味・未記載事項の扱い）: `README_v3.md`
"""), md("**[Cell 0] インストール**（Stim と PyMatching。numba は不要）"), code("!pip -q install stim pymatching")]
    main_cells(nb, MAIN_DESC)
    nb.append(md("""**[Cell 9] 実行**: `CFG` の設定で掃引します（1 点ごとに CSV を上書き保存。`resume=True` なら、同じランタイム内で
中断・再実行したときに CSV にある点を飛ばします）。3 ケースをまとめて回す場合は `CFG["cases_to_run"] = ["in_only", "in_out", "in_T_out"]`。
出力（`CFG["out_dir"]` 内）: `chsc_v3_<case>_<ideal|noisy>.csv / .png / .pdf`、論理状態の内訳 `_frames.png`、交点 `_crossings.csv`、
実行条件 `_manifest.json`。`ideal` と `noisy` を両方計算すると比較図 `chsc_v3_ideal_vs_noisy.png` も作ります。
d を分けて複数回実行しても、同じ条件なら同じ CSV に追記されます。条件（ラウンド数・境界処理・seed など）を変えたのに同じ `out_dir` を
使うとエラーで止まるので、別の `out_dir` を指定してください。"""))
    nb.append(code("""# 例: 3 ケースを一度に（不要ならコメントアウト）
# CFG["cases_to_run"] = ["in_only", "in_out", "in_T_out"]
# CFG["boundaries_to_run"] = ["ideal", "noisy"]   # 途中の理想ラウンドあり/なしを並べて計算（README_v3 §2）
# CFG["shots"] = 2000   # 動作確認用
results = run_experiments()
results.head()"""))
    nb.append(md("""**[Cell 10] 結果のダウンロード**（Google Drive はマウントしません）: `out_dir` を zip にまとめ、ブラウザにダウンロードします。
Colab 以外では zip を作るだけです。ランタイムが切断されるとファイルは消えるので、計算が終わったら早めに実行してください。"""))
    nb.append(code("""import shutil, os
zip_path = shutil.make_archive(CFG["out_dir"], "zip", CFG["out_dir"])
print("created", zip_path, sorted(os.listdir(CFG["out_dir"])))
try:
    from google.colab import files
    files.download(zip_path)
except ImportError:
    print("not running on Colab: download skipped")"""))
    return notebook(nb)


def build_val():
    nb = [md("""
# CHSC code switching v3 — 正当化（validation）ノートブック

v3 のエンジン（v2 と同一。Pauli proxy + Stim）は、本物の量子状態の代わりに「Pauli 安定化子状態（proxy）＋誤り演算子 E」を追跡します。
これが同じプロトコルの**厳密な**シミュレーションであることを、2 つの独立な厳密計算と 1 ショットずつ突き合わせて確かめます。

- 参照 1: **総当たりの状態ベクトル計算**（演算子を文字どおり作用させる。d=2）
- 参照 2: **v1 の twisted-frame エンジン**（v1 で総当たりと照合済みの厳密エンジン。d=3, 4）
- 参照 3（v3 で追加）: **本番の経路**（MPP の一括実行と符号ビット、Stim 自身の乱数）の結果を総当たり計算に再生し、
  本番エンジンが出したすべての結果が起こりうるものか（総当たり側で決定的な結果と一致するか）を確認（d=2）
- 判定基準: **mismatches = 0**（全測定の決定的/ランダムの別と結果、Z チェック、X フレーム、最終の論理状態クラスがすべて一致）。
  最後のセルの `assert_validation` が合否を判定し、不合格なら `AssertionError` で止まります（**`VALIDATION PASSED` と表示されれば合格**）
- Colab の CPU ランタイムで十数分（GPU 不要）。前半のセルはメインノートブックと同じコード、後半（Cell V1–V5）が正当化です。
"""), md("**[Cell 0] インストール**（参照 2 は numba を使います。Colab には通常入っています）"),
        code("!pip -q install stim pymatching numba")]
    main_cells(nb, MAIN_DESC)
    vhead, vcells = split_cells(SRC_VAL)
    for title, body in vcells:
        key = title.split("]")[0].strip("[")
        nb.append(md(VAL_DESC.get(key, f"**{title}**")))
        if key == "Cell V1":
            body = body.replace("""try:
    from chsc_switch_v3 import *  # noqa: F401,F403
except ImportError:  # inside the notebook everything is already defined
    pass
""", "")
        if key == "Cell V4":
            body = body[:body.index('if __name__ == "__main__":')].rstrip() + "\n"
        nb.append(code(body))
    nb.append(md("**[Cell V5] 実行**: (1) 自己診断（d=3,5,7 のそれぞれで幾何、U の論理作用、厳密性の条件 C1–C3、p=0、符号空間）、"
                 "(2) U の総当たり検査、(3) ロックステップ（総当たり: d=2、v1 エンジン: d=3, 4。各 3 ケース × 2 境界処理 × 4 p）、"
                 "(4) 本番経路の再生（d=2）、(5) 合否判定。"))
    nb.append(code("""import pandas as pd
for k_, v_ in run_self_checks(CFG).items():
    print(k_, ":", v_)
print()
u_check = check_U_bruteforce(2)
print("[U by brute force]", u_check)
print()
rows = run_lockstep("bf") + run_lockstep("tf") + run_replay()
val_df = pd.DataFrame(rows)
val_df.to_csv("chsc_v3_validation.csv", index=False)
for ref in ("bf", "tf", "bf_replay"):
    s = val_df[val_df.reference == ref]
    print(f"[{ref}] total mismatches: {int(s.mismatches.sum())}  / shots: {int(s.shots.sum())}"
          f"  / compared outputs: {int(s.compared_outputs.sum())}")
print(assert_validation(val_df, u_check))"""))
    return notebook(nb)


if __name__ == "__main__":
    out_main = sys.argv[1] if len(sys.argv) > 1 else "CHSC_switch_v3_main.ipynb"
    out_val = sys.argv[2] if len(sys.argv) > 2 else "CHSC_switch_v3_validation.ipynb"
    json.dump(build_main(), open(out_main, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(build_val(), open(out_val, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out_main, out_val)
