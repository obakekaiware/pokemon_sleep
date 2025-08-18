# streamlit_app.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D

st.set_page_config(page_title="育成コスト計算", layout="centered")
st.title("🧮 育成コスト計算")

CSV_PATH = Path("data/exp_table.csv")
required_cols = [
    "レベル",
    "アメ必要量",
    "累計アメ量",
    "ゆめのかけら必要量",
    "累計ゆめのかけら量",
]


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("\u3000", "", regex=False),
        errors="coerce",
    )


# ---- CSV読み込み ----
if not CSV_PATH.exists():
    st.error(f"CSV が見つからない: {CSV_PATH}")
    st.stop()

# あなたのCSV構造に合わせて header=2（=3行目がヘッダ）
df_raw = pd.read_csv(CSV_PATH, header=2)
df_raw = df_raw.loc[:, ~df_raw.columns.astype(str).str.contains("^Unnamed")]

# 列名の名寄せ
if not all(col in df_raw.columns for col in required_cols):
    trim = {c: str(c).replace("\u3000", " ").strip() for c in df_raw.columns}
    rev = {}
    for k, v in trim.items():
        rev.setdefault(v, k)
    cols_map = {}
    for need in required_cols:
        if need in rev:
            cols_map[rev[need]] = need
    df_raw = df_raw.rename(columns=cols_map)

if not all(col in df_raw.columns for col in required_cols):
    st.error("必要な列が見つからないよ。")
    st.stop()

df = df_raw[required_cols].copy()
for c in required_cols:
    df[c] = coerce_numeric(df[c])
df = (
    df.dropna(subset=["レベル"])
    .assign(レベル=lambda x: x["レベル"].astype(int))
    .drop_duplicates(subset=["レベル"])
    .sort_values("レベル")
    .reset_index(drop=True)
)

# ---- 入力UI（ミニマル） ----
min_lv, max_lv = int(df["レベル"].min()), int(df["レベル"].max())


def clamp(v, lo, hi):
    return int(max(lo, min(v, hi)))


default_s = clamp(30, min_lv, max_lv - 1)
default_t = clamp(60, min_lv + 1, max_lv)
if default_s >= default_t:
    default_s, default_t = min_lv, min(min_lv + 1, max_lv)

c1, c2 = st.columns(2)
with c1:
    s = st.number_input(
        "開始レベル", min_value=min_lv, max_value=max_lv, value=default_s, step=1
    )
with c2:
    t = st.number_input(
        "目標レベル", min_value=min_lv, max_value=max_lv, value=default_t, step=1
    )

kind = st.radio(
    "ポケモン種別", ["通常", "高性能", "伝説", "幻"], horizontal=True, index=0
)
mult_map = {"通常": 1.0, "高性能": 1.5, "伝説": 1.8, "幻": 2.2}
xp_mult = mult_map[kind]

use_boost = st.checkbox("ミニアメブースト", value=False)  # 全区間ON/OFFの単一結果用

# 換算レート（かけら/アメ）
default_rate = 17772.0 / 28.0  # ≈ 634.7142857
rate = st.sidebar.number_input(
    "換算レート（かけら/アメ）", min_value=0.0001, value=float(default_rate), step=1.0
)

if s >= t:
    st.warning("開始レベルは目標レベルより小さくしてね。")
    st.stop()

lvl_to_row = df.set_index("レベル")
missing = [lv for lv in range(s, t + 1) if lv not in lvl_to_row.index]
if missing:
    st.error("データに存在しないレベルがあるよ。")
    st.stop()

# ---- 区間抽出 ----
slice_rows = df[(df["レベル"] >= s) & (df["レベル"] < t)].copy()
N_base = slice_rows["アメ必要量"].fillna(0).astype(float).to_numpy()
S_base = slice_rows["ゆめのかけら必要量"].fillna(0).astype(float).to_numpy()
per_candy_shard = np.where(N_base > 0, S_base / N_base, 0.0)


# ---- 単一結果（全区間ON/OFF） ----
def compute_cost_all_or_none(N_base, S_base, per_candy_shard, xp_mult, use_boost):
    N_multi = np.ceil(N_base * xp_mult).astype(int)
    if use_boost:
        N_used = np.ceil(N_multi / 2.0).astype(int)  # アメ半分（切上）
        shard_factor = 4.0
    else:
        N_used = N_multi
        shard_factor = 1.0
    shards_each = np.ceil(per_candy_shard * shard_factor * N_used).astype(int)
    return int(N_used.sum()), int(shards_each.sum())


total_candies, total_shards = compute_cost_all_or_none(
    N_base, S_base, per_candy_shard, xp_mult, use_boost
)

# ---- 結果（ミニマル表示）----
m1, m2 = st.columns(2)
with m1:
    st.metric("アメ", f"{total_candies:,} 個")
with m2:
    st.metric("ゆめのかけら", f"{total_shards:,} 個")

# ---- x個だけブースト→残り通常の推移 ----
# 通常運用時に必要なアメ（種別倍率反映）
N_multi_normal = np.ceil(N_base * xp_mult).astype(int)

# ブーストできる最大個数：各レベル ceil(n/2) の合計
x_cap = int(np.ceil(N_multi_normal / 2.0).sum())

# 0..x_cap で計算
x_vals = np.arange(0, x_cap + 1, dtype=int)
candies_series = np.zeros_like(x_vals, dtype=int)
shards_series = np.zeros_like(x_vals, dtype=int)


def totals_given_x_boost_first(x):
    """先に x 個のアメをブースト（1個で2個分 & かけら4倍）に使い、残りは通常で消費。"""
    x_rem = int(x)
    candies_total = 0
    shards_total = 0
    for n0, per_shard in zip(N_multi_normal, per_candy_shard):
        if n0 <= 0:
            continue
        b_eff_max = int(np.ceil(n0 / 2.0))  # そのレベルで使えるブースト上限
        b = min(x_rem, b_eff_max)  # 実際に使うブースト数
        normal_cnt = max(n0 - 2 * b, 0)  # 残りは通常アメ
        candies_total += b + normal_cnt
        shards_total += int(
            np.ceil(per_shard * (normal_cnt + 4 * b))
        )  # レベル単位で切上
        x_rem -= b
        if x_rem <= 0:
            pass
    return candies_total, shards_total


for i, x in enumerate(x_vals):
    c_tot, s_tot = totals_given_x_boost_first(int(x))
    candies_series[i] = c_tot
    shards_series[i] = s_tot

# ---- 合成コスト ----
composite_cost = candies_series * float(rate) + shards_series
best_idx = int(np.argmin(composite_cost))
best_x = int(x_vals[best_idx])
best_c = int(candies_series[best_idx])
best_s = int(shards_series[best_idx])

bm1, bm2, bm3 = st.columns(3)
with bm1:
    st.metric("最適 x", f"{best_x}")
with bm2:
    st.metric("アメ（最適）", f"{best_c:,} 個")
with bm3:
    st.metric("かけら（最適）", f"{best_s:,} 個")

# ---- 交点 x（アメ×rate と かけら が等しい所）----
diff = shards_series - candies_series * float(rate)
cross_idx = int(np.argmin(np.abs(diff)))  # 離散xなので絶対値最小を採用
cross_x = int(x_vals[cross_idx])

# ---- 図1：Candy/Shard（右軸＝左軸×rate）、最適x=赤, 交点x=緑 ----
fig, ax_left = plt.subplots()
ax_right = ax_left.twinx()

color_candy = "tab:blue"
color_shard = "tab:orange"
color_best = "tab:red"
color_cross = "tab:green"

(line_candy,) = ax_left.plot(
    x_vals, candies_series, label="アメ（左軸）", color=color_candy
)
(line_shard,) = ax_right.plot(
    x_vals, shards_series, label="ゆめのかけら（右軸）", color=color_shard
)

# 右軸 = 左軸 × rate に合わせる
y_left_min = min(candies_series.min(), (shards_series / rate).min())
y_left_max = max(candies_series.max(), (shards_series / rate).max())
pad = max(1.0, 0.05 * (y_left_max - y_left_min))
ax_left.set_ylim(y_left_min - pad, y_left_max + pad)
ax_right.set_ylim((y_left_min - pad) * rate, (y_left_max + pad) * rate)

# 縦線
ax_left.axvline(best_x, color=color_best, linestyle="--", linewidth=2)
ax_left.axvline(cross_x, color=color_cross, linestyle=":", linewidth=2)

ax_left.set_xlabel("先にブーストするアメの数 x")
ax_left.set_ylabel("必要アメ総数")
ax_right.set_ylabel("必要ゆめのかけら総数")

legend_lines = [
    line_candy,
    line_shard,
    Line2D(
        [0],
        [0],
        color=color_best,
        linestyle="--",
        linewidth=2,
        label="最適 x（合成最小）",
    ),
    Line2D(
        [0],
        [0],
        color=color_cross,
        linestyle=":",
        linewidth=2,
        label="交点 x（アメ×rate＝かけら）",
    ),
]
ax_left.legend(handles=legend_lines, loc="best")
plt.tight_layout()
st.pyplot(fig)

# ---- 図2：合成コスト C(x) の折れ線（最適xを赤で強調）----
fig2, ax = plt.subplots()
ax.plot(x_vals, composite_cost, label="C(x) = アメ×レート + かけら")
ax.axvline(best_x, color=color_best, linestyle="--", linewidth=2)
ax.set_xlabel("先にブーストするアメの数 x")
ax.set_ylabel("合成コスト（かけら換算）")
ax.legend(loc="best")
plt.tight_layout()
st.pyplot(fig2)
