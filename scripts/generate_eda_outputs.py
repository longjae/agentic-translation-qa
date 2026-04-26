from pathlib import Path
from collections import Counter
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

root = Path(__file__).resolve().parents[1]
data_path = root / "data" / "datasets" / "eval_set.json"
out_dir = root / "outputs"
fig_dir = out_dir / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_json(data_path, encoding="utf-8").rename(
    columns={"source_text": "source_ko", "reference": "reference_en"}
)

for col in ["source_ko", "reference_en"]:
    df[col] = df[col].astype(str).str.strip()
df = df.replace({"": np.nan}).dropna(subset=["source_ko", "reference_en"])
df = df.drop_duplicates(subset=["source_ko", "reference_en"]).reset_index(drop=True)

def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

df["char_len_ko"] = df["source_ko"].str.len()
df["char_len_en"] = df["reference_en"].str.len()
df["token_len_ko"] = df["source_ko"].apply(lambda x: len(tokenize(x)))
df["token_len_en"] = df["reference_en"].apply(lambda x: len(tokenize(x)))
df["length_ratio"] = df["char_len_en"] / df["char_len_ko"].replace(0, np.nan)

domain_terms = ["계약", "조항", "책임", "의무", "손해배상", "위반", "합의"]
df["matched_terms"] = df["source_ko"].apply(lambda x: [t for t in domain_terms if t in x])
df["matched_term_count"] = df["matched_terms"].apply(len)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["char_len_ko"], bins=20, kde=True, ax=axes[0], color="royalblue")
axes[0].set_title("Korean Sentence Length Distribution")
sns.histplot(df["char_len_en"], bins=20, kde=True, ax=axes[1], color="darkorange")
axes[1].set_title("English Sentence Length Distribution")
plt.tight_layout()
plt.savefig(fig_dir / "01_length_distribution.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="char_len_ko", y="char_len_en", alpha=0.5, s=30)
plt.title("KO-EN Sentence Length Correlation")
plt.tight_layout()
plt.savefig(fig_dir / "02_length_correlation.png", dpi=200, bbox_inches="tight")
plt.close()

counter = Counter()
for terms in df["matched_terms"]:
    counter.update(terms)
term_df = pd.DataFrame(counter.items(), columns=["term", "count"]).sort_values("count", ascending=False)
term_label_map = {
    "계약": "contract",
    "조항": "clause",
    "책임": "liability",
    "의무": "obligation",
    "손해배상": "damages",
    "위반": "breach",
    "합의": "agreement",
}
term_df["term_display"] = term_df["term"].map(term_label_map).fillna(term_df["term"])
plt.figure(figsize=(8, 5))
sns.barplot(data=term_df, x="count", y="term_display", color="steelblue")
plt.title("Domain Term Frequency")
plt.tight_layout()
plt.savefig(fig_dir / "03_domain_term_frequency.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="matched_term_count", color="seagreen")
plt.title("Matched Domain Terms per Sample")
plt.tight_layout()
plt.savefig(fig_dir / "04_terms_per_sample.png", dpi=200, bbox_inches="tight")
plt.close()

candidate_df = df[
    (df["matched_term_count"] >= 1)
    & (df["char_len_ko"].between(15, 120))
    & (df["char_len_en"].between(10, 180))
].copy()
if len(candidate_df) == 0:
    selected_df = df.sample(min(40, len(df)), random_state=42).copy()
else:
    bins = min(5, candidate_df["char_len_ko"].nunique())
    if bins >= 2:
        candidate_df["length_bin"] = pd.qcut(candidate_df["char_len_ko"], q=bins, duplicates="drop")
        selected_df = (
            candidate_df.groupby("length_bin", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), 8), random_state=42))
            .reset_index(drop=True)
        )
    else:
        selected_df = candidate_df.sample(min(40, len(candidate_df)), random_state=42)
selected_df = selected_df.head(40).copy()

compare_df = pd.concat(
    [candidate_df.assign(group="candidate"), selected_df.assign(group="selected")], ignore_index=True
)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(
    data=compare_df, x="char_len_ko", hue="group", stat="density", common_norm=False, bins=20, ax=axes[0]
)
axes[0].set_title("Candidate vs Selected: KO Length")
sns.histplot(
    data=compare_df,
    x="matched_term_count",
    hue="group",
    stat="density",
    common_norm=False,
    discrete=True,
    ax=axes[1],
)
axes[1].set_title("Candidate vs Selected: Domain Term Density")
plt.tight_layout()
plt.savefig(fig_dir / "05_candidate_vs_selected.png", dpi=200, bbox_inches="tight")
plt.close()

summary_stats = df[["char_len_ko", "char_len_en", "token_len_ko", "token_len_en", "length_ratio"]].describe().T
selected_df.to_csv(out_dir / "selected_eval_set.csv", index=False, encoding="utf-8-sig")
summary_stats.to_csv(out_dir / "summary_stats.csv", encoding="utf-8-sig")
term_df.to_csv(out_dir / "domain_term_frequency.csv", index=False, encoding="utf-8-sig")

print("EDA outputs generated:", fig_dir)
