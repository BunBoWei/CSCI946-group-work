# CSCI446/946 Big Data Analytics - Assignment 2
# 03 - Association rules

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)

FLAGS = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
         "has_coord", "location_missing", "timezone_missing", "link_default", "sidebar_default"]
BINNED = ["tweet_count", "fav_number", "tweets_per_day", "favs_per_day",
          "account_age_days", "text_len", "desc_len", "text_n_urls"]


# 1. load the full data - rules describe the data, so no train/test split
df = pd.read_csv(PROC / "twitter_full.csv")
print("shape:", df.shape)


# 2. build the transaction matrix.
items = pd.DataFrame(index=df.index)

# the flags, plus negations where the absence is informative
for col in FLAGS:
    items[col] = df[col] == 1
items["has_description"] = df["desc_missing"] == 0
items["has_location"] = df["location_missing"] == 0
items["custom_link_colour"] = df["link_default"] == 0
items["custom_sidebar_colour"] = df["sidebar_default"] == 0
items["uploaded_image"] = df["default_image"] == 0

# numeric features must be discretised first
for col in BINNED:
    low, high = df[col].quantile([.25, .75])
    items[col + "_low"] = df[col] <= low
    items[col + "_high"] = df[col] >= high

# the label as an item, so rules can conclude with it
label = df["is_human"].map({1: "human", 0: "non_human"})
items["human"] = label == "human"
items["non_human"] = label == "non_human"

print("items:", items.shape[1])
print("support of each item:\n", items.mean().sort_values(ascending=False).round(3))


# 3. frequent itemsets (max_len=4: longer ones are just padded copies of shorter rules)
frequent = apriori(items, min_support=0.05, use_colnames=True, max_len=4)
frequent["length"] = frequent["itemsets"].apply(len)
print("frequent itemsets:", len(frequent))
print(frequent["length"].value_counts().sort_index())
print(frequent.sort_values("support", ascending=False).head(15))


# 4. rules: at least 60% confident, and better than the base rate (lift > 1)
rules = association_rules(frequent, metric="confidence", min_threshold=0.6)
rules = rules[rules["lift"] > 1]
rules["antecedent_len"] = rules["antecedents"].apply(len)
print("rules:", len(rules))
print(rules.sort_values("lift", ascending=False).head(10)[
    ["antecedents", "consequents", "support", "confidence", "lift"]])


# 5. support vs confidence, coloured by lift
plt.figure(figsize=(8, 5))
points = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"],
                     cmap="viridis", alpha=0.6)
plt.colorbar(points, label="lift")
plt.xlabel("support")
plt.ylabel("confidence")
plt.title("all rules")
plt.show()


# 6. rules that conclude human or non-human
TARGETS = [frozenset(["human"]), frozenset(["non_human"])]
label_rules = rules[rules["consequents"].isin(TARGETS)].copy()
label_rules = label_rules.sort_values(["confidence", "lift"], ascending=False)
print("rules concluding a label:", len(label_rules))
print(label_rules.head(15)[["antecedents", "consequents", "support", "confidence", "lift"]])

# compare the two classes
for target in TARGETS:
    best = label_rules[label_rules["consequents"] == target].head(5)
    print("\nstrongest rules for", set(target))
    for _, r in best.iterrows():
        print("  %-60s conf %.3f  lift %.2f  support %.3f"
              % (" & ".join(sorted(r["antecedents"])), r["confidence"], r["lift"], r["support"]))


# 7. profiles matching a strong rule for the other class are candidate mislabels
STRONG = label_rules[(label_rules["confidence"] >= 0.8) & (label_rules["lift"] >= 1.3)]
print()
print("rules used for flagging:", len(STRONG))

# score by the strongest contradicting rule, not the count - the rules overlap
flagged = pd.Series(0, index=df.index)
best_conf = pd.Series(0.0, index=df.index)
evidence = pd.Series("", index=df.index)
for _, r in STRONG.iterrows():
    said = "human" if r["consequents"] == frozenset(["human"]) else "non_human"
    hit = items[sorted(r["antecedents"])].all(axis=1) & label.notna() & (label != said)
    flagged[hit] += 1
    stronger = hit & (r["confidence"] > best_conf)
    evidence[stronger] = " & ".join(sorted(r["antecedents"])) + " -> " + said
    best_conf[stronger] = r["confidence"]

result = df.loc[flagged > 0, ["_unit_id", "name", "gender", "gender:confidence",
                              "label_conflict", "tweet_count", "fav_number"]].copy()
result["rules_contradicting"] = flagged[flagged > 0]
result["best_confidence"] = best_conf[flagged > 0].round(3)
result["strongest_rule"] = evidence[flagged > 0]
result = result.sort_values("best_confidence", ascending=False)

print("profiles contradicted by at least one rule:", len(result))
print(result.groupby("gender")[["rules_contradicting", "best_confidence"]].mean().round(2))
print(result["gender"].value_counts())
print(result.head(10)[["name", "gender", "gender:confidence", "best_confidence", "strongest_rule"]])
result.to_csv(OUT / "association_flagged.csv", index=False)

# low crowd confidence as well makes a stronger candidate
print("contradicted and below full crowd confidence:",
      (result["gender:confidence"] < 1).sum())
plt.figure(figsize=(8, 4))
plt.hist([result["gender:confidence"], df["gender:confidence"]], bins=20, density=True,
         label=["contradicted profiles", "all profiles"])
plt.xlabel("gender:confidence")
plt.ylabel("density")
plt.legend()
plt.show()