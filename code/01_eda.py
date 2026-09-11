# CSCI446/946 Big Data Analytics - Assignment 2
# Exploratory data analysis of the raw Twitter user data

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "twitter_user_data.csv"
NUMERIC = ["fav_number", "retweet_count", "tweet_count"]
LABELS = ["male", "female", "brand", "unknown"]


# 1. load the data (the file is not UTF-8; Mac Roman reads it without errors)
df = pd.read_csv(DATA, encoding="mac_roman")
print("shape:", df.shape)
print(df.dtypes)
print(df.head())


# 2. missing and unique values per column
summary = pd.DataFrame({"missing": df.isna().sum(),
                        "missing_%": (df.isna().mean() * 100).round(2),
                        "unique": df.nunique()})
print(summary)

summary.loc[summary["missing"] > 0, "missing_%"].sort_values().plot.barh()
plt.xlabel("missing (%)")
plt.show()


# 3. label: gender and its confidence
print(df["gender"].value_counts(dropna=False))
print(pd.crosstab(df["profile_yn"], df["gender"].fillna("missing")))
print(df.groupby("gender")["gender:confidence"].describe())

fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
for ax, g in zip(axes, LABELS):
    ax.hist(df.loc[df["gender"] == g, "gender:confidence"], bins=30)
    ax.set_title(g)
    ax.set_xlabel("gender:confidence")
axes[0].set_ylabel("records")
plt.show()

# gold-standard records: crowd label against gold label
gold = df[df["_golden"]]
print(pd.crosstab(gold["gender"], gold["gender_gold"]))


# 4. numeric columns: summary, distribution before and after log1p, correlation
print(df[NUMERIC].describe().round(2))
print("skew:\n", df[NUMERIC].skew().round(2))
print("zeros:\n", (df[NUMERIC] == 0).sum())

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for j, col in enumerate(NUMERIC):
    axes[0, j].hist(df[col], bins=50)
    axes[0, j].set_yscale("log")
    axes[0, j].set_title(col)
    axes[1, j].hist(np.log1p(df[col]), bins=50)
    axes[1, j].set_title("log1p(" + col + ")")
plt.tight_layout()
plt.show()

print(np.log1p(df[NUMERIC]).corr().round(3))

# numeric columns by label
print(df.groupby("gender")[NUMERIC].median())
df.boxplot(column=["fav_number", "tweet_count"], by="gender", figsize=(10, 4))
plt.yscale("log")
plt.show()


# 5. date columns
for col in ["created", "tweet_created", "_last_judgment_at"]:
    t = pd.to_datetime(df[col], format="%m/%d/%y %H:%M")
    print(col, "min", t.min(), "max", t.max(), "unique", t.nunique())
created = pd.to_datetime(df["created"], format="%m/%d/%y %H:%M")
pd.crosstab(created.dt.year, df["gender"], normalize="columns").plot(marker="o")
plt.ylabel("share of label")
plt.show()

# tweet_id
print("tweet_id unique:", df["tweet_id"].nunique(), df["tweet_id"].unique())


# 6. categorical columns: time zone and location
print(df["user_timezone"].value_counts().head(15))
print(df["tweet_location"].value_counts().head(15))
print("tweet_coord present:", df["tweet_coord"].notna().sum())


# 7. colour columns: stored length of the hex codes, most common codes by label
for col in ["link_color", "sidebar_color"]:
    codes = df[col].astype(str)
    print(col, "length:\n", codes.str.len().value_counts().sort_index())
    print(codes[codes.str.len() != 6].value_counts().head(10))
    print(pd.crosstab(codes, df["gender"]).sort_values("male", ascending=False).head(10))


# 8. profile image: default (no uploaded picture) by label
default_image = df["profileimage"].str.contains("default_profile_images")
print(pd.crosstab(df["gender"], default_image, normalize="index").round(3))


# 9. text columns: encoding damage, length and content by label
for col in ["text", "description"]:
    s = df[col].fillna("")
    print(col, "records with non-ASCII characters:", s.map(lambda x: any(ord(c) > 127 for c in x)).sum())
    print(col, "length by label:\n", s.str.len().groupby(df["gender"]).median())
print(df.loc[df["text"].str.contains("â€"), "text"].head())

text_props = pd.DataFrame({"description missing": df["description"].isna(),
                           "url in tweet": df["text"].str.contains("http"),
                           "mention in tweet": df["text"].str.contains("@"),
                           "hashtag in tweet": df["text"].str.contains("#")})
print(text_props.groupby(df["gender"]).mean().round(3))

# most common words in descriptions by label, without stop words
words = df["description"].fillna("").str.lower().str.findall(r"[a-z]{3,}")
for g in ["male", "female", "brand"]:
    w = pd.Series(words[df["gender"] == g].sum())
    print(g, w[~w.isin(ENGLISH_STOP_WORDS | {"http", "https", "com", "www"})].value_counts().head(20).to_dict())

# most repeated tweets
print(df["text"].value_counts().head(10))


# 10. duplicated records and repeated accounts
print("duplicated rows (excluding _unit_id):", df.drop(columns="_unit_id").duplicated().sum())
copies = df["name"].value_counts()
print("accounts:", len(copies), "appearing more than once:", (copies > 1).sum())
print(copies[copies > 1].value_counts().sort_index())

labels_per_account = df[df["name"].isin(copies[copies > 1].index)].groupby("name")["gender"].agg(
    lambda s: "+".join(sorted(s.dropna().unique())))
print(labels_per_account.value_counts())
