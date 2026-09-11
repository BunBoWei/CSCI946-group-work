# CSCI446/946 Big Data Analytics - Assignment 2
# Data preprocessing: clean the raw data and prepare each data type for the models

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "twitter_user_data.csv"
OUT = ROOT / "data" / "processed" / "twitter_clean.csv"
DATE_FORMAT = "%m/%d/%y %H:%M"


def repair_text(s):
    # undo the wrong decoding (UTF-8 read as Windows-1252, saved as Mac Roman)
    if not isinstance(s, str):
        return ""
    try:
        s = s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        s = re.sub("_Ÿ[^\x00-\x7f]{0,3}", " ", s)   # emoji remnants
        s = re.sub(r"[^\x00-\x7f]+", " ", s)                # other unrepairable characters
        s = re.sub(r"(?<!\w)_(?!\w)", " ", s)               # leftover placeholders
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def clean_words(s):
    # lower-case words only: no links, mentions, numbers or punctuation
    s = re.sub(r"https?://\S+|@\w+", " ", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]+", " ", s)).strip()


def fix_colour(code):
    # restore a six-character hex code; codes saved as numbers (e.g. 2.21E+09) cannot be restored
    code = str(code)
    if "E+" in code:
        return np.nan
    date = re.fullmatch(r"(\d{1,2})-([A-Za-z]{3})-(\d{2})", code)   # e.g. 2FEB45 saved as 2-Feb-45
    if date:
        code = "".join(date.groups())
    return code.zfill(6).upper()


# 1. load the raw data
df = pd.read_csv(DATA, encoding="mac_roman")
print("raw:", df.shape)


# 2. remove records whose profile was unavailable (no label)
df = df[df["profile_yn"] == "yes"].copy()
print("profile available:", df.shape)


# 3. one record per account; flag accounts labelled both human and brand
group = df["gender"].map({"male": "human", "female": "human", "brand": "brand"})
df["label_conflict"] = group.groupby(df["name"]).transform("nunique").gt(1).astype(int)
df = df.sort_values(["name", "_golden", "gender:confidence", "_unit_id"],
                    ascending=[True, True, False, True])
df = df.drop_duplicates("name").sort_values("_unit_id")
print("one record per account:", df.shape, "| label conflicts:", df["label_conflict"].sum())


# 4. label: 1 human, 0 non-human, missing for unknown
df["is_human"] = df["gender"].map({"male": 1, "female": 1, "brand": 0})
print(df["gender"].value_counts())


# 5. text: repair, cleaned words, and counts
df["text_has_emoji"] = df["text"].str.contains("_Ÿ").astype(int)
df["desc_missing"] = df["description"].isna().astype(int)
for col in ["text", "description"]:
    df[col] = df[col].map(repair_text)
df["text_clean"] = df["text"].map(clean_words)
df["desc_clean"] = df["description"].map(clean_words)
df["text_len"] = df["text"].str.len()
df["desc_len"] = df["description"].str.len()
df["text_n_urls"] = df["text"].str.count(r"https?://")
df["text_n_mentions"] = df["text"].str.count(r"@\w+")
df["text_n_hashtags"] = df["text"].str.count(r"#\w+")
df["desc_has_url"] = df["description"].str.contains(r"https?://|www\.").astype(int)


# 6. colour: repair the hex codes, split into red, green and blue (0-1)
for col in ["link_color", "sidebar_color"]:
    prefix = col.split("_")[0]
    df[col] = df[col].map(fix_colour)
    print(col, "unrecoverable:", df[col].isna().sum())
    for i, channel in enumerate("rgb"):
        value = df[col].str[2 * i:2 * i + 2].map(lambda h: int(h, 16) / 255, na_action="ignore")
        df[prefix + "_" + channel] = value.fillna(value.median())
df["link_default"] = (df["link_color"] == "0084B4").astype(int)
df["sidebar_default"] = (df["sidebar_color"] == "C0DEED").astype(int)


# 7. dates: account age at the time the sample was taken, and activity rates
sampled_at = pd.to_datetime(df["tweet_created"], format=DATE_FORMAT).max()
df["account_age_days"] = (sampled_at - pd.to_datetime(df["created"], format=DATE_FORMAT)).dt.days
df["tweets_per_day"] = df["tweet_count"] / df["account_age_days"].clip(lower=1)
df["favs_per_day"] = df["fav_number"] / df["account_age_days"].clip(lower=1)


# 8. other profile fields as flags
df["default_image"] = df["profileimage"].str.contains("default_profile_images").astype(int)
df["has_retweets"] = (df["retweet_count"] > 0).astype(int)
df["has_coord"] = df["tweet_coord"].notna().astype(int)
df["location_missing"] = df["tweet_location"].isna().astype(int)
df["timezone_missing"] = df["user_timezone"].isna().astype(int)
df["name_n_digits"] = df["name"].str.count(r"\d")


# 9. categorical: the ten most common time zones, the rest as Other, one-hot encoded
top_zones = df["user_timezone"].value_counts().index[:10]
zone = df["user_timezone"].where(df["user_timezone"].isin(top_zones), "Other")
zone[df["user_timezone"].isna()] = "Missing"
zones = pd.get_dummies(zone, prefix="tz", dtype=int).drop(columns="tz_Missing")
df = pd.concat([df, zones], axis=1)


# 10. numeric: log1p for the right-skewed counts, then standardise
LOG_COLS = ["fav_number", "tweet_count", "tweets_per_day", "favs_per_day"]
NUM_COLS = LOG_COLS + ["account_age_days", "text_len", "desc_len", "text_n_urls",
                       "text_n_mentions", "text_n_hashtags", "name_n_digits",
                       "link_r", "link_g", "link_b", "sidebar_r", "sidebar_g", "sidebar_b"]
scaled = df[NUM_COLS].copy()
scaled[LOG_COLS] = np.log1p(scaled[LOG_COLS])
df[[c + "_z" for c in NUM_COLS]] = StandardScaler().fit_transform(scaled)


# 11. keep the identifiers, label, text, raw values for interpretation and the model features
ID_COLS = ["_unit_id", "name", "gender", "gender:confidence", "label_conflict", "is_human"]
TEXT_COLS = ["description", "text", "desc_clean", "text_clean", "link_color", "sidebar_color"]
FLAG_COLS = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
             "has_coord", "location_missing", "timezone_missing", "link_default", "sidebar_default"]
df = df[ID_COLS + TEXT_COLS + NUM_COLS + FLAG_COLS + list(zones.columns) + [c + "_z" for c in NUM_COLS]]

print("clean:", df.shape)
print(df[[c + "_z" for c in NUM_COLS]].describe().round(2).T)
df.to_csv(OUT, index=False)
