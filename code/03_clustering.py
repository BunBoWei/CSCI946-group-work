# CSCI446/946 Big Data Analytics - Assignment 2
# 03 - Clustering: group the profiles by behaviour and use the groups to question the labels

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "output"
OUT.mkdir(exist_ok=True)
SEED = 7
SAMPLE = 5000          # hierarchical clustering and DBSCAN run on a sample, see step 8
K_RANGE = range(1, 16)
K_MAIN = 4             # chosen in step 3
K_TREE = 15            # where the hierarchical tree is cut in step 8
K_FINE = 12            # smaller clusters, used in step 10 to question the labels
PURE_HUMAN = 0.85      # a cluster this human-heavy is treated as one-sided
PURE_NON_HUMAN = 0.35  # and this is the other side, well below the 0.69 rate of the data
# one colour per label, used by every chart that shows the label
LABEL_COLOURS = {"human": "tab:blue", "non_human": "tab:orange"}

# the attributes, grouped by what they describe
ACTIVITY = ["fav_number_log", "tweet_count_log", "tweets_per_day_log", "favs_per_day_log",
            "account_age_days"]
CONTENT = ["text_len", "desc_len", "text_n_urls", "text_n_mentions", "text_n_hashtags",
           "name_n_digits"]
PROFILE = ["desc_missing", "desc_has_url", "text_has_emoji", "default_image", "has_retweets",
           "has_coord", "location_missing", "timezone_missing"]
COLOUR = ["link_r", "link_g", "link_b", "sidebar_r", "sidebar_g", "sidebar_b",
          "link_default", "sidebar_default"]
BEHAVIOUR = ACTIVITY + CONTENT + PROFILE


# 1. load the full cleaned data - clustering is unsupervised, so no train/test split.
# The numeric attributes were already rescaled to z-scores in 02_preprocess.py, which k-means
# needs: it uses Euclidean distance, so an attribute with large units would dominate
df = pd.read_csv(PROC / "twitter_full.csv")
labelled = df["is_human"].notna()
truth = df.loc[labelled, "is_human"]
print("Twitter dataset size:", df.shape)
print("labelled:", labelled.sum(), "| human rate:", round(truth.mean(), 3))


# 2. which attributes to include.
# First, highly correlated attributes: two similar attributes count the same thing twice
corr = df[BEHAVIOUR].corr().abs()
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
print("\nbehaviour attributes correlated above 0.7:\n", pairs[pairs > 0.7].round(2))
# Only two pairs pass 0.7 (0.76 each): tweet count with tweets per day, and favourites with
# favourites per day. Both are kept - the count is the total, the rate adjusts it for how old
# the account is, and account age is an attribute of its own

# Second, colour. Cluster on three attribute sets and compare how the human rate differs
# between clusters: if every cluster has the same rate, the clusters say nothing about the label
spread = {}
for name, cols in [("behaviour + colour", BEHAVIOUR + COLOUR),
                   ("behaviour only", BEHAVIOUR),
                   ("colour only", COLOUR)]:
    km = KMeans(n_clusters=K_MAIN, n_init=10, random_state=SEED)
    km.fit(df[cols])
    rate = truth.groupby(km.predict(df.loc[labelled, cols])).agg("mean")
    spread[name] = rate.round(2).tolist()
    print(name, "- human rate per cluster:", spread[name])

# Colour alone gives clusters whose human rates stay close to the 0.69 of the whole data
# (0.66 to 0.78), while behaviour alone spreads them from 0.19 to 0.89. Adding colour to
# behaviour narrows that spread (0.23 to 0.78), so colour is left out from here on
FEATURES = BEHAVIOUR
X = df[FEATURES]
print("\nclustering on", len(FEATURES), "behaviour attributes:", FEATURES)


# 3. how many clusters: the within sum of squares (WSS) for k = 1 to 15, and look for the elbow.
# n_init=10 runs k-means ten times from different starting centroids and keeps the lowest WSS
wss = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    km.fit(X)
    wss.append(km.inertia_)
plt.plot(list(K_RANGE), wss, marker="o")
plt.axvline(K_MAIN, color="red", linestyle="--")
plt.xlabel("Number of Clusters")
plt.ylabel("Within Sum of Squares")
plt.title("WSS for k = 1 to 15")
plt.show()

# The curve has no sharp elbow, so compare k-1, k and k+1 with the three diagnostic questions:
# are the clusters separated, are any too small, are any two centroids too close?
for k in [K_MAIN - 1, K_MAIN, K_MAIN + 1]:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    km.fit(X)
    groups = km.predict(X)
    print("\nk =", k, "| WSS:", round(km.inertia_))
    print("  cluster sizes:", np.bincount(groups).tolist())
    print("  closest two centroids:", round(pdist(km.cluster_centers_, "euclidean").min(), 2))
    print("  human rate per cluster:", truth.groupby(groups[labelled]).agg("mean").round(2).tolist())

# No k gives a tiny cluster. k=4 has the largest gap between its closest centroids (2.74 against
# 2.59 for k=3 and 2.48 for k=5), and it separates a mixed group from the rest that k=3 does
# not. k=5 only adds a cluster whose centroid sits closer to the others, and if more clusters
# do not distinguish the groups better, fewer is better. So k=4


# 4. apply k-means with the chosen k and describe each cluster by its mean attribute values
km = KMeans(n_clusters=K_MAIN, n_init=10, random_state=SEED)
km.fit(X)
df["cluster"] = km.predict(X)
print("\ncluster sizes:\n", df["cluster"].value_counts().sort_index())

# the attributes are z-scores, so a mean of +1 means "one standard deviation above average";
# for the 0/1 flags the mean is the share of the cluster with that flag
cluster_mean = df.groupby(["cluster"])[FEATURES].agg("mean")
print("\ncluster means:\n", cluster_mean.T.round(2))

fig, ax = plt.subplots(figsize=(7, 8))
image = ax.imshow(cluster_mean.T, cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(K_MAIN), ["cluster " + str(c) for c in cluster_mean.index])
ax.set_yticks(range(len(FEATURES)), FEATURES)
for i in range(len(FEATURES)):
    for j in range(K_MAIN):
        ax.text(j, i, round(cluster_mean.iloc[j, i], 2), ha="center", va="center", fontsize=7)
fig.colorbar(image, label="mean value")
plt.title("what each cluster looks like")
plt.tight_layout()
plt.show()


# 5. a cluster mean can hide a mix of very low and very high values, so describe each cluster
# (quartiles show where the middle of the cluster sits) and draw the spread as box plots
DESCRIBE = ["tweet_count", "tweets_per_day", "fav_number", "text_n_urls", "text_n_hashtags"]
for cluster in range(K_MAIN):
    print("\ndescribe cluster labeled " + str(cluster) + ": \n",
          df.loc[df["cluster"] == cluster, DESCRIBE].describe().round(2))

RAW_COUNTS = ["tweet_count", "tweets_per_day", "fav_number", "favs_per_day"]
BOX = RAW_COUNTS + ["account_age_days", "text_len", "desc_len"]
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, col in zip(axes.ravel(), BOX):
    ax.boxplot([df.loc[df["cluster"] == c, col] for c in range(K_MAIN)], showfliers=False)
    ax.set_xticks(range(1, K_MAIN + 1), range(K_MAIN))
    ax.set_xlabel("cluster")
    ax.set_title(col)
    # the first four are raw counts spread over several orders of magnitude; symlog is a log
    # scale that can also show zero. The last three were only kept as z-scores
    if col in RAW_COUNTS:
        ax.set_yscale("symlog")
        ax.set_ylim(bottom=0)   # counts cannot be negative
    else:
        ax.set_ylabel("z-score")
axes.ravel()[-1].axis("off")
plt.suptitle("spread of each attribute inside each cluster (outliers hidden)")
plt.tight_layout()
plt.show()

# What the spread confirms: cluster 0 tweets little throughout (its 75% quartile is below the
# median of the other clusters); cluster 2 almost never likes anything (even its 75% quartile
# is 0) and tweets at the highest rates; cluster 3 likes far more than the others.
# Cluster 1 mostly posts links, but its hashtag median is -0.38, the z-score for no hashtag
# at all - the high hashtag mean comes from a minority


# 6. read the clusters against the crowd label. The clusters were built without the label,
# so any difference between them is something the behaviour alone found
mix = pd.crosstab(df["cluster"], df["is_human"].map({1: "human", 0: "non_human"}))
mix["human_rate"] = (mix["human"] / mix.sum(axis=1)).round(3)
mix["unlabelled"] = df.loc[~labelled, "cluster"].value_counts().sort_index()
print("\nlabel mix per cluster (human rate of the whole data:", round(truth.mean(), 3), ")")
print(mix)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
mix[["non_human", "human"]].plot.bar(stacked=True, ax=axes[0], rot=0, color=LABEL_COLOURS)
axes[0].set_ylabel("profiles")
axes[0].set_title("how the labels fall in each cluster")
axes[1].bar(mix.index, mix["human_rate"], color=LABEL_COLOURS["human"])
axes[1].set_xticks(mix.index)   # one tick per cluster, not a number scale
axes[1].axhline(truth.mean(), color="red", linestyle="--", label="rate of the whole data")
axes[1].set_xlabel("cluster")
axes[1].set_ylabel("human rate")
axes[1].set_title("share of each cluster labelled human")
axes[1].legend()
plt.tight_layout()
plt.show()


# 7. visualise the clusters: find the attributes whose means differ most between clusters,
# then plot pairs of them with a different colour for each cluster
distinct = (cluster_mean.max() - cluster_mean.min()) / cluster_mean.abs().max()
print("\nhow different each attribute is between clusters:\n",
      distinct.sort_values(ascending=False).round(2))
top = list(distinct.sort_values(ascending=False).index[:4])
# draw the biggest cluster first so the small ones stay visible on top
by_size = df["cluster"].value_counts().index
CLUSTER_COLOURS = ["tab:purple", "tab:green", "tab:red", "tab:olive"]

fig, axs = plt.subplots(2, 3, figsize=(15, 9))
i2, j2 = 0, 0
for i in range(len(top) - 1):
    for j in range(i + 1, len(top)):
        if j2 > 2:
            j2 = 0
            i2 += 1
        for c in by_size:
            hit = df["cluster"] == c
            axs[i2, j2].scatter(df.loc[hit, top[i]], df.loc[hit, top[j]], s=3, alpha=0.3,
                                color=CLUSTER_COLOURS[c], label="cluster " + str(c))
        centres = km.cluster_centers_[:, [FEATURES.index(top[i]), FEATURES.index(top[j])]]
        axs[i2, j2].scatter(centres[:, 0], centres[:, 1], c="black", marker="X", s=120)
        axs[i2, j2].set_title("{} vs {}".format(top[i], top[j]))
        j2 += 1
axs[0, 0].legend(markerscale=4)
plt.suptitle("clusters on the four most distinct attributes (black X = centroid)")
plt.tight_layout()
plt.show()

# the most distinct pair twice: coloured by cluster, then by the crowd label
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for c in by_size:
    hit = df["cluster"] == c
    axes[0].scatter(df.loc[hit, top[0]], df.loc[hit, top[1]], s=4, alpha=0.3,
                    color=CLUSTER_COLOURS[c], label="cluster " + str(c))
axes[0].legend(markerscale=4)
axes[0].set_title("clusters found by k-means")
for value, name in [(1, "human"), (0, "non_human")]:
    hit = df["is_human"] == value
    axes[1].scatter(df.loc[hit, top[0]], df.loc[hit, top[1]], s=4, alpha=0.3, label=name,
                    color=LABEL_COLOURS[name])
axes[1].set_title("the crowd label, same points")
axes[1].legend(markerscale=4)
for ax in axes:
    ax.set_xlabel(top[0])
    ax.set_ylabel(top[1])
plt.tight_layout()
plt.show()


# 8. hierarchical clustering, to check the groups without choosing k in advance.
# It needs the distance between every pair of profiles (175 million pairs for all 18,715),
# so it runs on a random sample
part = np.random.RandomState(SEED).choice(len(df), SAMPLE, replace=False)
sample = df.iloc[part].copy()
dist = pdist(sample[FEATURES], "euclidean")

# the four ways of measuring the distance between two clusters, each cut into K_MAIN groups
for method in ["single", "complete", "average", "centroid"]:
    if method == "centroid":
        linkage_matrix = linkage(sample[FEATURES], method=method)   # needs the raw points
    else:
        linkage_matrix = linkage(dist, method=method)
    labels = cut_tree(linkage_matrix, n_clusters=K_MAIN).ravel()
    print(method, "linkage, cut into", K_MAIN, "- cluster sizes:", np.bincount(labels).tolist())

# At four clusters every method puts almost the whole sample in one group and splits off a few
# far-away profiles: single and centroid linkage chain everything together, and complete and
# average linkage give the outliers their own clusters. Complete linkage, used in the lab,
# is kept and the tree is cut lower down, where the main group starts to split
linkage_matrix = linkage(dist, method="complete")
cut = (linkage_matrix[-K_TREE, 2] + linkage_matrix[-K_TREE + 1, 2]) / 2
plt.figure(figsize=(15, 7))
dendrogram(linkage_matrix, truncate_mode="lastp", p=40, no_labels=True, color_threshold=cut)
plt.axhline(cut, color="red", linestyle="--", label="cut into " + str(K_TREE) + " clusters")
plt.ylabel("merge distance")
plt.title("complete linkage dendrogram (" + str(SAMPLE) + " profiles)")
plt.legend()
plt.show()

sample["tree_cluster"] = cut_tree(linkage_matrix, n_clusters=K_TREE).ravel()
branches = sample.groupby("tree_cluster").agg(
    size=("tree_cluster", "size"), human_rate=("is_human", "mean"),
    default_image=("default_image", "mean"), median_likes=("fav_number", "median"),
    median_tweets_per_day=("tweets_per_day", "median"))
print("\nhierarchical clusters (complete linkage, cut into", K_TREE, "):")
print(branches.sort_values("size", ascending=False).round(2))
# where the profiles of each hierarchical cluster went in k-means
print("\nhierarchical cluster (rows) vs k-means cluster (columns):")
print(pd.crosstab(sample["tree_cluster"], sample["cluster"]))

# One hierarchical cluster matches the bot-like k-means cluster 2: 332 profiles, 7% human, 92%
# with the default picture, all but a handful in cluster 2. A method that does not use
# centroids or a chosen k found the same group


# 9. DBSCAN finds dense regions and decides the number of clusters itself. It needs a radius
# (Eps) and a minimum number of points within it (MinPts); try a range of radii
MIN_PTS = 20
for eps in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    dbscan = DBSCAN(eps=eps, min_samples=MIN_PTS).fit(sample[FEATURES])
    found = len(set(dbscan.labels_) - {-1})
    print("Eps", eps, "| clusters:", found,
          "| noise points:", str(round((dbscan.labels_ == -1).mean() * 100)) + "%")

# A small radius leaves nearly everything as noise; a radius large enough to keep most points
# joins them into one cluster. There is no empty space between the groups for DBSCAN to use,
# and DBSCAN is known to struggle with varying densities and many attributes. Splitting the
# data into parts (k-means) suits it better
dbscan = DBSCAN(eps=2.0, min_samples=MIN_PTS).fit(sample[FEATURES])
noise = dbscan.labels_ == -1
plt.scatter(sample.loc[~noise, top[0]], sample.loc[~noise, top[1]], s=4, alpha=0.4,
            label="in a cluster")
plt.scatter(sample.loc[noise, top[0]], sample.loc[noise, top[1]], s=4, alpha=0.6,
            color="grey", label="noise")
plt.xlabel(top[0])
plt.ylabel(top[1])
plt.title("DBSCAN, Eps = 2.0, MinPts = " + str(MIN_PTS))
plt.legend(markerscale=4)
plt.show()


# 10. finer clusters for the actual question. Four clusters describe the data well but they are
# too broad to judge a single profile: even the most one-sided of them is 19% human. Once
# clusters are found, each can be given the label most of its members have - a cluster that
# nearly all agrees on one label is then used to question members that carry the other one
# the number of finer clusters and the two purity cut-offs are choices, so check how many
# profiles each combination would flag before fixing them
sweep = []
for k in [8, 10, 12, 14, 16]:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    km.fit(X)
    groups = pd.Series(km.predict(X), index=df.index)
    rate = truth.groupby(groups[labelled]).agg("mean")
    for pure_h, pure_nh in [(0.85, 0.35), (0.85, 0.30), (0.90, 0.30), (0.90, 0.20)]:
        says = groups.map(pd.Series(np.where(rate >= pure_h, 1,
                                             np.where(rate <= pure_nh, 0, np.nan)), index=rate.index))
        hit = labelled & says.notna() & (says != df["is_human"])
        sweep.append({"k": k, "cut-offs": str(pure_h) + " / " + str(pure_nh),
                      "flagged": int(hit.sum())})
print("\nprofiles flagged for each k and purity cut-offs (human / non-human):")
print(pd.DataFrame(sweep).pivot(index="k", columns="cut-offs", values="flagged"))
# k=12 flags the same 279 profiles under every cut-off, the other k values move between
# about 200 and 950, so k=12 is used and the result does not rest on the exact cut-off

fine = KMeans(n_clusters=K_FINE, n_init=10, random_state=SEED)
fine.fit(X)
df["fine_cluster"] = fine.predict(X)
rate = truth.groupby(df.loc[labelled, "fine_cluster"]).agg(human_rate="mean", labelled="size")
rate["size"] = df["fine_cluster"].value_counts()
# a cluster is one-sided if nearly all of its labelled members agree with each other
rate["leans"] = np.where(rate["human_rate"] >= PURE_HUMAN, "human",
                         np.where(rate["human_rate"] <= PURE_NON_HUMAN, "non_human", "mixed"))
print("\n", K_FINE, "finer clusters, sorted by how human they look:")
print(rate.sort_values("human_rate").round(3))
print("one-sided clusters:", (rate["leans"] != "mixed").sum(), "of", K_FINE)


# 11. a profile sitting in a one-sided cluster but carrying the opposite label is a candidate
# mislabel: everything about its behaviour matches the profiles it was not grouped with
leaning = rate[rate["leans"] != "mixed"]
suspect = df["fine_cluster"].map(leaning["leans"]).where(labelled)
cluster_says = suspect.map({"human": 1, "non_human": 0})
flagged = df[cluster_says.notna() & (cluster_says != df["is_human"])].copy()
flagged["cluster_says"] = suspect[flagged.index]
flagged["cluster_human_rate"] = flagged["fine_cluster"].map(rate["human_rate"]).round(3)
# shared output format: says = what the cluster suggests, score = how pure that cluster is
flagged["says"] = flagged["cluster_says"]
flagged["score"] = np.where(flagged["says"] == "human", flagged["cluster_human_rate"],
                            1 - flagged["cluster_human_rate"]).round(3)
# the Euclidean distance from the profile to its cluster centroid: a small distance means it
# is a typical member of a cluster that disagrees with its label, so the flag is harder to dismiss
flagged["distance_to_centre"] = np.linalg.norm(
    X.loc[flagged.index].to_numpy() - fine.cluster_centers_[flagged["fine_cluster"]],
    axis=1).round(2)

print("\nprofiles whose cluster contradicts their label:", len(flagged))
print(flagged["gender"].value_counts())
print(flagged.groupby("cluster_says")[["cluster_human_rate", "gender:confidence"]].mean().round(3))


# 12. check the flags against something the clustering never saw: how sure the crowd was.
# If the flags were noise the two distributions would sit on top of each other
print("\ncrowd confidence, flagged vs all labelled profiles:")
print(pd.DataFrame({"flagged": flagged["gender:confidence"].describe(),
                    "all": df.loc[labelled, "gender:confidence"].describe()}).round(3))
print("below full confidence - flagged:", round((flagged["gender:confidence"] < 1).mean(), 3),
      "| all:", round((df.loc[labelled, "gender:confidence"] < 1).mean(), 3))

plt.figure(figsize=(8, 4))
# the two groups differ a lot in size (279 vs 17,678), so each bar is the percentage of its own
# group rather than a count - that way the two shapes can be compared directly
confidence = [flagged["gender:confidence"], df.loc[labelled, "gender:confidence"]]
plt.hist(confidence, bins=20, weights=[np.full(len(c), 100 / len(c)) for c in confidence],
         label=["flagged by clustering", "all labelled profiles"])
plt.xlabel("gender:confidence")
plt.ylabel("% of profiles in the group")
plt.title("the crowd was less sure about the profiles the clusters flagged")
plt.legend()
plt.show()

# the association rules flagged profiles the same way, from different evidence.
# Profiles found by both methods are the strongest candidates
rules_file = OUT / "association_flagged.csv"
if rules_file.exists():
    by_rules = set(pd.read_csv(rules_file)["_unit_id"])
    both = by_rules & set(flagged["_unit_id"])
    flagged["also_by_rules"] = flagged["_unit_id"].isin(by_rules).astype(int)
    print("\nflagged by the association rules:", len(by_rules),
          "| by clustering:", len(flagged), "| by both:", len(both))


# 13. the 1037 profiles the crowd could not label still fall into a cluster, so the cluster
# they landed in is a suggestion for what they most likely are
unlabelled = df.loc[~labelled].copy()
unlabelled["suggested"] = unlabelled["fine_cluster"].map(leaning["leans"])
unlabelled["cluster_human_rate"] = unlabelled["fine_cluster"].map(rate["human_rate"]).round(3)
print("\nsuggestions for the unlabelled profiles:")
print(unlabelled["suggested"].value_counts(dropna=False))


# 14. write the three lists out for the report.
# The flagged list is sorted so the most one-sided cluster comes first, and inside a cluster
# the profiles closest to its centroid come first
KEEP = ["_unit_id", "says", "score", "name", "gender", "gender:confidence", "cluster",
        "fine_cluster", "cluster_human_rate", "distance_to_centre"]
if "also_by_rules" in flagged:
    KEEP.append("also_by_rules")
flagged = flagged.sort_values(["cluster_human_rate", "distance_to_centre"])
flagged[KEEP].to_csv(OUT / "clustering_flagged.csv", index=False)

suggestions = unlabelled[unlabelled["suggested"].notna()]
suggestions[["_unit_id", "name", "cluster", "fine_cluster", "cluster_human_rate",
             "suggested"]].to_csv(OUT / "cluster_suggested_labels.csv", index=False)

df[["_unit_id", "name", "gender", "is_human", "cluster", "fine_cluster"]].to_csv(
    OUT / "cluster_assignments.csv", index=False)
print("\nwritten:", [f.name for f in sorted(OUT.glob("cluster*.csv"))])
print("\nclearest candidates (typical members of clusters that disagree with their label):")
print(flagged.head(10)[["name", "gender", "gender:confidence", "says",
                        "cluster_human_rate", "distance_to_centre"]])
