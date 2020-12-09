import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/train.csv")

identities = [
    "male",
    "female",
    "transgender",
    "other_gender",
    "heterosexual",
    "homosexual_gay_or_lesbian",
    "bisexual",
    "other_sexual_orientation",
    "christian",
    "jewish",
    "muslim",
    "hindu",
    "buddhist",
    "atheist",
    "other_religion",
    "black",
    "white",
    "asian",
    "latino",
    "other_race_or_ethnicity",
    "physical_disability",
    "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness",
    "other_disability",
]

# counts on toxic/non-toxic comments for each identity
labeled_df = df.loc[:, ["target"] + identities].dropna()
toxic_df = labeled_df[labeled_df["target"] >= 0.5][identities]
non_toxic_df = labeled_df[labeled_df["target"] < 0.5][identities]

toxic_count = toxic_df.where(labeled_df == 0, other=1).sum()
non_toxic_count = non_toxic_df.where(labeled_df == 0, other=1).sum()

toxic_non_toxic = pd.concat([toxic_count, non_toxic_count], axis=1)
toxic_non_toxic = toxic_non_toxic.rename(
    index=str, columns={1: "non-toxic", 0: "toxic"}
)
toxic_non_toxic.sort_values(by="toxic").plot(
    kind="bar",
    stacked=True,
    color={"toxic": "#F34C50", "non-toxic": "#3AA7A0"},
    figsize=(15, 5),
    fontsize=10,
).legend(prop={"size": 20})

# percentage of toxic comments on each identity
weighted_toxic = (
    labeled_df.iloc[:, 1:].multiply(labeled_df.iloc[:, 0], axis="index").sum()
)
identity_label_count = labeled_df[identities].where(labeled_df == 0, other=1).sum()
weighted_toxic = weighted_toxic / identity_label_count
weighted_toxic = weighted_toxic.sort_values(ascending=False)
plt.figure(figsize=(15, 10))
sns.set(font_scale=1)
ax = sns.barplot(
    x=weighted_toxic.values, y=weighted_toxic.index, palette="coolwarm", alpha=0.8
)
plt.xlabel("Weighted Toxicity")
plt.title("Weighted Analysis of Most Frequent Identities")
plt.show()

# Reference: https://www.kaggle.com/ekhtiar/unintended-eda-with-tutorial-notes
