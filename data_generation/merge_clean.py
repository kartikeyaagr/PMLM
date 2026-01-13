import pandas as pd

df1 = pd.read_csv("data/data.csv")
df2 = pd.read_csv("data/data2.csv")

df = pd.concat([df1, df2], axis=0)

df.drop(columns=["source", "velocity", "status"], inplace=True)

# Separate the classes
df_toxic = df[df["label"] == "passive_aggressive"]
df_safe = df[df["label"] != "passive_aggressive"]

# Find the lowest common denominator
min_count = len(df_safe)

# Downsample the majority class
df_toxic_balanced = df_toxic.sample(n=min_count, random_state=42)

# Recombine
df_balanced = pd.concat([df_toxic_balanced, df_safe])

# Shuffle
df_final = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(df_final["label"].value_counts(normalize=True))


df_final.to_csv("balanced_data.csv", index=False)
