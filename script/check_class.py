import pandas as pd

df = pd.read_csv("outputs/dataset_landmarks.csv")

print("=" * 50)
print("Dataset Shape")
print("=" * 50)
print(df.shape)

print("\nLabels Found")
print("=" * 50)
print(sorted(df["label"].unique()))

print("\nSamples Per Class")
print("=" * 50)
print(df["label"].value_counts().sort_index())

print("\nClasses with fewer than 10 samples")
print("=" * 50)
print(df["label"].value_counts()[df["label"].value_counts() < 10])

print("\nMissing Values")
print("=" * 50)
print(df.isnull().sum().sum())