import pandas as pd

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv("outputs/dataset_landmarks.csv")

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print(f"Rows before cleaning : {len(df)}")

# ============================================
# Remove Duplicate Rows
# ============================================

df = df.drop_duplicates()

print(f"Rows after duplicates removed : {len(df)}")

# ============================================
# Remove Classes with Less Than 2 Samples
# ============================================

counts = df["label"].value_counts()

bad_classes = counts[counts < 2].index.tolist()

if bad_classes:
    print("\nRemoving classes with fewer than 2 samples...")
    print("Removed labels:", bad_classes)

    df = df[~df["label"].isin(bad_classes)]
else:
    print("\nNo classes needed to be removed.")

# ============================================
# Save Clean Dataset
# ============================================

df.to_csv("outputs/dataset_landmarks_clean.csv", index=False)

print("\nClean dataset saved as:")
print("outputs/dataset_landmarks_clean.csv")

# ============================================
# Verify Clean Dataset
# ============================================

df = pd.read_csv("outputs/dataset_landmarks_clean.csv")

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("\nDataset Shape")
print(df.shape)

print("\nFirst 5 Rows")
print(df.head())

print("\nColumns")
print(df.columns.tolist())

print("\nLabels Present")
print(sorted(df["label"].unique()))

print("\nSamples Per Class")
print(df["label"].value_counts().sort_index())

print("\nClasses with fewer than 2 samples")
small = df["label"].value_counts()[df["label"].value_counts() < 2]

if len(small) == 0:
    print("None ✅")
else:
    print(small)

print("\nMissing Values")
print(df.isnull().sum().sum())

print("\nCleaning Completed Successfully!")