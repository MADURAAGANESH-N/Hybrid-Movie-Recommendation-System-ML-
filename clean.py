import pandas as pd
import re

# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

input_file = input_file = "C:/Users/Admin/.cache/kagglehub/datasets/grouplens/movielens-latest-full/versions/1/movies.csv"
output_file = "movies_cleaned.csv"

df = pd.read_csv(input_file)

print("Original shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())


# -----------------------------------------
# 2. Basic cleaning
# -----------------------------------------

# Remove leading/trailing spaces
df["title"] = df["title"].astype(str).str.strip()
df["genres"] = df["genres"].astype(str).str.strip()


# -----------------------------------------
# 3. Check missing values
# -----------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# -----------------------------------------
# 4. Check duplicate rows
# -----------------------------------------

print("\nDuplicate rows:", df.duplicated().sum())


# Remove exact duplicate rows
df = df.drop_duplicates()


# -----------------------------------------
# 5. Check duplicate movie IDs
# -----------------------------------------

print("\nDuplicate movie IDs:", df["movieId"].duplicated().sum())

# Keep first occurrence if duplicate movieId exists
df = df.drop_duplicates(subset="movieId", keep="first")


# -----------------------------------------
# 6. Extract movie year from title
# -----------------------------------------

def extract_year(title):
    match = re.search(r"\((\d{4})\)\s*$", title)

    if match:
        return int(match.group(1))

    return None


df["year"] = df["title"].apply(extract_year)


# -----------------------------------------
# 7. Remove year from title
# -----------------------------------------

df["title"] = df["title"].str.replace(
    r"\s*\(\d{4}\)\s*$",
    "",
    regex=True
)

df["title"] = df["title"].str.strip()


# -----------------------------------------
# 8. Clean genres
# -----------------------------------------

df["genres"] = df["genres"].str.strip()

# Convert "(no genres listed)" to "Unknown"
df["genres"] = df["genres"].replace(
    "(no genres listed)",
    "Unknown"
)


# -----------------------------------------
# 9. Standardize genre formatting
# -----------------------------------------

df["genres"] = df["genres"].apply(
    lambda x: "|".join(
        genre.strip().title()
        for genre in x.split("|")
    )
)


# -----------------------------------------
# 10. Check invalid years
# -----------------------------------------

# print("\nMissing years:", df["year"].isnull().sum())

# print("\nYear range:")
# print(df["year"].min(), "-", df["year"].max())


# -----------------------------------------
# 11. Check final dataset
# -----------------------------------------

# print("\nFinal shape:", df.shape)

# print("\nFinal data:")
# print(df.head())

# print("\nFinal data types:")
# print(df.dtypes)

# print("\nFinal missing values:")
# print(df.isnull().sum())


# -----------------------------------------
# 12. Save cleaned dataset
# -----------------------------------------

# df.to_csv(output_file, index=False)
print("Duplicate rows:", df.duplicated().sum())
duplicates = df[df.duplicated(keep=False)]

print(duplicates)
# print(f"\nCleaned dataset saved as: {output_file}")