import kagglehub

# Download latest version
path = kagglehub.dataset_download("grouplens/movielens-latest-full")

print("Path to dataset files:", path)