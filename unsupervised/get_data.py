import kagglehub

# Download latest version
data_path = "unsupervised/data/data.csv"
path = kagglehub.dataset_download("imakash3011/customer-personality-analysis")

print("Path to dataset files:", path)