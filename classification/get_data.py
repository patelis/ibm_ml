import kagglehub
import shutil
import argparse
import os

# Set up parser
parser = argparse.ArgumentParser(description='Script to download kaggle data')
parser.add_argument('--destination', action="store", dest='destination', default='/home/')
args = parser.parse_args()

# Download latest version
path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")

files=os.listdir(path)

# Iterating over all the files in the source directory
for fname in files:
    # copying the files to the destination directory
    shutil.copy2(os.path.join(path,fname), args.destination)

print("Path to initial download location:", path)
print("Path to dataset files:", args.destination)