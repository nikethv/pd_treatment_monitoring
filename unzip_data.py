import zipfile
import os

# Define paths
zip_file = "filtered_data.zip"
extract_folder = "filtered_data"

# Make sure the output folder exists
os.makedirs(extract_folder, exist_ok=True)

# Unzip
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print(f"✅ Unzipped '{zip_file}' into folder '{extract_folder}'.")
