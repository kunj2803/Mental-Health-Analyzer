import gdown
import os

# Google Drive file ID
file_id = "1rcN8X8PLUOlm5fGvRzj3lwfnZytdqrsi"

# Output path to save the file
output_path = "model/model.safetensors"

# Make sure the directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Download the file
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

print(f"Model downloaded successfully to {output_path}")
