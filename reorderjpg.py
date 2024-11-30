import os

def rename_files(folder_path, outfolder_path):
    # Get all files in the folder
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    
    # Rename each file sequentially
    for index, filename in enumerate(files, start=1):
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(outfolder_path, f"{index+1}.jpg")
        os.rename(old_file, new_file)
        print(f"Renamed: {old_file} -> {new_file}")

if __name__ == "__main__":
    folder_path = "JPEGImages"  # Replace with your folder path
    outfolder_path = "reorder"  # Replace with your folder path
    rename_files(folder_path, outfolder_path)

