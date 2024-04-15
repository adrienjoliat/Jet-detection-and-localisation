import os

# Specify the path to the folder containing the .npy files
folder_path = "./data_boxes/events"

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file has the .npy extension
    if filename.endswith(".npy"):
        # Extract the file number (e.g., "0", "1", etc.)
        file_number = os.path.splitext(filename)[0]

        # Pad the file number with leading zeros to make it three digits
        new_filename = file_number.zfill(3) + ".npy"

        # Construct the full paths for the old and new filenames
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_filepath, new_filepath)

        print(f"Renamed {filename} to {new_filename}")