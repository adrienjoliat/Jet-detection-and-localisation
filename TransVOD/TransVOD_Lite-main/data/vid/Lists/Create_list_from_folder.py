import os

def list_files_in_folder(folder_path):
    # Initialize an empty list to store file names
    file_list = []
    
    # Iterate over all files and subdirectories in the specified folder
    for root, dirs, files in os.walk(folder_path):
        # Iterate over each file in the current directory
        for file in files:
            # Get the full path of the file
            file_path = os.path.join(root, file)
            # Add the file path to the list
            file_list.append(file_path)
    
    # Return the list of file paths
    return file_list

def save_file_list_to_txt(file_list, output_file):
    # Open the output file in write mode
    with open(output_file, 'w') as f:
        # Write each file path to a new line in the text file
        for file_path in file_list:
            f.write(file_path + '\n')

# Specify the folder path
folder_path = './TransVOD/TransVOD_Lite-main/data/vid/Data/VID'
# Specify the output file path
output_file = './TransVOD/TransVOD_Lite-main/data/vid/Lists/VID_train_15frames.txt'

# Call the function to list all files in the folder
files = list_files_in_folder(folder_path)

# Call the function to save the list of files to a text file
save_file_list_to_txt(files, output_file)

print(f'List of files saved to {output_file}')
