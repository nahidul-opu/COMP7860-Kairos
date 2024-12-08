import os
import subprocess

def convert_files_in_folder(folder_path):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    files = os.listdir(folder_path)

    if not files:
        print(f"No files found in the folder {folder_path}.")
        return

    print(f"Found {len(files)} bin files. Converting them now...")

    for bin_file in files:
        full_path = os.path.join(folder_path, bin_file)
        print(f"Converting:\n./json_consumer.sh {folder_path}/{bin_file} /home/cs/grad/opumni/Fall2024/COMP7860_Project/DatasetE5/Data/theia-json")

        try:
            # Execute the .sh file
            process = subprocess.Popen(f"./json_consumer.sh {folder_path}/{bin_file} /home/cs/grad/opumni/Fall2024/COMP7860_Project/DatasetE5/Data/theia-json",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )
            process.wait()
            stdout, stderr = process.communicate()
            if stderr is not None:
                print(f"Errors in {bin_file}:\n{stderr}")
            else:
            	os.remove(f"{folder_path}/{bin_file}")
        except Exception as e:
            print(f"An error occurred while running {bin_file}: {e}")

if __name__ == "__main__":
    folder_path = "/home/cs/grad/opumni/Fall2024/COMP7860_Project/DatasetE5/Data/theia"
    convert_files_in_folder(folder_path)
