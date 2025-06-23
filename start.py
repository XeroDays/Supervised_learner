import os

# Define the path to the 'dataset' folder inside the current directory
dataset_path = os.path.join(os.getcwd(), 'dataset')

# Check if the dataset directory exists
if not os.path.exists(dataset_path):
    print(f"The directory '{dataset_path}' does not exist.")
else:
    # List all files in the dataset folder
    files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

    # Print all file names
    print("Files in 'dataset' folder:")
    for file in files:
        print(file)

    # Optional: Read the content of each file
    for file in files:
        file_path = os.path.join(dataset_path, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"\nContents of {file}:\n{content[:200]}...")  # Print first 200 characters
        except Exception as e:
            print(f"Could not read {file}: {e}")
