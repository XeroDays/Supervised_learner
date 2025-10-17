import os
import sys
from Engine.start import main as start_main

def list_model_folders():
    """List all folders in the models directory"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"Error: '{models_dir}' directory does not exist.")
        return []
    
    folders = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]
    return sorted(folders)

def list_model_files(folder_path):
    """List all .pt files in the selected folder"""
    if not os.path.exists(folder_path):
        print(f"Error: '{folder_path}' directory does not exist.")
        return []
    
    model_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    return sorted(model_files)

def select_model_folder():
    """Allow user to select a model folder"""
    folders = list_model_folders()
    
    if not folders:
        print("No model folders found in the 'models' directory.")
        return None
    
    print("\nAvailable model folders:")
    print("-" * 30)
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")
    
    while True:
        try:
            choice = input(f"\nSelect a folder (1-{len(folders)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(folders):
                selected_folder = folders[choice_num - 1]
                print(f"Selected folder: {selected_folder}")
                return selected_folder
            else:
                print(f"Please enter a number between 1 and {len(folders)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

def select_model_file(folder_path):
    """Allow user to select a model file"""
    model_files = list_model_files(folder_path)
    
    if not model_files:
        print(f"No .pt model files found in '{folder_path}'")
        return None
    
    print(f"\nAvailable models in '{os.path.basename(folder_path)}':")
    print("-" * 40)
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file}")
    
    while True:
        try:
            choice = input(f"\nSelect a model (1-{len(model_files)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(model_files):
                selected_model = model_files[choice_num - 1]
                print(f"Selected model: {selected_model}")
                return selected_model
            else:
                print(f"Please enter a number between 1 and {len(model_files)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

def main():
    """Main entry point for the application"""
    print("=" * 50)
    print("YOLO Model Selection and Detection System")
    print("=" * 50)
    
    # Step 1: Select model folder
    selected_folder = select_model_folder()
    if not selected_folder:
        return
    
    folder_path = os.path.join("models", selected_folder)
    
    # Step 2: Select model file
    selected_model = select_model_file(folder_path)
    if not selected_model:
        return
    
    # Step 3: Create full model path
    model_path = os.path.join(folder_path, selected_model)
    print(f"\nSelected model path: {model_path}")
    
    # Step 4: Pass model path to start.py
    print("\nStarting detection process...")
    print("-" * 30)
    
    try:
        start_main(model_path)
    except Exception as e:
        print(f"Error running detection: {e}")
        return
    
    print("\nDetection process completed!")

if __name__ == "__main__":
    main()
