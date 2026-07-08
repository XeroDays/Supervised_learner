import os
import sys

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

def select_feature():
    """Allow user to select which feature to use"""
    print("\nAvailable features:")
    print("-" * 20)
    print("1. Process Images (Detection)")
    print("2. Compare Models")
    print("3. Train Model")
    
    while True:
        try:
            choice = input("\nSelect a feature (1-3): ").strip()
            choice_num = int(choice)
            if choice_num == 1:
                return "detection"
            elif choice_num == 2:
                return "compare"
            elif choice_num == 3:
                return "train"
            else:
                print("Please enter 1, 2, or 3")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


def select_training_model():
    """Allow user to select which YOLO model to train"""
    from Engine.train import TRAINING_MODEL_OPTIONS

    print("\nAvailable training models:")
    print("-" * 30)
    for i, option in enumerate(TRAINING_MODEL_OPTIONS, 1):
        print(f"{i}. {option['label']}")

    while True:
        try:
            choice = input(f"\nSelect a model (1-{len(TRAINING_MODEL_OPTIONS)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(TRAINING_MODEL_OPTIONS):
                selected = TRAINING_MODEL_OPTIONS[choice_num - 1]
                print(f"Selected model: {selected['label']} ({selected['base_model']})")
                return selected["base_model"]
            print(f"Please enter a number between 1 and {len(TRAINING_MODEL_OPTIONS)}")
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
    
    # Step 1: Select feature
    feature = select_feature()

    if feature == "train":
        base_model = select_training_model()
        print("\nStarting model training...")
        print("-" * 30)
        try:
            from Engine.train import train_model
            train_model(base_model=base_model)
        except Exception as e:
            print(f"Error running training: {e}")
            return
        print("\nTraining process completed!")
        return

    # Step 2: Select model folder
    selected_folder = select_model_folder()
    if not selected_folder:
        return
    
    folder_path = os.path.join("models", selected_folder)
    
    if feature == "detection":
        # Step 3: Select model file for detection
        selected_model = select_model_file(folder_path)
        if not selected_model:
            return
        
        # Step 4: Create full model path
        model_path = os.path.join(folder_path, selected_model)
        print(f"\nSelected model path: {model_path}")
        
        # Step 5: Pass model path to start.py
        print("\nStarting detection process...")
        print("-" * 30)
        
        try:
            from Engine.start import main as start_main
            start_main(model_path)
        except Exception as e:
            print(f"Error running detection: {e}")
            return
        
        print("\nDetection process completed!")
    
    elif feature == "compare":
        # Step 3: Compare all models in the folder
        print(f"\nStarting model comparison for '{selected_folder}' folder...")
        print("-" * 50)
        
        try:
            from Engine.compare import compare_models
            compare_models(folder_path)
        except Exception as e:
            print(f"Error running model comparison: {e}")
            return
        
        print("\nModel comparison completed!")

if __name__ == "__main__":
    main()
