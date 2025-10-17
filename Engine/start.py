import os
from .yolo_detection import detect_objects, draw_and_save_detections, save_yolo_txt, save_classes_txt, delete_txt_files, set_model_path
from create_video import initVideo

def main(model_path):
    """Main function that accepts model path"""
    # Set the model path in yolo_detection module
    set_model_path(model_path)
    
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    if not os.path.exists(dataset_path):
        print(f"The directory '{dataset_path}' does not exist.")
        return
    
    files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    print(f"Total number of files in 'dataset' folder: {len(files)}") 
    print(f"Total number of image files: {len([f for f in files if f.lower().endswith(image_extensions)])}")
    print(f"Total number of text files: {len([f for f in files if not f.lower().endswith(image_extensions)])}")
 
    # delete_txt_files(files)
    serial_number = 1
    for file_name in files:
        if not file_name.lower().endswith(image_extensions):
            continue

        image_path = os.path.join(dataset_path, file_name)
        detections = detect_objects(image_path)
        if detections:
            print(f"\n{serial_number}. Detections in {file_name}: Count {len(detections)}") 
            draw_and_save_detections(image_path, detections)
            save_yolo_txt(image_path, detections)
            serial_number += 1
        else:
            print(f"No detections above threshold in {file_name}")

    save_classes_txt()
    initVideo()

# For backward compatibility, keep the original execution if run directly
if __name__ == "__main__":
    # Use default model path if run directly
    default_model_path = "models/receipts/best1.pt"
    main(default_model_path)