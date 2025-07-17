import cv2
import os
import glob
from pathlib import Path

# Framerate variable - you can change this to control video speed
FRAMERATE = 20

def createvideo(image_paths, output_path='output_video.mp4', framerate=FRAMERATE):
    """
    Create a video from a list of image file paths.
    
    Args:
        image_paths (list): List of image file paths
        output_path (str): Output video file path
        framerate (int): Frames per second for the video
    """
    if not image_paths:
        print("No images provided!")
        return
    
    # Sort images based on filename
    sorted_images = sorted(image_paths, key=lambda x: os.path.basename(x))
    
    # Read the first image to get dimensions
    first_image = cv2.imread(sorted_images[0])
    if first_image is None:
        print(f"Error reading first image: {sorted_images[0]}")
        return
    
    height, width, layers = first_image.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, framerate, (width, height))
    
    print(f"Creating video with {len(sorted_images)} images at {framerate} FPS...")
    
    # Process each image
    for i, image_path in enumerate(sorted_images):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        # Resize image if dimensions don't match
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        video_writer.write(img)
        
        # Show progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(sorted_images)} images")
    
    # Release everything
    video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved as: {output_path}")

def initVideo(folder_path='saved', output_path='output_video.mp4'):
    """
    Initialize video creation by reading all image files from a folder.
    
    Args:
        folder_path (str): Path to the folder containing images
        output_path (str): Output video file path
    """
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        return
    
    # Supported image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # Collect all image files
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
        image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))
    
    if not image_files:
        print(f"No image files found in folder: {folder_path}")
        return
    
    print(f"Found {len(image_files)} image files in '{folder_path}'")
    
    # Call createvideo method
    createvideo(image_files, output_path)