import cv2
import zipfile
import os
import numpy as np
from pathlib import Path

def extract_images(zip_file, extract_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

def process_image(image):
  
    resized_image = cv2.resize(image, (128, 128))
    normalized_image = resized_image / 255.0
    blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)
    return (blurred_image * 255).astype(np.uint8)

def save_processed_image(image, output_path):
    cv2.imwrite(output_path, image)

def main():
  
    zip_file = 'images.zip'
    extract_dir = 'resized_images'
    output_dir = 'processed_images'

    Path(extract_dir).mkdir(parents=True, exist_ok=True)
  
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    extract_images(zip_file, extract_dir)

    for file in os.listdir(extract_dir):
        img_file_path = os.path.join(extract_dir, file)
        
        if os.path.isfile(img_file_path):
            image = cv2.imread(img_file_path)
          
            if image is None:
                print(f"Failed to load image: {file}")
                continue

            processed_image = process_image(image)
            output_path = os.path.join(output_dir, file)
            save_processed_image(processed_image, output_path)

            print(f"Processed image saved: {output_path}")

    print("Image processing completed!")
  
if __name__ == "__main__":
    main()
