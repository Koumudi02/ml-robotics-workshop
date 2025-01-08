import os
import zipfile
import cv2
import numpy as np
from pathlib import Path

ZIP_FILE = 'images.zip'
EXTRACT_DIR = 'extracted_images'
OUTPUT_DIR = 'processed_images'

Path(EXTRACT_DIR).mkdir(parents=True, exist_ok=True)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
  
    zip_ref.extractall(EXTRACT_DIR)

def process_image(image_path, output_dir):
  
    image = cv2.imread(image_path)
  
    if image is None:
      
        return None, None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    corners = cv2.cornerHarris(gray_image, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    image[corners > 0.01 * corners.max()] = [0, 0, 255]
  
    edge_output_path = os.path.join(output_dir, f"edges_{os.path.basename(image_path)}")
  
    corner_output_path = os.path.join(output_dir, f"corners_{os.path.basename(image_path)}")
    return edges, image, edge_output_path, corner_output_path

for file in os.listdir(EXTRACT_DIR):
  
    img_file_path = os.path.join(EXTRACT_DIR, file)
    if os.path.isfile(img_file_path):
      
        edges, corner_image, edge_path, corner_path = process_image(img_file_path, OUTPUT_DIR)
        if edges is not None:
          
            cv2.imwrite(edge_path, edges)
            cv2.imwrite(corner_path, corner_image)
            print(f"Processed images saved: {edge_path}, {corner_path}")

print("Edge and corner detection completed!")
