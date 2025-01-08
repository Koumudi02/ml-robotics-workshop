import cv2
import zipfile
import os
from pathlib import Path

def extract_images(zip_path, dest_dir):
  
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
      
        zip_ref.extractall(dest_dir)

def detect_faces_in_image(image_path, cascade_classifier):
  
    img = cv2.imread(image_path)
    if img is None:
      
        return None

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(gray_img, 1.1, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return img

def main():
  
    zip_path = 'images.zip'
    extracted_dir = 'extracted_images'
    output_dir = 'output_images'
    Path(extracted_dir).mkdir(parents=True, exist_ok=True)
  
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    extract_images(zip_path, extracted_dir)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for image_file in os.listdir(extracted_dir):
      
        image_path = os.path.join(extracted_dir, image_file)
        
        if os.path.isfile(image_path):
          
            result_img = detect_faces_in_image(image_path, face_cascade)
            
            if result_img is not None:
              
                output_image_path = os.path.join(output_dir, f"detected_{image_file}")
                cv2.imwrite(output_image_path, result_img)
                print(f"Processed image saved: {output_image_path}")

    print("Object detection completed!!!")

if __name__ == "__main__":
  
    main()
