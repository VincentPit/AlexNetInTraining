import os
import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages')
import cv2
import numpy as np

def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype(np.float32) / 255.0
    gray_image = np.expand_dims(gray_image, axis=-1)
    return gray_image

def preprocess_images_in_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, filenames in os.walk(input_dir):
        
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)
        print(output_subdir)
        print(filenames)
        for filename in filenames:
            if filename.endswith(".jpg") or filename.endswith(".JPEG"):
                input_path = os.path.join(root, filename)
                output_path = os.path.join(output_subdir, filename[:-4] + ".npy")
                processed_image = preprocess_image(input_path)
                np.save(output_path, processed_image)
                print(f"Processed {input_path} and saved to {output_path}")

if __name__ == "__main__":
    input_directory = "imagenet-mini"
    output_directory = "greyScaleMini"
    train_input_dir = os.path.join(input_directory, "train")
    train_output_dir = os.path.join(output_directory, "train")
    preprocess_images_in_directory(train_input_dir, train_output_dir)
    val_input_dir = os.path.join(input_directory, "val")
    val_output_dir = os.path.join(output_directory, "val")
    preprocess_images_in_directory(val_input_dir, val_output_dir)
