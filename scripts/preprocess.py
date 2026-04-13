import os
import cv2
import argparse
import numpy as np
import glob
from pathlib import Path

try:
    from mtcnn import MTCNN
except ImportError:
    print("MTCNN not found. Please install via: pip install mtcnn")
    MTCNN = None

def apply_clahe(img):
    """Apply CLAHE to normalize lighting."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def align_and_crop(img, detector, target_size=(768, 768)):
    """Detect face, align and pad to square."""
    if detector is None:
        # Fallback to center crop and resize if no detector
        h, w = img.shape[:2]
        min_dim = min(h, w)
        cy, cx = h // 2, w // 2
        cropped = img[cy - min_dim//2:cy + min_dim//2, cx - min_dim//2:cx + min_dim//2]
        return cv2.resize(cropped, target_size)

    faces = detector.detect_faces(img)
    if not faces:
        return None
    
    # Take largest face
    faces = sorted(faces, key=lambda x: x['box'][2] * x['box'][3], reverse=True)
    x, y, width, height = faces[0]['box']
    
    # Expand bounding box slightly for context
    margin_x = int(width * 0.2)
    margin_y = int(height * 0.2)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img.shape[1], x + width + margin_x)
    y2 = min(img.shape[0], y + height + margin_y)
    
    face_crop = img[y1:y2, x1:x2]
    
    # Pad to make it square
    h, w = face_crop.shape[:2]
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    
    padded = cv2.copyMakeBorder(face_crop, pad_h, max_dim - h - pad_h, pad_w, max_dim - w - pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(padded, target_size)

def generate_augmentations(img):
    """Generate Canny maps and Stylized sketches."""
    # OpenCV Pencil Sketch
    gray, color_sketch = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    
    # Canny Edge Map
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    canny = cv2.Canny(blurred, 50, 150)
    canny_3channel = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    
    return gray, canny_3channel

def process_dataset(input_dir, output_dir, resolution=768):
    os.makedirs(output_dir, exist_ok=True)
    detector = MTCNN() if MTCNN is not None else None
    
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(Path(input_dir).rglob(ext))
        
    print(f"Found {len(image_paths)} images to process...")
    
    for idx, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        if img is None:
            continue
            
        # 1. Align and crop
        aligned = align_and_crop(img, detector, target_size=(resolution, resolution))
        if aligned is None:
            continue
            
        # 2. Normalize lighting
        normalized = apply_clahe(aligned)
        
        # 3. Generate augmentations
        sketch, canny = generate_augmentations(normalized)
        
        # Save pairs
        base_name = f"sample_{idx:05d}"
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_photo.png"), normalized)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_sketch.png"), sketch)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_canny.png"), canny)
        
        # If we need 512 for controlnet, we can save a copy or let training script resize
        cn_dir = os.path.join(output_dir, 'controlnet_512')
        os.makedirs(cn_dir, exist_ok=True)
        cv2.imwrite(os.path.join(cn_dir, f"{base_name}_photo.png"), cv2.resize(normalized, (512, 512)))
        cv2.imwrite(os.path.join(cn_dir, f"{base_name}_sketch.png"), cv2.resize(sketch, (512, 512)))
        
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(image_paths)} images")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../data/raw', help='Path to raw datasets')
    parser.add_argument('--output_dir', type=str, default='../data/cleaned_dataset/train', help='Path to output cleaned data')
    parser.add_argument('--resolution', type=int, default=768, help='Resolution for LoRA (768)')
    args = parser.parse_args()
    
    root_dir = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(root_dir, args.input_dir.lstrip("../")) if args.input_dir.startswith("../") else args.input_dir
    output_path = os.path.join(root_dir, args.output_dir.lstrip("../")) if args.output_dir.startswith("../") else args.output_dir
    
    if not os.path.exists(input_path):
        print(f"Input directory {input_path} does not exist. Please run download_datasets.py or place data.")
        return
        
    process_dataset(input_path, output_path, args.resolution)
    print("Preprocessing completed!")

if __name__ == "__main__":
    main()
