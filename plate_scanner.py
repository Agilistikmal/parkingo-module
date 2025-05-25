import re
import cv2
import numpy as np
import onnxruntime as ort
import pytesseract
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Check if tesseract is installed
try:
    pytesseract.get_tesseract_version()
except Exception as e:
    print(f"[⚠️ ERROR] Tesseract not found: {e}")
    print("[ℹ️ INFO] Please install tesseract: sudo pacman -S tesseract tesseract-data-eng")
    exit(1)

# Load ONNX model for plate detection
try:
    session = ort.InferenceSession("./data/best.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
except Exception as e:
    print(f"[⚠️ ERROR] Failed to load ONNX model: {e}")
    exit(1)

# Initialize thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

def preprocess_image(image, size=(640, 640)):
    try:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"[⚠️ ERROR] Failed to preprocess image: {e}")
        return None

def postprocess(output, original_shape, conf_threshold=0.25, iou_threshold=0.4):
    try:
        boxes = []
        confidences = []
        img_h, img_w = original_shape[:2]
        scale_w, scale_h = img_w / 640, img_h / 640

        # Define expected aspect ratio range for license plates
        MIN_ASPECT_RATIO = 2.0  # Width should be at least 2x the height
        MAX_ASPECT_RATIO = 5.0  # Width should be at most 5x the height
        
        # Define minimum and maximum relative size of plate
        MIN_PLATE_AREA = 0.01  # Plate should be at least 1% of image
        MAX_PLATE_AREA = 0.5   # Plate should be at most 50% of image
        
        # Padding percentage for boxes (10% on each side)
        PADDING_PERCENT = 0

        rows = output.shape[1]
        for i in range(rows):
            row = output[0][i]
            confidence = row[4]
            if confidence > conf_threshold:
                x, y, w, h = row[:4]
                
                # Calculate initial box coordinates
                x1 = int((x - w / 2) * scale_w)
                y1 = int((y - h / 2) * scale_h)
                x2 = int((x + w / 2) * scale_w)
                y2 = int((y + h / 2) * scale_h)
                
                # Calculate padding
                w_padding = int(w * scale_w * PADDING_PERCENT)
                h_padding = int(h * scale_h * PADDING_PERCENT)
                
                # Apply padding
                x1 = max(0, x1 - w_padding)
                y1 = max(0, y1 - h_padding)
                x2 = min(img_w, x2 + w_padding)
                y2 = min(img_h, y2 + h_padding)
                
                width = x2 - x1
                height = y2 - y1
                
                aspect_ratio = width / height if height > 0 else 0
                relative_area = (width * height) / (img_w * img_h)
                
                if (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO and 
                    MIN_PLATE_AREA <= relative_area <= MAX_PLATE_AREA):
                    boxes.append([x1, y1, width, height])
                    confidences.append(float(confidence))
        
        if len(boxes) == 0:
            return [], []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
        
        if indices is None or len(indices) == 0:
            return [], []
        
        final_boxes = [boxes[i] for i in indices.flatten()]
        final_confidences = [confidences[i] for i in indices.flatten()]
        
        if len(final_boxes) > 1:
            sorted_indices = sorted(range(len(final_boxes)), key=lambda k: final_boxes[k][1])
            final_boxes = [final_boxes[i] for i in sorted_indices]
            final_confidences = [final_confidences[i] for i in sorted_indices]
        
        return final_boxes, final_confidences
    except Exception as e:
        print(f"[⚠️ ERROR] Failed to postprocess: {e}")
        return [], []

@lru_cache(maxsize=32)
def get_tesseract_config(psm):
    """Cache tesseract configs for better performance"""
    return f'--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

def preprocess_plate(plate_region):
    """Optimize plate image for OCR with support for both dark and light backgrounds"""
    try:
        # 1. Convert to grayscale
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        
        # 2. Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 3. Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        
        # 4. Create both normal and inverted binary images
        _, thresh_normal = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_inverse = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 5. Add white border to both
        border_size = 20
        thresh_normal = cv2.copyMakeBorder(thresh_normal, border_size, border_size, 
                                         border_size, border_size, 
                                         cv2.BORDER_CONSTANT, value=255)
        thresh_inverse = cv2.copyMakeBorder(thresh_inverse, border_size, border_size, 
                                          border_size, border_size, 
                                          cv2.BORDER_CONSTANT, value=255)
        
        # Save debug images
        cv2.imwrite("debug_plate_normal.jpg", thresh_normal)
        cv2.imwrite("debug_plate_inverse.jpg", thresh_inverse)
        
        # Return both versions for OCR
        return [thresh_normal, thresh_inverse]
    except Exception as e:
        print(f"[⚠️ ERROR] Failed to preprocess plate: {e}")
        return [plate_region]

def try_ocr_with_psm(image, psm):
    """Try OCR with specific PSM mode"""
    try:
        config = get_tesseract_config(psm)
        text = pytesseract.image_to_string(image, config=config, lang='eng').strip()
        return text if text else None
    except Exception as e:
        print(f"[⚠️ ERROR] OCR failed with PSM {psm}: {e}")
        return None

def read_license_plate(frame, box):
    try:
        x, y, w, h = map(int, box)
        plate_region = frame[y:y+h, x:x+w]
        
        # Save original plate for debugging
        cv2.imwrite("debug_plate_original.jpg", plate_region)
        
        # Calculate optimal scale based on expected character height
        target_height = 100  # Optimal height for OCR
        current_height = h
        scale = target_height / current_height
        
        # Resize plate
        width = int(w * scale)
        height = int(h * scale)
        resized = cv2.resize(plate_region, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Preprocess plate image - now returns list of processed images
        processed_images = preprocess_plate(resized)
        
        # Try different PSM modes
        psm_modes = [6, 7, 3]  # Most common PSM modes for license plates
        results = []
        
        # Try OCR on each processed image
        for processed in processed_images:
            for psm in psm_modes:
                config = f'--oem 3 --psm {psm}'
                text = pytesseract.image_to_string(processed, config=config, lang='eng').strip()
                if text:
                    results.append(text)
        
        print(f"[🔍 DEBUG] Raw OCR results: {results}")
        
        # Clean and validate results
        cleaned_results = []
        for text in results:
            # Remove any non-alphanumeric characters except spaces
            text = re.sub(r'[^A-Z0-9 ]', '', text.upper())
            
            # Try to fix common misreadings
            text = fix_common_errors(text)
            
            if text:
                cleaned_results.append(text)
        
        print(f"[🔍 DEBUG] Cleaned OCR results: {cleaned_results}")
        
        # Try to clean each result
        for text in cleaned_results:
            cleaned = clean_license_plate(text)
            if cleaned:
                return cleaned
                
        return ""
            
    except Exception as e:
        print(f"[⚠️ ERROR] Failed to read plate: {e}")
        return ""

def fix_common_errors(text):
    """Fix common OCR errors in the raw text"""
    try:
        # Split text into parts
        parts = text.split()
        if len(parts) == 1:
            # Try to split based on pattern if no spaces
            matches = re.findall(r'([A-Z0-9]{1,2})?([0-9]{1,4})?([A-Z0-9]{1,3})?', text)
            if matches:
                parts = [p for p in matches[0] if p]
        
        if not parts:
            return text
            
        # Fix first part (area code)
        if parts[0] in ['8', '0']:
            parts[0] = 'B'

        if parts[0] in ['6']:
            parts[0] = 'G'
            
        # Fix middle part (numbers)
        if len(parts) >= 2:
            # Replace common number misreadings
            parts[1] = parts[1].replace('O', '0').replace('I', '1').replace('S', '5')
            # Ensure 4 digits
            if parts[1].isdigit():
                parts[1] = parts[1].zfill(4)
            
        # Fix last part (letters)
        if len(parts) >= 3:
            last = parts[2]
            # Common patterns for TOR
            if re.match(r'^(T[O0]R|T[O0]P|TOP|TQR)$', last):
                parts[2] = 'TOR'
                
        return ' '.join(parts)
    except Exception as e:
        print(f"[⚠️ ERROR] Failed to fix common errors: {e}")
        return text

def clean_license_plate(text):
    try:
        # First try to fix common errors
        text = fix_common_errors(text)
        
        # Remove any non-alphanumeric characters except spaces
        text = re.sub(r'[^A-Z0-9 ]', '', text.upper())
        
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Pattern for Indonesian license plates
        # Format: 1-2 letters + 1-4 numbers + 0-3 letters
        pattern = re.compile(r'^[A-Z]{1,2}\s*[0-9]{1,4}(?:\s*[A-Z]{0,3})?$')
        
        if pattern.match(text):
            # Split into parts
            parts = text.split()
            
            # Handle area code (1-2 letters)
            area_code = parts[0]
            if area_code in ['8', '0']:
                area_code = 'B'
            if area_code in ['6']:
                area_code = 'G'
            
            # Handle numbers (1-4 digits)
            numbers = parts[1] if len(parts) > 1 else ''
            numbers = numbers.zfill(4)  # pad with leading zeros if needed
            
            # Handle optional suffix (0-3 letters)
            letters = parts[2] if len(parts) > 2 else ''
            
            # Build final text based on available parts
            if letters:
                final_text = f"{area_code} {numbers} {letters}"
            else:
                final_text = f"{area_code} {numbers}"
                
            return final_text
            
        print(f"[⚠️ DEBUG] Text does not match license plate pattern: '{text}'")
        return ""
            
    except Exception as e:
        print(f"[⚠️ ERROR] Failed to clean plate text: {e}")
        return ""

def scan(frame: cv2.typing.MatLike) -> str:
    try:
        print("[🔍 DEBUG] Starting plate detection...")
        
        original_shape = frame.shape
        img = preprocess_image(frame)
        if img is None:
            return ""
            
        outputs = session.run([output_name], {input_name: img})
        boxes, confidences = postprocess(outputs[0], original_shape, conf_threshold=0.2, iou_threshold=0.4)
        
        if not boxes:
            print("[ℹ️ INFO] No plates detected")
            return ""
            
        print(f"[🔍 DEBUG] Found {len(boxes)} potential plates")
        
        # Process all detected plates
        best_plate = ""
        for i, box in enumerate(boxes):
            print(f"[🔍 DEBUG] Processing plate {i+1}/{len(boxes)}")
            plate_text = read_license_plate(frame, box)
            if plate_text:  # If valid plate found
                best_plate = plate_text
                break  # Use first valid plate
                
        # Save annotated frame for debugging
        for box in boxes:
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite("debug_output.jpg", frame)
        
        return best_plate
    except Exception as e:
        print(f"[⚠️ ERROR] Error in plate scanning: {e}")
        return ""