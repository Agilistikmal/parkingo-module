import re
import cv2
import numpy as np
import onnxruntime as ort
import pytesseract
import os

# Check if tesseract is installed
try:
    pytesseract.get_tesseract_version()
except Exception as e:
    print(f"[⚠️ ERROR] Tesseract not found: {e}")
    print("[ℹ️ INFO] Please install tesseract: sudo pacman -S tesseract tesseract-data-eng")
    exit(1)

# Load ONNX model
try:
    session = ort.InferenceSession("./data/best.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
except Exception as e:
    print(f"[⚠️ ERROR] Failed to load ONNX model: {e}")
    exit(1)

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

        rows = output.shape[1]
        for i in range(rows):
            row = output[0][i]
            confidence = row[4]  
            if confidence > conf_threshold:
                x, y, w, h = row[:4]  
                x1, y1, x2, y2 = int((x - w / 2) * scale_w), int((y - h / 2) * scale_h), int((x + w / 2) * scale_w), int((y + h / 2) * scale_h)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                
                boxes.append([x1, y1, x2 - x1, y2 - y1])  
                confidences.append(float(confidence))
        
        if len(boxes) == 0:
            print("[ℹ️ INFO] No license plates detected")
            return [], []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
        
        if indices is None or len(indices) == 0:
            print("[ℹ️ INFO] No license plates after NMS")
            return [], []
        
        final_boxes = [boxes[i] for i in indices.flatten()]
        final_confidences = [confidences[i] for i in indices.flatten()]
        return final_boxes, final_confidences
    except Exception as e:
        print(f"[⚠️ ERROR] Failed to postprocess: {e}")
        return [], []

def read_license_plate(frame, box):
    try:
        x, y, w, h = map(int, box)
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        # Tambah padding
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        plate_region = frame[y1:y2, x1:x2]
        
        # Save original plate region for debugging
        cv2.imwrite("debug_plate_original.jpg", plate_region)
        
        # Convert ke grayscale
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Invert kembali untuk OCR
        thresh = cv2.bitwise_not(thresh)
        
        # Save preprocessed plate for debugging
        cv2.imwrite("debug_plate_thresh.jpg", thresh)
        
        # Resize gambar 2x lebih besar untuk OCR
        h, w = thresh.shape
        thresh = cv2.resize(thresh, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        
        # Save resized plate for debugging
        cv2.imwrite("debug_plate_resized.jpg", thresh)
        
        # Configure tesseract for better plate recognition
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        plate_text = pytesseract.image_to_string(thresh, config=custom_config)
        
        cleaned_text = clean_license_plate(plate_text)
        print(f"[🔍 DEBUG] Raw text: {plate_text.strip()}")
        print(f"[🔍 DEBUG] Cleaned text: {cleaned_text}")
        
        # Jika tidak ada hasil, coba dengan gambar asli
        if not cleaned_text:
            print("[🔍 DEBUG] Trying with original image...")
            plate_text = pytesseract.image_to_string(plate_region, config=custom_config)
            cleaned_text = clean_license_plate(plate_text)
            print(f"[🔍 DEBUG] Raw text (original): {plate_text.strip()}")
            print(f"[🔍 DEBUG] Cleaned text (original): {cleaned_text}")
        
        return cleaned_text
    except Exception as e:
        print(f"[⚠️ ERROR] Failed to read plate: {e}")
        return ""

def clean_license_plate(text):
    try:
        text = text.upper().strip()
        text = re.sub(r'[^A-Z0-9 ]', '', text)
        
        pattern = re.compile(r'^[A-Z]{1,2} \d{1,4}(?: [A-Z]{1,3})?$')
        match = pattern.match(text)
        return match.group(0) if match else ""
    except Exception as e:
        print(f"[⚠️ ERROR] Failed to clean plate text: {e}")
        return ""

# Scanner
def scan(frame: cv2.typing.MatLike) -> str:
    try:
        print("[🔍 DEBUG] Starting plate detection...")
        
        # Save input frame for debugging
        cv2.imwrite("debug_input.jpg", frame)
        
        original_shape = frame.shape
        img = preprocess_image(frame)
        if img is None:
            return ""
            
        outputs = session.run([output_name], {input_name: img})
        boxes, confidences = postprocess(outputs[0], original_shape)
        
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