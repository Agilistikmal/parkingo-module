import re
import cv2
import numpy as np
import onnxruntime as ort
import pytesseract

session = ort.InferenceSession("./data/best.onnx", providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess_image(image, size=(640, 640)):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(output, original_shape, conf_threshold=0.5, iou_threshold=0.4):
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
        return [], []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
    
    if indices is None or len(indices) == 0:
        return [], []
    
    final_boxes = [boxes[i] for i in indices.flatten()]
    final_confidences = [confidences[i] for i in indices.flatten()]
    return final_boxes, final_confidences

def read_license_plate(frame, box):
    x, y, w, h = map(int, box)
    x1, y1, x2, y2 = x, y, x + w, y + h
    
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    
    plate_region = frame[y1:y2, x1:x2]
    
    
    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)  
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
    
    plate_text = pytesseract.image_to_string(thresh, config='--psm 6')  
    return clean_license_plate(plate_text)

def clean_license_plate(text):
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9 ]', '', text)
    
    pattern = re.compile(r'^[A-Z]{1,2} \d{1,4}(?: [A-Z]{1,3})?$')
    match = pattern.match(text)
    return match.group(0) if match else ""

# Scanner
def scan(frame: cv2.typing.MatLike) -> str:

    original_shape = frame.shape
    img = preprocess_image(frame)
    
    outputs = session.run([output_name], {input_name: img})
    
    boxes, confidences = postprocess(outputs[0], original_shape)

    for box in boxes:
        x, y, w, h = map(int, box)
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  
        plate_text = read_license_plate(frame, box)
        cv2.putText(frame, f"{plate_text}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  
        return plate_text