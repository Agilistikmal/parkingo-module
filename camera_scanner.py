import cv2
import plate_scanner
import numpy as np
import os
import requests
import json
import time
from dotenv import load_dotenv
import logging
import threading
from queue import Queue
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add global variables for tracking boxes
class DetectionResult:
    def __init__(self):
        self.boxes = []
        self.plate = None
        self.validation = None
        self.processing = False
        self.lock = threading.Lock()

    def update(self, boxes=None, plate=None, validation=None):
        with self.lock:
            if boxes is not None:
                self.boxes = boxes
            if plate is not None:
                self.plate = plate
            if validation is not None:
                self.validation = validation

    def get_data(self):
        with self.lock:
            return {
                'boxes': self.boxes.copy() if self.boxes else [],
                'plate': self.plate,
                'validation': self.validation
            }

# Global detection result
detection_result = DetectionResult()

# Queue for frames to process
frame_queue = Queue(maxsize=1)

def process_frames():
    """Background thread for processing frames"""
    while True:
        try:
            if detection_result.processing:
                time.sleep(0.1)  # Don't process if still processing previous frame
                continue

            frame = frame_queue.get()
            if frame is None:  # Sentinel value to stop the thread
                break

            detection_result.processing = True
            
            # Process frame
            try:
                # Convert to RGB for plate detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get plate boxes
                outputs = plate_scanner.session.run([plate_scanner.output_name], 
                                                  {plate_scanner.input_name: plate_scanner.preprocess_image(frame_rgb)})
                boxes, _ = plate_scanner.postprocess(outputs[0], frame.shape)
                
                if boxes:
                    # Update boxes immediately
                    detection_result.update(boxes=boxes)
                    
                    # Process each box for plate number
                    for box in boxes:
                        plate_number = plate_scanner.read_license_plate(frame, box)
                        if plate_number:
                            # If plate number found, validate booking
                            validation_result = validate_booking_order(
                                plate_number,
                                os.environ.get("PARKING_SLUG", "default-parking"),
                                os.environ.get("PARKING_SLOT", "A1")
                            )
                            
                            # Update results
                            detection_result.update(
                                plate=plate_number,
                                validation=validation_result
                            )
                            break
                
            except Exception as e:
                logger.error(f"Error in frame processing: {e}")
            
            finally:
                detection_result.processing = False
                
        except Exception as e:
            logger.error(f"Error in process_frames: {e}")
            detection_result.processing = False

def validate_booking_order(plate_number, parking_slug, slot):
    """
    Validate if the plate number has a valid booking order for the given parking slug and slot
    using the external validation API
    """
    if not plate_number:
        logger.warning("Cannot validate booking: No plate number detected")
        return False
        
    try:
        # Get API endpoint from .env file
        api_base_url = os.environ.get("API_BASE_URL")
        if not api_base_url:
            logger.error("API_BASE_URL not set in environment variables")
            return False
            
        endpoint = f"{api_base_url}/v1/bookings/validate"
        
        # Create payload
        payload = {
            "plate_number": plate_number,
            "parking_slug": parking_slug,
            "slot": slot
        }
        
        # Make API request to validation endpoint
        response = requests.post(
            endpoint, 
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Check if request was successful
        result = response.json()
        return result
            
    except Exception as e:
        logger.exception(f"Error during booking validation: {e}")
        return None

def draw_text_with_background(img, text, position, scale=0.7, thickness=2, color=(255, 255, 255)):
    """Helper function to draw text with background on the image"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    
    # Calculate background rectangle position
    x, y = position
    padding = 5
    
    # Draw background rectangle
    cv2.rectangle(img, 
                 (x - padding, y - text_height - padding),
                 (x + text_width + padding, y + padding),
                 (0, 0, 0),
                 -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, scale, color, thickness)

def draw_plate_box(img, box, plate_text, validation_status=None):
    """Draw box around license plate with plate number above it"""
    x, y, w, h = box
    
    # Draw the box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw plate number above the box
    text_y = y - 10 if y - 10 > 10 else y + h + 20
    draw_text_with_background(img, f"Plate: {plate_text}", (x, text_y))
    
    # If validation status exists, draw it below the plate number
    if validation_status:
        if validation_status.get("data", {}).get("is_valid", False):
            status_color = (0, 255, 0)  # Green for valid
            status_text = f"VALID ({validation_status.get('data', {}).get('similarity', 0)}%)"
        else:
            status_color = (0, 0, 255)  # Red for invalid
            status_text = f"INVALID ({validation_status.get('data', {}).get('similarity', 0)}%)"
            
        status_y = text_y + 25
        draw_text_with_background(img, status_text, (x, status_y), color=status_color)

def main():
    # Load environment variables
    load_dotenv()
    
    # Get parking details from environment
    parking_slug = os.environ.get("PARKING_SLUG", "default-parking")
    slot = os.environ.get("PARKING_SLOT", "A1")
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    if not cap.isOpened():
        logger.error("Error: Could not open camera")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    logger.info("Camera initialized successfully")
    logger.info(f"Parking: {parking_slug}")
    logger.info(f"Slot: {slot}")
    
    # Start processing thread
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    processing_thread.start()
    
    # Frame counter for processing
    frame_count = 0
    process_interval = 10  # Process every 10 frames
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Error: Could not read frame")
                break
            
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Only queue frame for processing at intervals
            frame_count += 1
            if frame_count % process_interval == 0:
                # Update frame queue (non-blocking)
                try:
                    # Remove old frame if queue is full
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except:
                            pass
                    frame_queue.put_nowait(frame.copy())
                except:
                    pass
            
            # Get current detection results
            current_results = detection_result.get_data()
            
            # Draw boxes and information on frame
            if current_results['boxes']:
                for box in current_results['boxes']:
                    draw_plate_box(display_frame, box, 
                                 current_results['plate'], 
                                 current_results['validation'])
            
            # Draw the original information display (top-left corner)
            if current_results['plate']:
                draw_text_with_background(display_frame, 
                                       f"Plate: {current_results['plate']}",
                                       (10, 30))
                
                if current_results['validation']:
                    try:
                        if current_results['validation'].get("data", {}).get("is_valid", False):
                            status = "VALID BOOKING"
                            color = (0, 255, 0)  # Green
                        else:
                            status = "INVALID BOOKING"
                            color = (0, 0, 255)  # Red
                            
                        draw_text_with_background(display_frame, status, (10, 70), color=color)
                    except:
                        draw_text_with_background(display_frame,
                                               "Validation Error",
                                               (10, 70))
            
            # Draw parking info
            draw_text_with_background(display_frame,
                                   f"Parking: {parking_slug} | Slot: {slot}",
                                   (10, display_frame.shape[0] - 20))
            
            # Show processing indicator if active
            if detection_result.processing:
                draw_text_with_background(display_frame,
                                       "Processing...",
                                       (display_frame.shape[1] - 150, 30),
                                       color=(0, 255, 255))
            
            # Show the frame
            cv2.imshow('License Plate Scanner', display_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
    
    finally:
        # Stop processing thread
        frame_queue.put(None)
        processing_thread.join(timeout=1.0)
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released and windows closed")

if __name__ == "__main__":
    main()
