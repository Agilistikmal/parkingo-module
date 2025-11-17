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

try:
    import tkinter as tk

    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
                "boxes": self.boxes.copy() if self.boxes else [],
                "plate": self.plate,
                "validation": self.validation,
            }


class CameraHandler:
    def __init__(
        self, camera_source, camera_name, parking_slug, slot, resolution=(1280, 720)
    ):
        self.camera_source = camera_source
        self.camera_name = camera_name
        self.parking_slug = parking_slug
        self.slot = slot
        self.resolution = resolution

        self.detection_result = DetectionResult()
        self.frame_queue = Queue(maxsize=1)
        self.cap = None
        self.running = False
        self.processing_thread = None
        self.capture_thread = None

    def initialize_camera(self):
        """Initialize the camera capture"""
        try:
            # Support both integer indices and string URLs
            if isinstance(self.camera_source, int):
                self.cap = cv2.VideoCapture(self.camera_source)
            else:
                self.cap = cv2.VideoCapture(self.camera_source)

            if not self.cap.isOpened():
                logger.error(
                    f"Error: Could not open camera {self.camera_name} (Source: {self.camera_source})"
                )
                return False

            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            logger.info(f"Camera {self.camera_name} initialized successfully")
            logger.info(f"  Parking: {self.parking_slug}, Slot: {self.slot}")
            return True
        except Exception as e:
            logger.error(f"Error initializing camera {self.camera_name}: {e}")
            return False

    def process_frames(self):
        """Background thread for processing frames"""
        while self.running:
            try:
                if self.detection_result.processing:
                    time.sleep(0.1)
                    continue

                frame = self.frame_queue.get(timeout=1.0)
                if frame is None:
                    break

                self.detection_result.processing = True

                try:
                    # Convert to RGB for plate detection
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Get plate boxes
                    outputs = plate_scanner.session.run(
                        [plate_scanner.output_name],
                        {
                            plate_scanner.input_name: plate_scanner.preprocess_image(
                                frame_rgb
                            )
                        },
                    )
                    boxes, _ = plate_scanner.postprocess(outputs[0], frame.shape)

                    if boxes:
                        # Update boxes immediately
                        self.detection_result.update(boxes=boxes)

                        # Process each box for plate number
                        for box in boxes:
                            plate_number = plate_scanner.read_license_plate(frame, box)
                            if plate_number:
                                # If plate number found, validate booking
                                validation_result = validate_booking_order(
                                    plate_number, self.parking_slug, self.slot
                                )

                                # Update results
                                self.detection_result.update(
                                    plate=plate_number, validation=validation_result
                                )
                                break

                except Exception as e:
                    logger.error(
                        f"Error in frame processing for {self.camera_name}: {e}"
                    )

                finally:
                    self.detection_result.processing = False

            except Exception:
                # Timeout or queue empty, continue
                continue

    def capture_frames(self):
        """Background thread for capturing frames"""
        frame_count = 0
        process_interval = 10  # Process every 10 frames

        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error(f"Error: Could not read frame from {self.camera_name}")
                    time.sleep(0.1)
                    continue

                # Store frame for display (latest frame)
                self.latest_frame = frame.copy()

                # Queue frame for processing at intervals
                frame_count += 1
                if frame_count % process_interval == 0:
                    try:
                        # Remove old frame if queue is full
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except Exception:
                                pass
                        self.frame_queue.put_nowait(frame.copy())
                    except Exception:
                        pass

                time.sleep(0.03)  # ~30 FPS

            except Exception as e:
                logger.error(f"Error capturing frames from {self.camera_name}: {e}")
                time.sleep(0.1)

    def start(self):
        """Start camera capture and processing"""
        if not self.initialize_camera():
            return False

        self.running = True
        self.latest_frame = None

        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self.process_frames, daemon=True
        )
        self.processing_thread.start()

        # Start capture thread
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()

        return True

    def stop(self):
        """Stop camera capture and processing"""
        self.running = False

        # Stop processing thread
        if self.frame_queue:
            self.frame_queue.put(None)

        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)

        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)

        # Release camera
        if self.cap:
            self.cap.release()

        logger.info(f"Camera {self.camera_name} stopped")

    def get_display_frame(self):
        """Get the latest frame with annotations"""
        if self.latest_frame is None:
            return None

        display_frame = self.latest_frame.copy()

        # Get current detection results
        current_results = self.detection_result.get_data()

        # Draw boxes and information on frame
        if current_results["boxes"]:
            for box in current_results["boxes"]:
                draw_plate_box(
                    display_frame,
                    box,
                    current_results["plate"],
                    current_results["validation"],
                )

        # Draw the original information display (top-left corner)
        if current_results["plate"]:
            draw_text_with_background(
                display_frame, f"Plate: {current_results['plate']}", (10, 30)
            )

            if current_results["validation"]:
                try:
                    if (
                        current_results["validation"]
                        .get("data", {})
                        .get("is_valid", False)
                    ):
                        status = "VALID BOOKING"
                        color = (0, 255, 0)  # Green
                    else:
                        status = "INVALID BOOKING"
                        color = (0, 0, 255)  # Red

                    draw_text_with_background(
                        display_frame, status, (10, 70), color=color
                    )
                except Exception:
                    draw_text_with_background(
                        display_frame, "Validation Error", (10, 70)
                    )

        # Draw camera name
        draw_text_with_background(
            display_frame, f"Camera: {self.camera_name}", (10, 20), color=(255, 255, 0)
        )

        # Draw parking info
        draw_text_with_background(
            display_frame,
            f"Parking: {self.parking_slug} | Slot: {self.slot}",
            (10, display_frame.shape[0] - 20),
        )

        # Show processing indicator if active
        if self.detection_result.processing:
            draw_text_with_background(
                display_frame,
                "Processing...",
                (display_frame.shape[1] - 150, 30),
                color=(0, 255, 255),
            )

        return display_frame


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
            "slot": slot,
        }

        # Make API request to validation endpoint
        response = requests.post(
            endpoint, json=payload, headers={"Content-Type": "application/json"}
        )

        if response.status_code == 404:
            return None

        # Check if request was successful
        result = response.json()
        return result

    except Exception as e:
        logger.exception(f"Error during booking validation: {e}")
        return None


def draw_text_with_background(
    img, text, position, scale=0.7, thickness=2, color=(255, 255, 255)
):
    """Helper function to draw text with background on the image"""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)

    # Calculate background rectangle position
    x, y = position
    padding = 5

    # Draw background rectangle
    cv2.rectangle(
        img,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + padding),
        (0, 0, 0),
        -1,
    )

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
            status_text = f"VALID ({validation_status.get('data', {}).get('similarity', 0) * 100}%)"
        else:
            status_color = (0, 0, 255)  # Red for invalid
            status_text = f"INVALID ({validation_status.get('data', {}).get('similarity', 0) * 100}%)"

        status_y = text_y + 25
        draw_text_with_background(img, status_text, (x, status_y), color=status_color)
    else:
        status_y = text_y + 25
        draw_text_with_background(
            img, "AVAILABLE (Belum ada pemesanan)", (x, status_y), color=(0, 255, 255)
        )


def get_screen_size():
    """Get screen resolution"""
    try:
        if HAS_TKINTER:
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            return screen_width, screen_height
        else:
            logger.warning("Tkinter not available, using default 1920x1080")
            return 1920, 1080
    except Exception as e:
        logger.warning(f"Could not get screen size: {e}, using default 1920x1080")
        return 1920, 1080


def create_grid_layout(frames, grid_cols=2, max_screen_usage=1):
    if not frames:
        return None

    valid_frames = [f for f in frames if f is not None]
    if not valid_frames:
        return None

    orig_h, orig_w = valid_frames[0].shape[:2]

    num_frames = len(valid_frames)
    grid_rows = (num_frames + grid_cols - 1) // grid_cols

    screen_width, screen_height = get_screen_size()

    # Account for window borders and taskbar (use max_screen_usage of screen)
    available_width = int(screen_width * max_screen_usage)
    available_height = int(screen_height * max_screen_usage)

    # Calculate desired grid dimensions
    desired_grid_w = orig_w * grid_cols
    desired_grid_h = orig_h * grid_rows

    # Calculate scale factors to fit screen
    scale_w = available_width / desired_grid_w
    scale_h = available_height / desired_grid_h

    # Use the smaller scale to ensure both dimensions fit
    scale = min(scale_w, scale_h)

    # If grid is smaller than screen, don't scale up
    if scale > 1.0:
        scale = 1.0

    # Calculate scaled dimensions
    scaled_w = int(orig_w * scale)
    scaled_h = int(orig_h * scale)
    grid_w = scaled_w * grid_cols
    grid_h = scaled_h * grid_rows

    # Create grid canvas
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # Place frames in grid (resize if needed)
    for idx, frame in enumerate(valid_frames):
        row = idx // grid_cols
        col = idx % grid_cols

        # Resize frame if scale is not 1.0
        if scale != 1.0:
            resized_frame = cv2.resize(
                frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
            )
        else:
            resized_frame = frame

        y_start = row * scaled_h
        x_start = col * scaled_w
        grid[y_start : y_start + scaled_h, x_start : x_start + scaled_w] = resized_frame

    return grid


def main():
    load_dotenv()

    configs = json.load(open("config.json"))
    camera_configs = configs["cameras"]

    camera_handlers = []
    for config in camera_configs:
        handler = CameraHandler(
            camera_source=config.get("source", 0),
            camera_name=config.get("name", f"Camera {len(camera_handlers) + 1}"),
            parking_slug=config.get("parking_slug", "default-parking"),
            slot=config.get("slot", "A1"),
        )
        if handler.start():
            camera_handlers.append(handler)
        else:
            logger.warning(f"Failed to start {handler.camera_name}, skipping...")

    if not camera_handlers:
        logger.error("No cameras could be initialized. Exiting.")
        return

    logger.info(f"Started {len(camera_handlers)} camera(s)")

    # Display mode: 'grid' or 'separate'
    display_mode = configs.get("display_mode", "grid")
    grid_cols = configs.get("grid_cols", 2)
    max_screen_usage = configs.get("max_screen_usage", 0.95)

    # Create named window for grid mode (allows resizing)
    if display_mode == "grid":
        cv2.namedWindow("Multi-Camera License Plate Scanner", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Get frames from all cameras
            frames = []
            for handler in camera_handlers:
                frame = handler.get_display_frame()
                frames.append(frame)

            # Display frames
            if display_mode == "grid":
                # Display as grid
                grid_frame = create_grid_layout(
                    frames, grid_cols=grid_cols, max_screen_usage=max_screen_usage
                )
                if grid_frame is not None:
                    # Set window size to match grid size
                    h, w = grid_frame.shape[:2]
                    cv2.resizeWindow("Multi-Camera License Plate Scanner", w, h)
                    cv2.imshow("Multi-Camera License Plate Scanner", grid_frame)
            else:
                # Display in separate windows
                for idx, frame in enumerate(frames):
                    if frame is not None:
                        cv2.imshow(f"Camera {camera_handlers[idx].camera_name}", frame)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error in main loop: {e}")

    finally:
        # Stop all cameras
        for handler in camera_handlers:
            handler.stop()

        cv2.destroyAllWindows()
        logger.info("All cameras released and windows closed")


if __name__ == "__main__":
    main()
