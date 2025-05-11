import os
import plate_scanner
import numpy as np
import cv2
import json
import base64
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from dotenv import load_dotenv
import logging
import requests
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'parkingo-websocket-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Validate booking order
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
        api_base_url = os.environ.get("VALIDATION_API_URL")
        if not api_base_url:
            logger.error("VALIDATION_API_URL not set in environment variables")
            return False
            
        endpoint = f"{api_base_url}/v1/bookings/validate"
        
        # Create payload
        payload = {
            "plate_number": plate_number,
            "parking_slug": parking_slug,
            "slot": slot
        }
        
        logger.info(f"Validating booking order: {payload}")
        logger.info(f"Validation endpoint: {endpoint}")
        
        # Make API request to validation endpoint
        response = requests.post(
            endpoint, 
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            is_valid = result.get("is_valid", False)
            logger.info(f"Booking validation result: {is_valid}")
            return is_valid
        else:
            logger.error(f"Error validating booking: Status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.exception(f"Error during booking validation: {e}")
        return False

# Simple API endpoint that describes WebSocket usage
@app.route('/ws')
def index():
    return jsonify({
        "service": "ParkingGo Scanner WebSocket API",
        "description": "WebSocket API for realtime scanner feed from ESP32-CAM",
        "websocket_endpoint": "ws://" + request.host,
        "events": {
            "scanner_image": {
                "description": "Emitted when a new image is received from ESP32-CAM",
                "data": {
                    "image": "Base64 encoded JPEG image",
                    "mac_address": "ESP32-CAM MAC address",
                    "parking_slug": "Parking location identifier",
                    "slot": "Parking slot identifier",
                    "plate_number": "Detected license plate or null if none detected",
                    "is_valid_booking": "Boolean indicating if booking is valid"
                }
            }
        },
        "usage": "Connect to the WebSocket endpoint using Socket.IO client and listen for 'scanner_image' events"
    })

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@app.route('/scanner', methods=['POST'])
def scanner_endpoint():
    logger.info("[📩 Received HTTP request]")
    
    # Get headers with metadata
    x_api_key = request.headers.get('X-API-KEY')
    x_mac_address = request.headers.get('X-MAC-ADDRESS')
    x_parking_slug = request.headers.get('X-PARKING-SLUG')
    x_slot = request.headers.get('X-SLOT')
    
    # Check if we received binary image data
    image_data = request.data
    content_type = request.headers.get('Content-Type', '')
    
    # Log request information
    logger.info(f"MAC Address: {x_mac_address}")
    logger.info(f"Parking: {x_parking_slug}")
    logger.info(f"Slot: {x_slot}")
    logger.info(f"Content-Type: {content_type}")
    logger.info(f"Data size: {len(image_data)} bytes")
    
    # Check API key
    api_key = os.environ.get("API_KEY")

    if not x_api_key or not x_mac_address:
        logger.error("[⚠️ ERROR] Required headers X-API-KEY and X-MAC-ADDRESS")
        return jsonify({
            "mac_address": x_mac_address,
            "error": "Required headers X-API-KEY and X-MAC-ADDRESS",
            "data": None
        }), 400

    if api_key != x_api_key:
        logger.error(f"[⚠️ ERROR] Invalid X-API-KEY: {x_mac_address} {x_api_key}")
        return jsonify({
            "mac_address": x_mac_address,
            "error": "Invalid X-API-KEY",
            "data": None
        }), 401

    try:
        # Decode image based on content type
        if 'image/jpeg' in content_type:
            # Direct binary image data
            frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            # Assume it's JSON with base64 data
            try:
                data = json.loads(image_data)
                image_base64 = data.get("image")
                if image_base64:
                    image_bytes = base64.b64decode(image_base64)
                    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                else:
                    raise ValueError("No image data found in JSON")
            except json.JSONDecodeError:
                # If not JSON, try to decode as raw JPEG
                frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Debug: Save image for checking
        cv2.imwrite("debug_input.jpg", frame)
        logger.info(f"[🔍 DEBUG] Image shape: {frame.shape}")
        logger.info(f"[🔍 DEBUG] Image type: {frame.dtype}")
        
        # Convert image to base64 for WebSocket
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
    except Exception as e:
        logger.exception(f"[⚠️ ERROR] Failed to decode image: {e}")
        return jsonify({
            "mac_address": x_mac_address,
            "error": f"Failed to decode image: {str(e)}",
            "data": None
        }), 400

    # Scan license plate
    try:
        logger.info("[🔍 Starting plate scan...]")
        plate_number = plate_scanner.scan(frame=frame)
        logger.info(f"[🔍 DEBUG] Raw plate result: {plate_number}")
    except Exception as e:
        logger.exception(f"[⚠️ ERROR] Error during plate scanning: {e}")
        plate_number = None

    logger.info(f"[📩 Response] Plate: {plate_number}")
    
    # Validate booking order
    is_valid_booking = validate_booking_order(plate_number, x_parking_slug, x_slot)
    
    # Create WebSocket payload
    payload = {
        'image': image_base64,
        'mac_address': x_mac_address,
        'parking_slug': x_parking_slug,
        'slot': x_slot,
        'plate_number': plate_number,
        'is_valid_booking': is_valid_booking,
        'timestamp': time.time()
    }
    
    # Emit the image and results through WebSocket
    try:
        connected_clients = len(socketio.server.eio.sockets)
        logger.info(f"Broadcasting scanner data to {connected_clients} connected clients")
        socketio.emit('scanner_image', payload)
    except Exception as e:
        logger.exception(f"Error broadcasting to WebSocket: {e}")

    # Return response with MAC Address
    return jsonify({
        "mac_address": x_mac_address,
        "parking_slug": x_parking_slug,
        "slot": x_slot,
        "error": None,
        "data": {
            "plate_number": plate_number,
            "is_valid_booking_order": is_valid_booking
        }
    })

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Get the port from environment or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Run the Flask app with SocketIO
    logger.info(f"[🚀 Starting WebSocket server on port {port}]")
    logger.info(f"WebSocket endpoint: ws://0.0.0.0:{port}/")
    logger.info(f"Use a Socket.IO client to connect and listen for 'scanner_image' events")
    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)
