import os
import plate_scanner
import numpy as np
import cv2
from PIL import Image
import io
import json
import base64
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# MQTT broker configuration
broker_address = "173.234.15.83"
broker_port = 1883

def on_message(client, userdata, message: mqtt.MQTTMessage):
    print(f"[📩 Received] Topic: {message.topic}")

    try:
        # Decode JSON payload dengan error handling yang lebih baik
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except UnicodeDecodeError:
            # Jika UTF-8 gagal, coba dengan latin-1
            payload = json.loads(message.payload.decode("latin-1"))
            
        x_api_key = payload.get("X-API-KEY")
        x_mac_address = payload.get("X-MAC-ADDRESS")
        image_base64 = payload.get("image")
    except (json.JSONDecodeError, Exception) as e:
        print(f"[⚠️ ERROR] Gagal decode payload: {str(e)}")
        return

    api_key = os.environ.get("API_KEY")

    if not x_api_key or not x_mac_address:
        print(f"[⚠️ ERROR] Required header X-API-KEY and X-MAC-ADDRESS")
        response = json.dumps({
            "mac_address": x_mac_address,
            "error": "Required header X-API-KEY and X-MAC-ADDRESS",
            "data": None
        })
        client.publish("parkingo/response", response)
        return

    if api_key != x_api_key:
        print(f"[⚠️ ERROR] Invalid X-API-KEY: {x_mac_address} {x_api_key}")
        response = json.dumps({
            "mac_address": x_mac_address,
            "error": "Invalid X-API-KEY",
            "data": None
        })
        client.publish("parkingo/response", response)
        return

    try:
        # Decode gambar dari Base64
        image_bytes = base64.b64decode(image_base64)
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Debug: Simpan gambar untuk dicek
        cv2.imwrite("debug_input.jpg", frame)
        print(f"[🔍 DEBUG] Image shape: {frame.shape}")
        print(f"[🔍 DEBUG] Image type: {frame.dtype}")
        
    except Exception as e:
        print(f"[⚠️ ERROR] Gagal mendecode gambar: {e}")
        return

    # Scan plat nomor
    try:
        print("[🔍 Starting plate scan...]")
        plate_number = plate_scanner.scan(frame=frame)
        print(f"[🔍 DEBUG] Raw plate result: {plate_number}")
    except Exception as e:
        print(f"[⚠️ ERROR] Error during plate scanning: {e}")
        plate_number = None

    print(f"[📩 Response] Plate: {plate_number}")
    if plate_number is not None:
        # TODO: Check if plate_number is valid booking order API
        pass

    # Kirim response dengan MAC Address
    response = json.dumps({
        "mac_address": x_mac_address,
        "error": None,
        "data": plate_number
    })
    client.publish("parkingo/response", response)

def connect_to_broker(client, userdata, flags, reason_code, properties):
    print("[✅ Connected to MQTT Broker]")
    client.subscribe("parkingo/scanner")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Inisialisasi client MQTT dengan Paho v2.x
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    # Set callback
    client.on_connect = connect_to_broker
    client.on_message = on_message

    # Connect ke MQTT broker
    client.connect(broker_address, broker_port)

    # Start MQTT loop
    client.loop_forever()
