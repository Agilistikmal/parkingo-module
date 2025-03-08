import os
import plate_scanner
import numpy as np
import cv2
from PIL import Image
import io
import json
import base64
import paho.mqtt.client as mqtt

# MQTT broker configuration
broker_address = "localhost"  # Jika dalam Docker, bisa gunakan "mosquitto"
broker_port = 1883

def on_message(client, userdata, message):
    print(f"[📩 Received] Topic: {message.topic}")

    try:
        # Decode JSON payload
        payload = json.loads(message.payload.decode("utf-8"))
        x_api_key = payload.get("X-API-KEY")
        x_mac_address = payload.get("X-MAC-ADDRESS")
        image_base64 = payload.get("image")
    except json.JSONDecodeError:
        print("[⚠️ ERROR] Payload bukan JSON")
        return

    api_key = os.environ.get("API_KEY")

    if not x_api_key or not x_mac_address:
        response = json.dumps({
            "mac_address": x_mac_address,
            "error": "Required header X-API-KEY and X-MAC-ADDRESS",
            "data": None
        })
        client.publish("parkingo/response", response)
        return

    if api_key != x_api_key:
        response = json.dumps({
            "mac_address": x_mac_address,
            "error": "Invalid X-API-KEY",
            "data": None
        })
        client.publish("parkingo/response", response)
        return

    # Decode gambar dari Base64
    try:
        image_bytes = base64.b64decode(image_base64)
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[⚠️ ERROR] Gagal mendecode gambar: {e}")
        return

    # Scan plat nomor
    plate_number = plate_scanner.scan(frame=frame)

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

# Inisialisasi client MQTT dengan Paho v2.x
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# Set callback
client.on_connect = connect_to_broker
client.on_message = on_message

# Connect ke MQTT broker
client.connect(broker_address, broker_port)

# Start MQTT loop
client.loop_forever()
