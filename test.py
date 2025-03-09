import paho.mqtt.client as mqtt
import json
import time
import base64
import dotenv
import os

# MQTT Configuration
broker_address = "173.234.15.83"
broker_port = 1883
topic_request = "parkingo/scanner"
topic_response = "parkingo/response"

dotenv.load_dotenv()

# Informasi MAC Address (disesuaikan)
mac_address = "00:1A:2B:3C:4D:5E"
api_key = os.getenv("API_KEY")

# Load dan encode gambar ke Base64
image_path = "./data/plate/test1.webp"  # Ubah ke path gambar yang ingin dikirim
with open(image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Variabel untuk menyimpan respons
response_received = None

# Callback ketika ada pesan masuk
def on_message(client, userdata, message):
    global response_received
    response_text = message.payload.decode("utf-8")

    try:
        response_json = json.loads(response_text)

        # 🔥 Filter response hanya untuk MAC ini
        if response_json.get("mac_address") == mac_address:
            print("\n[📩 RESPONSE RECEIVED]")
            print(json.dumps(response_json, indent=4))
            response_received = response_json

    except json.JSONDecodeError:
        print("Invalid JSON response:", response_text)

# Setup MQTT Client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_message = on_message

# Connect ke broker
client.connect(broker_address, broker_port)

# Subscribe ke response
client.subscribe(topic_response)

# Start MQTT Loop
client.loop_start()

# Kirim data dengan MAC Address
payload = json.dumps({
    "X-API-KEY": api_key,
    "X-MAC-ADDRESS": mac_address,
    "image": image_base64
})

print("\n[🚀 SENDING IMAGE...]")
client.publish(topic_request, payload)

# Tunggu response maksimal 10 detik
timeout = 10
start_time = time.time()

while response_received is None and (time.time() - start_time) < timeout:
    time.sleep(0.5)

client.loop_stop()

if response_received is None:
    print("\n[⚠️ No response received within timeout!]")

print("\n[✅ TEST COMPLETED]")
