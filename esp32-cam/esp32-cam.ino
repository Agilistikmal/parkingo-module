#include <WiFi.h>
#include <WiFiClient.h>
#include <PubSubClient.h>
#include "esp_camera.h"
#include "base64.h"  // Library Base64 encoding

// 🔥 Konfigurasi WiFi & MQTT
#define WIFI_SSID "ST"
#define WIFI_PASSWORD "1234567890"
#define MQTT_BROKER "173.234.15.83"  // Ganti dengan IP broker MQTT
#define MQTT_PORT 1883
#define TOPIC_REQUEST "parkingo/scanner"
#define TOPIC_RESPONSE "parkingo/response"

// 🔑 API Key Static
#define API_KEY "your_api_key_here"

WiFiClient espClient;
PubSubClient client(espClient);
String mac_address;

// 🔥 Fungsi untuk inisialisasi kamera ESP32-CAM
void initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    if (psramFound()) {
        config.frame_size = FRAMESIZE_VGA;
        config.jpeg_quality = 10;
        config.fb_count = 2;
    } else {
        config.frame_size = FRAMESIZE_CIF;
        config.jpeg_quality = 12;
        config.fb_count = 1;
    }

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("[⚠️ ERROR] Kamera gagal diinisialisasi!");
        while (true);
    }
}

// 🔥 Callback untuk menerima response dari MQTT
void callback(char* topic, byte* payload, unsigned int length) {
    Serial.print("\n[📩 MQTT Response] Topic: ");
    Serial.println(topic);

    String message;
    for (unsigned int i = 0; i < length; i++) {
        message += (char)payload[i];
    }

    Serial.println("Payload: " + message);

    // Cek apakah response mengandung MAC Address ESP32-CAM
    if (message.indexOf(mac_address) != -1) {
        Serial.println("[✅ Response cocok dengan MAC Address]");
    } else {
        Serial.println("[⚠️ Response bukan untuk perangkat ini, abaikan]");
    }
}

// 🔥 Fungsi untuk mengambil gambar & mengirim ke MQTT
void captureAndSend() {
    Serial.println("\n[📸 Capturing image...]");

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("[⚠️ ERROR] Gagal mengambil gambar!");
        return;
    }

    // Konversi gambar ke Base64
    String image_base64 = base64::encode(fb->buf, fb->len);
    esp_camera_fb_return(fb);  // Lepaskan buffer kamera

    // Buat payload JSON
    String payload = "{";
    payload += "\"X-API-KEY\": \"" + String(API_KEY) + "\",";
    payload += "\"X-MAC-ADDRESS\": \"" + mac_address + "\",";
    payload += "\"image\": \"" + image_base64 + "\"";
    payload += "}";

    Serial.println("[🚀 SENDING IMAGE...]");
    client.publish(TOPIC_REQUEST, payload.c_str());
}

// 🔥 Fungsi untuk koneksi MQTT
void reconnect() {
    while (!client.connected()) {
        Serial.print("[🔄 Connecting to MQTT...] ");
        if (client.connect("ESP32Client")) {
            Serial.println("[✅ Connected]");
            client.subscribe(TOPIC_RESPONSE);
        } else {
            Serial.print("[⚠️ Failed, rc=");
            Serial.print(client.state());
            Serial.println("] retrying...");
            delay(2000);
        }
    }
}

void setup() {
    Serial.begin(115200);

    // Koneksi ke WiFi
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\n[✅ WiFi Connected]");

    // Dapatkan MAC Address ESP32
    mac_address = WiFi.macAddress();
    Serial.print("ESP32 MAC Address: ");
    Serial.println(mac_address);

    // Inisialisasi kamera
    initCamera();

    // Koneksi MQTT
    client.setServer(MQTT_BROKER, MQTT_PORT);
    client.setCallback(callback);
}

void loop() {
    if (!client.connected()) {
        reconnect();
    }
    client.loop();

    // Kirim gambar setiap 5 detik
    captureAndSend();
    delay(5000);
}
