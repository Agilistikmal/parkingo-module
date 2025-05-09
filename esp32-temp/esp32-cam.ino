#include <WiFi.h>
#include <WiFiClient.h>
#include <PubSubClient.h>
#include "esp_camera.h"
#include "base64.h"  // Library Base64 encoding

// 🔥 Konfigurasi WiFi & MQTT
#define WIFI_SSID "AGL 1"
#define WIFI_PASSWORD "1234567890"
#define MQTT_BROKER "173.234.15.83"  // Ganti dengan IP broker MQTT
#define MQTT_PORT 1883
#define TOPIC_REQUEST "parkingo/scanner"
#define TOPIC_RESPONSE "parkingo/response"

// 🔑 API Key Static
#define API_KEY "cGFya2luZ28tbW9kdWxl"

// PIN
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM     0
#define SIOD_GPIO_NUM    26
#define SIOC_GPIO_NUM    27

#define Y9_GPIO_NUM      32  // Sebelumnya 35
#define Y8_GPIO_NUM      35  // Sebelumnya 34
#define Y7_GPIO_NUM      34  // Sebelumnya 39
#define Y6_GPIO_NUM      39  // Sebelumnya 36
#define Y5_GPIO_NUM      36  // Sebelumnya 21
#define Y4_GPIO_NUM      21  // Sebelumnya 19
#define Y3_GPIO_NUM      19  // Sebelumnya 18
#define Y2_GPIO_NUM      18  // Sebelumnya 5
#define VSYNC_GPIO_NUM   25
#define HREF_GPIO_NUM    23
#define PCLK_GPIO_NUM    22


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
        config.frame_size = FRAMESIZE_QVGA;
        config.jpeg_quality = 12;
        config.fb_count = 1;

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
    Serial.print("Connecting to WiFi");
    int counter = 0;
    while (WiFi.status() != WL_CONNECTED) {
        Serial.print(".");
        delay(1000);
        counter++;
        if (counter > 20) {  // Kalau lebih dari 20 detik, restart ESP32
            Serial.println("\n[⚠️ ERROR] WiFi Connection Failed. Restarting...");
            ESP.restart();
        }
    }
    Serial.println("\n[✅ WiFi Connected]");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    // Dapatkan MAC Address ESP32
    mac_address = WiFi.macAddress();
    Serial.print("ESP32 MAC Address: ");
    Serial.println(mac_address);

    // Inisialisasi kamera
    delay(2000); // Tambahkan delay agar sensor siap
    sensor_t * s = esp_camera_sensor_get();
    if (s == NULL) {
        Serial.println("[⚠️ ERROR] Sensor kamera tidak ditemukan!");
    } else {
        Serial.println("[✅ Kamera terdeteksi]");
    }

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
