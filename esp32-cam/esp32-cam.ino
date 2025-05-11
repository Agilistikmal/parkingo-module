#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "base64.h"
#include <esp_task_wdt.h>
#include <ArduinoJson.h>

// Debug flags
#define DEBUG_MODE true

// Watchdog timeout in seconds
#define WDT_TIMEOUT 15

// PIN
#define CAMERA_MODEL_AI_THINKER
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27

#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM    5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22
// 4 for flash led or 33 for normal led
#define LED_GPIO_NUM   4
// Buzzer pin - adjust as needed for your hardware setup
#define BUZZER_PIN     2

// API endpoint - changed to HTTP for now to avoid SSL issues
#define API_ENDPOINT "https://parkingo-module.agil.zip/scanner"

// 🔑 API Key Static
#define API_KEY "cGFya2luZ28tbW9kdWxl"

// WiFi credentials and slot name - UBAH DI SINI
const char* ssid = "AGL 1";
const char* password = "1234567890";

const char* parkingSlug = "krasty-krab-uty";
const char* slotName = "P12";

String mac_address;

// Debug print helper
void debugPrint(String message) {
    if (DEBUG_MODE) {
        Serial.println(message);
    }
}

// Function to alert for invalid booking (buzzer and LED)
void alertInvalidBooking() {
    Serial.println("[🚨 ALERT] Invalid booking detected!");
    
    // Flash LED 5 times and beep buzzer
    for (int i = 0; i < 5; i++) {
        // Turn on LED and buzzer
        digitalWrite(LED_GPIO_NUM, HIGH);
        tone(BUZZER_PIN, 2000); // 2kHz tone
        delay(200);
        
        // Turn off LED and buzzer
        digitalWrite(LED_GPIO_NUM, LOW);
        noTone(BUZZER_PIN);
        delay(200);
        
        // Reset watchdog
        esp_task_wdt_reset();
    }
    
    // Final long beep
    digitalWrite(LED_GPIO_NUM, HIGH);
    tone(BUZZER_PIN, 1500); // 1.5kHz tone
    delay(500);
    digitalWrite(LED_GPIO_NUM, LOW);
    noTone(BUZZER_PIN);
    
    esp_task_wdt_reset();
}

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
        config.frame_size = FRAMESIZE_CIF;     // Gunakan resolusi CIF yang lebih kecil (352x288)
        config.jpeg_quality = 20;              // Kualitas lebih rendah untuk mengurangi ukuran
        config.fb_count = 2;
    } else {
        config.frame_size = FRAMESIZE_QVGA;    // QVGA jika tidak ada PSRAM (320x240)
        config.jpeg_quality = 20;              // Lower quality
        config.fb_count = 1;
    }

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("[⚠️ ERROR] Kamera gagal diinisialisasi!");
        while (true);
    }

    sensor_t * s = esp_camera_sensor_get();
    if (s) {
        // Konfigurasi tambahan untuk optimisasi gambar
        s->set_brightness(s, 2);     // Tingkatkan brightness lebih
        s->set_contrast(s, 2);       // Tingkatkan contrast lebih
        s->set_saturation(s, 1);     // Sedikit saturasi
        s->set_special_effect(s, 0); // No special effect
        s->set_whitebal(s, 1);       // Enable white balance
        s->set_awb_gain(s, 1);       // Enable auto white balance gain
        s->set_wb_mode(s, 0);        // Auto white balance
        s->set_exposure_ctrl(s, 1);  // Enable auto exposure
        s->set_aec2(s, 0);          // Disable night mode
        s->set_gain_ctrl(s, 1);      // Enable auto gain control
        s->set_agc_gain(s, 0);       // No extra gain
        s->set_gainceiling(s, (gainceiling_t)0);  // No gain ceiling
        s->set_bpc(s, 0);           // No black pixel correction
        s->set_wpc(s, 1);           // Enable white pixel correction
        s->set_raw_gma(s, 1);       // Enable gamma correction
        s->set_lenc(s, 1);          // Enable lens correction
        s->set_hmirror(s, 0);       // No horizontal mirror
        s->set_vflip(s, 0);         // No vertical flip
        s->set_dcw(s, 1);           // Enable downsize crop
    }
}

// Function to send image data with headers for metadata - simplified for stability
bool sendImageData(camera_fb_t * fb) {
    if (!fb) {
        Serial.println("[⚠️ ERROR] Invalid camera frame buffer");
        return false;
    }
    
    // Reset watchdog before starting HTTP operations
    esp_task_wdt_reset();
    
    bool success = false;
    bool is_valid_booking = false;
    
    Serial.print("Connecting to endpoint: ");
    Serial.println(API_ENDPOINT);
    
    HTTPClient http;
    
    // Begin HTTP connection with shorter timeout
    http.begin(API_ENDPOINT);
    http.setTimeout(5000); // 5 second timeout - shorter to prevent hanging
    
    // Add metadata as HTTP headers with X- prefix
    http.addHeader("X-API-KEY", API_KEY);
    http.addHeader("X-MAC-ADDRESS", mac_address);
    http.addHeader("X-PARKING-SLUG", parkingSlug);
    http.addHeader("X-SLOT", slotName);
    http.addHeader("Content-Type", "image/jpeg");
    
    Serial.print("Image size: ");
    Serial.println(fb->len);
    
    // Reset watchdog again just before POST
    esp_task_wdt_reset();
    
    // Start the request but don't wait for completion
    Serial.println("Starting HTTP POST request...");
    
    // Actual image POST
    int httpCode = http.POST(fb->buf, fb->len);
    
    Serial.println("HTTP request completed with code: " + String(httpCode));
    
    // Reset watchdog after POST
    esp_task_wdt_reset();
    
    // Process response
    if (httpCode > 0) {
        Serial.printf("[✅ HTTP Response code: %d]\n", httpCode);
        if (httpCode == HTTP_CODE_OK) {
            String payload = http.getString();
            Serial.println("Response: " + payload);
            
            // Parse JSON response
            DynamicJsonDocument doc(1024);
            DeserializationError error = deserializeJson(doc, payload);
            
            if (!error) {
                // Check if plate number was detected
                if (doc["data"].containsKey("plate_number")) {
                    String plateNumber = doc["data"]["plate_number"];
                    Serial.println("Detected plate: " + plateNumber);
                    
                    // Check booking validity
                    if (doc["data"].containsKey("is_valid_booking_order")) {
                        is_valid_booking = doc["data"]["is_valid_booking_order"];
                        Serial.print("Valid booking order: ");
                        Serial.println(is_valid_booking ? "YES" : "NO");
                        
                        // Trigger alert if not a valid booking
                        if (!is_valid_booking) {
                            alertInvalidBooking();
                        }
                    }
                } else {
                    Serial.println("No plate number detected in response");
                }
                
                success = true;
            } else {
                Serial.print("JSON parsing error: ");
                Serial.println(error.c_str());
            }
        } else {
            Serial.printf("Server responded with non-OK status code: %d\n", httpCode);
        }
    } else {
        Serial.printf("[⚠️ HTTP Error: %s]\n", http.errorToString(httpCode).c_str());
        Serial.println("Endpoint may be unreachable. Continuing operation...");
    }
    
    // Always clean up HTTP connection
    http.end();
    Serial.println("HTTP connection closed");
    
    return success;
}

// Function to send HTTP POST request with JSON (kept for reference)
void sendHttpRequest(String &payload) {
    HTTPClient http;
    
    Serial.print("Connecting to endpoint: ");
    Serial.println(API_ENDPOINT);
    
    // Configure http connection
    http.begin(API_ENDPOINT);
    http.addHeader("Content-Type", "application/json");
    
    // Send POST request
    int httpResponseCode = http.POST(payload);
    
    // Check response
    if (httpResponseCode > 0) {
        Serial.print("[✅ HTTP Response code: ");
        Serial.print(httpResponseCode);
        Serial.println("]");
        
        String response = http.getString();
        Serial.println("Response: " + response);
    }
    else {
        Serial.print("[⚠️ HTTP Error code: ");
        Serial.print(httpResponseCode);
        Serial.println("]");
    }
    
    // Free resources
    http.end();
}

// 🔥 Fungsi untuk mengambil gambar & mengirim via HTTP
void captureAndSend() {
    Serial.println("\n[📸 Capturing image...]");
    
    // Reset watchdog timer
    esp_task_wdt_reset();
    
    digitalWrite(LED_GPIO_NUM, HIGH);
    delay(100);

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("[⚠️ ERROR] Gagal mengambil gambar!");
        digitalWrite(LED_GPIO_NUM, LOW);
        return;
    }

    digitalWrite(LED_GPIO_NUM, LOW);
    Serial.println("[✅ Image captured]");
    Serial.print("Image size (bytes): ");
    Serial.println(fb->len);
    Serial.print("Free heap: ");
    Serial.println(ESP.getFreeHeap());

    // Send image using updated approach
    bool success = false;
    
    // Check heap before sending
    if (ESP.getFreeHeap() > fb->len + 10000) {
        Serial.println("Attempting to send image...");
        success = sendImageData(fb);
        Serial.println("Send attempt completed");
    } else {
        Serial.println("[⚠️ WARNING] Not enough memory to send image safely. Skipping send.");
    }
    
    // Free the buffer
    esp_camera_fb_return(fb);
    Serial.println("Camera buffer released");
    
    if (success) {
        Serial.println("[✅ Image sent successfully]");
    } else {
        Serial.println("[⚠️ Image sending failed or skipped. Will retry on next cycle.");
    }
    
    // Final watchdog reset after completing the cycle
    esp_task_wdt_reset();
}

void setup() {
    Serial.begin(115200);
    Serial.println("\n[🚀 ParkingGo Camera Starting...]");
    
    // Set up watchdog
    Serial.println("Setting up watchdog timer...");
    
    // Try disabling existing watchdog first
    esp_task_wdt_deinit();
    
    // Initialize with appropriate parameters for your version
    #if ESP_IDF_VERSION_MAJOR >= 4 // IDF 4+
        esp_task_wdt_config_t wdtConfig;
        wdtConfig.timeout_ms = WDT_TIMEOUT * 1000;
        wdtConfig.idle_core_mask = 0;
        wdtConfig.trigger_panic = true;
        esp_task_wdt_init(&wdtConfig);
    #else // ESP32 Arduino 1.0.x
        esp_task_wdt_init(WDT_TIMEOUT, true);
    #endif
    
    esp_task_wdt_add(NULL);
    Serial.println("Watchdog initialized");

    // Setup pins
    pinMode(LED_GPIO_NUM, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    
    digitalWrite(LED_GPIO_NUM, LOW);
    
    // Connect to WiFi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    
    // Wait up to 20 seconds for connection
    unsigned long startAttemptTime = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < 20000) {
        delay(500);
        Serial.print(".");
        esp_task_wdt_reset(); // Reset watchdog while waiting
    }
    
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\n[⚠️ Failed to connect to WiFi]");
        ESP.restart(); // Restart if connection fails
    } else {
        Serial.println("\n[✅ WiFi Connected]");
        Serial.print("IP Address: ");
        Serial.println(WiFi.localIP());
    }

    // Dapatkan MAC Address ESP32
    mac_address = WiFi.macAddress();
    Serial.print("ESP32 MAC Address: ");
    Serial.println(mac_address);
    Serial.print("Parking: ");
    Serial.println(parkingSlug);
    Serial.print("Slot Name: ");
    Serial.println(slotName);

    // Inisialisasi kamera
    Serial.println("[🔄 Initializing camera...]");
    initCamera();
    Serial.println("[✅ Camera initialized]");

    // Test the buzzer
    tone(BUZZER_PIN, 1000);
    delay(100);
    noTone(BUZZER_PIN);

    // Blink LED to indicate ready
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_GPIO_NUM, HIGH);
        delay(100);
        digitalWrite(LED_GPIO_NUM, LOW);
        delay(100);
    }
    
    Serial.println("[✅ System Ready]");
}

void loop() {
    // Reset watchdog at start of loop
    esp_task_wdt_reset();
    
    // Check if WiFi is connected
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[⚠️ WiFi disconnected, reconnecting...]");
        WiFi.begin(ssid, password);
        delay(5000);
        esp_task_wdt_reset(); // Reset watchdog after delay
        return;
    }

    // Get free heap memory
    uint32_t freeHeap = ESP.getFreeHeap();
    Serial.print("Free heap: ");
    Serial.println(freeHeap);
    
    // Add a minimum threshold before trying to capture
    if (freeHeap < 30000) {
        Serial.println("[⚠️ Low memory, waiting to recover...]");
        delay(5000);
        esp_task_wdt_reset(); // Reset watchdog after delay
        return;
    }

    // Kirim gambar setiap 8 detik (increased delay)
    Serial.println("[🔄 Starting capture cycle...]");
    
    // Capture and send, with extra debugging
    captureAndSend();
    
    Serial.println("[💤 Waiting for next cycle...]");
    
    // Use a series of shorter delays with watchdog resets between
    for (int i = 0; i < 8; i++) {
        delay(1000);
        esp_task_wdt_reset();  // Reset watchdog every second
    }
    
    Serial.println("[🔄 Loop completed]");
}