/*
 * ==================================================================
 *  3-PHASE ENERGY MONITOR — MUXED PZEM + DWIN + FIREBASE
 *  = Merged from:
 *     (1) WiFi.h.txt    → Firebase upload + vib/temp + backoff
 *     (2) Arduino.h.txt → CD74HC4067 mux + address auto-fix + DWIN
 * ==================================================================
 */

#include <WiFi.h>
#include <Firebase_ESP_Client.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <PZEM004Tv30.h>
#include <time.h>
#include <Arduino.h>
#include <HardwareSerial.h>

// ================== WIFI ==================
#define WIFI_SSID       "Airtel_sahi_0849"
#define WIFI_PASSWORD   "air99772"

// ================== FIREBASE ==================
#define API_KEY         "AIzaSyC2A3qi7vfPHr83Kev2TGyJBiSGMM7kmYs"
#define DATABASE_URL    "https://new-project-210e3-default-rtdb.firebaseio.com"
#define USER_EMAIL      "sb284160@gmail.com"
#define USER_PASSWORD   "Password@1"

// ================== SENSOR PINS ==================
#define VIBRATION_PIN   27
#define TEMP_PIN        22
#define PIEZO_PIN       34

// ================== PZEM BUS (UART2) ==================
#define PZEM_RX         16
#define PZEM_TX         17

// ================== CD74HC4067 MUX ==================
#define MUX_S0          32
#define MUX_S1          33
#define MUX_S2          25

// Target PZEM addresses after auto-fix
#define PZEM_ADDR_1     0x01
#define PZEM_ADDR_2     0x02
#define PZEM_ADDR_3     0x03

// ================== DWIN (UART1) ==================
// pins: TX = 18, RX = 19, 115200 baud
#define DWIN_TX_PIN     18
#define DWIN_RX_PIN     19

// ================== SERIALS ==================
HardwareSerial PZEM_SERIAL(2);   // UART2 → muxed PZEM bus
HardwareSerial DWIN_SERIAL(1);   // UART1 → DWIN HMI

// ================== SENSORS ==================
OneWire oneWire(TEMP_PIN);
DallasTemperature sensors(&oneWire);

// ================== PZEM OBJECTS (muxed) ==================
PZEM004Tv30 pzem1(PZEM_SERIAL, PZEM_RX, PZEM_TX, PZEM_ADDR_1);
PZEM004Tv30 pzem2(PZEM_SERIAL, PZEM_RX, PZEM_TX, PZEM_ADDR_2);
PZEM004Tv30 pzem3(PZEM_SERIAL, PZEM_RX, PZEM_TX, PZEM_ADDR_3);

PZEM004Tv30* pzems[]      = { &pzem1, &pzem2, &pzem3 };
const char*  phaseNames[] = { "R", "S", "T" };           // Firebase keys
const char*  phaseLabel[] = { "Phase 1", "Phase 2", "Phase 3" };
uint16_t     vpBase[]     = { 0x2000, 0x2100, 0x2200 };

unsigned long successCount[3] = { 0, 0, 0 };
unsigned long failCount[3]    = { 0, 0, 0 };

// ================== FIREBASE ==================
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

// ================== DIGITAL VIBRATION ==================
unsigned long lastMeasure = 0;
int vibrationCount = 0;
const unsigned long interval = 1000;
int lastState = HIGH;
String vibrationIntensity = "NO VIBRATION";
int lastDigitalCount = 0;

// ================== PIEZO VIBRATION ==================
const unsigned long PIEZO_INTERVAL = 300;
unsigned long lastPiezoRead = 0;
int lastPiezoIntensity = 0;

// ================== TEMPERATURE ==================
float currentTemp = 0.0;
bool  tempValid   = false;

// ================== PZEM DATA ==================
struct PhaseData {
  float   voltage   = 0.0;
  float   current   = 0.0;
  float   power     = 0.0;
  float   energy    = 0.0;
  float   frequency = 0.0;
  float   pf        = 0.0;
  uint8_t addr      = 0;
  bool    valid     = false;
};
PhaseData phaseData[3];

// ================== TIMING ==================
unsigned long lastPZEMRead         = 0;
unsigned long lastSensorFirebase   = 0;
unsigned long lastPZEMFirebase     = 0;
unsigned long lastSerialPrint      = 0;

const unsigned long PZEM_INTERVAL             = 3000;
const unsigned long SENSOR_FIREBASE_INTERVAL  = 500;
const unsigned long PZEM_FIREBASE_INTERVAL    = 5000;
const unsigned long SERIAL_PRINT_INTERVAL     = 200;

// ================== PZEM BACKOFF ==================
unsigned long currentPZEMInterval = PZEM_INTERVAL;
int pzemConsecutiveFailures = 0;
const int PZEM_FAIL_THRESHOLD       = 3;
const unsigned long PZEM_MAX_INTERVAL = 30000;

// ================== FORWARD DECLS ==================
void syncTime();
void testPZEMs();
void readAllPhases();
bool readPhaseMux(uint8_t idx, PhaseData &data);
void updateDWIN();
void printSerialStatus();
void printPhaseStatus(uint8_t i);
void uploadSensorData();
void uploadPZEMData();
void tokenStatusCallback(TokenInfo info);

// ==============================================================
//  MUX helpers
// ==============================================================
void muxSelect(uint8_t ch) {
  digitalWrite(MUX_S0, (ch & 0x01) ? HIGH : LOW);
  digitalWrite(MUX_S1, (ch & 0x02) ? HIGH : LOW);
  digitalWrite(MUX_S2, (ch & 0x04) ? HIGH : LOW);
}

void flushPzem() {
  while (PZEM_SERIAL.available()) PZEM_SERIAL.read();
}

void selectChannel(uint8_t ch) {
  muxSelect(0);   delay(20);
  muxSelect(ch);  delay(30);
  flushPzem();
}

void selectPhase(uint8_t idx) {
  selectChannel(idx + 1);   // CH1, CH2, CH3
}

// ==============================================================
//  PZEM diagnostic + address auto-fix (runs once in setup)
// ==============================================================
bool probeChannel(uint8_t ch, uint8_t addr) {
  selectChannel(ch);
  PZEM004Tv30 probe(PZEM_SERIAL, PZEM_RX, PZEM_TX, addr);
  float v = probe.voltage();
  return !isnan(v);
}

bool changeAddress(uint8_t ch, uint8_t cur, uint8_t nw) {
  selectChannel(ch);
  PZEM004Tv30 fixer(PZEM_SERIAL, PZEM_RX, PZEM_TX, cur);
  float v = fixer.voltage();
  if (isnan(v)) {
    Serial.printf("    CH%d: cannot talk at 0x%02X\n", ch, cur);
    return false;
  }
  Serial.printf("    CH%d: talking at 0x%02X (V=%.1f), setting to 0x%02X...\n",
                ch, cur, v, nw);
  if (fixer.setAddress(nw)) { delay(500); return true; }
  return false;
}

void testPZEMs() {
  Serial.println("\n=== PZEM ADDRESS DIAGNOSTIC + AUTO-FIX ===");
  uint8_t found[4] = {0};

  Serial.println("STEP 1 — Discover address on each channel:");
  for (uint8_t ch = 1; ch <= 3; ch++) {
    bool detected = false;
    for (uint8_t a = 0x01; a <= 0x04; a++) {
      if (probeChannel(ch, a)) {
        found[ch] = a;
        Serial.printf("  CH%d: PZEM responds at 0x%02X\n", ch, a);
        detected = true;
        break;
      }
      delay(100);
    }
    if (!detected) Serial.printf("  CH%d: *** NO PZEM DETECTED ***\n", ch);
    delay(150);
  }

  uint8_t target[4] = {0, PZEM_ADDR_1, PZEM_ADDR_2, PZEM_ADDR_3};
  Serial.println("\nSTEP 2 — Correct any wrong address:");
  for (uint8_t ch = 1; ch <= 3; ch++) {
    if (!found[ch]) { Serial.printf("  CH%d: skipped (not detected)\n", ch); continue; }
    if (found[ch] == target[ch]) {
      Serial.printf("  CH%d: already correct (0x%02X)\n", ch, target[ch]); continue;
    }
    if (changeAddress(ch, found[ch], target[ch]))
      Serial.printf("  CH%d: OK 0x%02X -> 0x%02X\n", ch, found[ch], target[ch]);
    else
      Serial.printf("  CH%d: FAILED to change address\n", ch);
    delay(200);
  }

  Serial.println("\nSTEP 3 — Verify final state:");
  for (uint8_t ch = 1; ch <= 3; ch++) {
    if (probeChannel(ch, target[ch]))
      Serial.printf("  CH%d @ 0x%02X: OK\n", ch, target[ch]);
    else
      Serial.printf("  CH%d @ 0x%02X: STILL FAILING\n", ch, target[ch]);
    delay(150);
  }
  muxSelect(0);
  Serial.println("=== End PZEM Test ===\n");
}

// ==============================================================
//  PZEM read (muxed, with retries)
// ==============================================================
bool readPhaseMux(uint8_t idx, PhaseData &data) {
  for (int attempt = 0; attempt < 2; attempt++) {
    selectPhase(idx);
    float v = pzems[idx]->voltage();
    if (!isnan(v)) {
      data.voltage   = v;
      data.current   = pzems[idx]->current();
      data.power     = pzems[idx]->power();
      data.energy    = pzems[idx]->energy();
      data.frequency = pzems[idx]->frequency();
      data.pf        = pzems[idx]->pf();
      data.addr      = pzems[idx]->readAddress();
      data.valid     = true;
      return true;
    }
    if (attempt < 1) delay(100);
  }
  data.valid = false;
  return false;
}

void readAllPhases() {
  for (uint8_t i = 0; i < 3; i++) {
    if (readPhaseMux(i, phaseData[i])) successCount[i]++;
    else                               failCount[i]++;
  }
  muxSelect(0);   // disconnect all
}

// ==============================================================
//  DWIN
// ==============================================================
void DWIN_SendValue(uint16_t vpAddr, uint16_t value) {
  uint8_t frame[] = {
    0x5A, 0xA5, 0x05, 0x82,
    (uint8_t)(vpAddr >> 8), (uint8_t)(vpAddr & 0xFF),
    (uint8_t)(value  >> 8), (uint8_t)(value  & 0xFF)
  };
  DWIN_SERIAL.write(frame, sizeof(frame));
}

void DWIN_SendZeros(uint16_t base) {
  DWIN_SendValue(base + 0x00, 0);
  DWIN_SendValue(base + 0x20, 0);
  DWIN_SendValue(base + 0x30, 0);
  DWIN_SendValue(base + 0x40, 0);
  DWIN_SendValue(base + 0x50, 0);
  DWIN_SendValue(base + 0x60, 0);
  DWIN_SendValue(base + 0x70, 0);
}

void updateDWIN() {
  for (uint8_t i = 0; i < 3; i++) {
    uint16_t base = vpBase[i];
    if (!phaseData[i].valid) { DWIN_SendZeros(base); continue; }
    DWIN_SendValue(base + 0x00, (uint16_t)phaseData[i].addr);
    DWIN_SendValue(base + 0x20, (uint16_t)(phaseData[i].voltage   * 10));
    DWIN_SendValue(base + 0x30, (uint16_t)(phaseData[i].current   * 100));
    DWIN_SendValue(base + 0x40, (uint16_t)phaseData[i].power);
    DWIN_SendValue(base + 0x50, (uint16_t)(phaseData[i].energy    * 100));
    DWIN_SendValue(base + 0x60, (uint16_t)(phaseData[i].frequency * 10));
    DWIN_SendValue(base + 0x70, (uint16_t)(phaseData[i].pf        * 1000));
  }
}

// ==============================================================
//  SETUP
// ==============================================================
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\n=== Booting 3-phase monitor (mux + DWIN + Firebase) ===");

  // Sensor pins
  pinMode(VIBRATION_PIN, INPUT);
  sensors.begin();
  analogReadResolution(12);
  analogSetPinAttenuation(PIEZO_PIN, ADC_11db);

  // Mux pins
  pinMode(MUX_S0, OUTPUT);
  pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT);
  muxSelect(0);

  // PZEM UART
  PZEM_SERIAL.begin(9600, SERIAL_8N1, PZEM_RX, PZEM_TX);

  // DWIN UART
  DWIN_SERIAL.begin(115200, SERIAL_8N1, DWIN_TX_PIN, DWIN_RX_PIN);
  delay(500);

  // WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) { Serial.print("."); delay(300); }
  Serial.println("\nWiFi connected: " + WiFi.localIP().toString());

  syncTime();

  // PZEM address auto-fix
  testPZEMs();

  // Firebase
  config.api_key      = API_KEY;
  config.database_url = DATABASE_URL;
  auth.user.email     = USER_EMAIL;
  auth.user.password  = USER_PASSWORD;
  config.token_status_callback = tokenStatusCallback;

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
  Serial.println("Firebase starting...\n");
}

// ==============================================================
//  LOOP
// ==============================================================
void loop() {
  unsigned long now = millis();

  // 1. Digital vibration
  int sensorValue = digitalRead(VIBRATION_PIN);
  if (lastState == HIGH && sensorValue == LOW) vibrationCount++;
  lastState = sensorValue;

  if (millis() - lastMeasure >= interval) {
    if (vibrationCount == 0)       vibrationIntensity = "NO VIBRATION";
    else if (vibrationCount <= 5)  vibrationIntensity = "LOW";
    else if (vibrationCount <= 15) vibrationIntensity = "MEDIUM";
    else                           vibrationIntensity = "HIGH";
    lastDigitalCount = vibrationCount;
    vibrationCount = 0;
    lastMeasure = millis();
  }

  // 2. Piezo vibration (intensity only)
  if (millis() - lastPiezoRead >= PIEZO_INTERVAL) {
    lastPiezoRead = millis();
    int minimumValue = 4095, maximumValue = 0;
    unsigned long startTime = millis();
    while (millis() - startTime < 100) {
      int reading = analogRead(PIEZO_PIN);
      if (reading < minimumValue) minimumValue = reading;
      if (reading > maximumValue) maximumValue = reading;
    }
    lastPiezoIntensity = maximumValue - minimumValue;
  }

  // 3. Temperature
  sensors.requestTemperatures();
  float temp = sensors.getTempCByIndex(0);
  if (temp != DEVICE_DISCONNECTED_C) { currentTemp = temp; tempValid = true; }
  else                               { currentTemp = 0.0;  tempValid = false; }

  // 4. PZEM read (muxed) + DWIN refresh + backoff
  if (now - lastPZEMRead >= currentPZEMInterval) {
    lastPZEMRead = now;
    readAllPhases();
    updateDWIN();

    bool anyValid = phaseData[0].valid || phaseData[1].valid || phaseData[2].valid;
    if (anyValid) {
      pzemConsecutiveFailures = 0;
      currentPZEMInterval = PZEM_INTERVAL;
    } else {
      if (++pzemConsecutiveFailures >= PZEM_FAIL_THRESHOLD)
        currentPZEMInterval = min(currentPZEMInterval * 2, PZEM_MAX_INTERVAL);
    }
  }

  // 5. Firebase — sensor node
  if (Firebase.ready() &&
      (now - lastSensorFirebase >= SENSOR_FIREBASE_INTERVAL || lastSensorFirebase == 0)) {
    lastSensorFirebase = now;
    uploadSensorData();
  }

  // 6. Firebase — PZEM node
  if (Firebase.ready() &&
      (now - lastPZEMFirebase >= PZEM_FIREBASE_INTERVAL || lastPZEMFirebase == 0)) {
    lastPZEMFirebase = now;
    uploadPZEMData();
  }

  // 7. Serial status
  if (now - lastSerialPrint >= SERIAL_PRINT_INTERVAL) {
    lastSerialPrint = now;
    printSerialStatus();
  }
}

// ==============================================================
//  Helpers
// ==============================================================
void syncTime() {
  Serial.print("Syncing NTP time");
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");
  time_t now = time(nullptr);
  while (now < 24 * 3600) { Serial.print("."); delay(500); now = time(nullptr); }
  Serial.println(" done.");
}

void printPhaseStatus(uint8_t i) {
  if (phaseData[i].valid) {
    Serial.printf("  [%s] addr=0x%02X V=%.1fV I=%.2fA P=%.1fW E=%.3fkWh F=%.1fHz PF=%.2f\n",
                  phaseNames[i], phaseData[i].addr,
                  phaseData[i].voltage, phaseData[i].current, phaseData[i].power,
                  phaseData[i].energy, phaseData[i].frequency, phaseData[i].pf);
  } else {
    Serial.printf("  [%s] READ FAILED (ok=%lu fail=%lu)\n",
                  phaseNames[i], successCount[i], failCount[i]);
  }
}

void printSerialStatus() {
  Serial.println("\n──────── SENSOR ────────");
  Serial.printf("  Digital Vib:  %d (%s)\n", lastDigitalCount, vibrationIntensity.c_str());
  Serial.printf("  Piezo Vib:    intensity=%d\n", lastPiezoIntensity);
  if (tempValid) Serial.printf("  Temperature:  %.2f °C\n", currentTemp);
  else           Serial.println("  Temperature:  READ FAILED");
  Serial.println("──────── PZEM ──────────");
  for (int i = 0; i < 3; i++) printPhaseStatus(i);
  Serial.printf("  Next PZEM read: %lu ms\n", currentPZEMInterval);
  Serial.println("────────────────────────");
}

void uploadSensorData() {
  FirebaseJson json;
  json.set("vibration/count",          lastDigitalCount);
  json.set("vibration/intensity",      vibrationIntensity);
  json.set("vibration/piezoIntensity", lastPiezoIntensity);
  json.set("temperature/celsius",      tempValid ? currentTemp : 0.0f);
  if (!Firebase.RTDB.updateNode(&fbdo, "/", &json))
    Serial.printf("[Firebase] Sensor upload failed: %s\n", fbdo.errorReason().c_str());
}

void uploadPZEMData() {
  FirebaseJson json;
  const char* keys[] = { "phaseR", "phaseS", "phaseT" };

  for (int i = 0; i < 3; i++) {
    String p = String(keys[i]) + "/";
    bool ok = phaseData[i].valid;
    json.set(p + "voltage",     ok ? phaseData[i].voltage   : 0.0f);
    json.set(p + "current",     ok ? phaseData[i].current   : 0.0f);
    json.set(p + "power",       ok ? phaseData[i].power     : 0.0f);
    json.set(p + "energy",      ok ? phaseData[i].energy    : 0.0f);
    json.set(p + "frequency",   ok ? phaseData[i].frequency : 0.0f);
    json.set(p + "powerFactor", ok ? phaseData[i].pf        : 0.0f);
  }

  if (!Firebase.RTDB.updateNode(&fbdo, "/", &json))
    Serial.printf("[Firebase] PZEM upload failed: %s\n", fbdo.errorReason().c_str());
}

void tokenStatusCallback(TokenInfo info) {
  if (info.status == token_status_ready) {
    Serial.println("[Firebase] Token: READY");
  } else if (info.status != token_status_on_request &&
             info.status != token_status_on_signing &&
             info.status != token_status_on_refresh) {
    Serial.printf("[Firebase] Token ERROR: code=%d, message=%s\n",
                  info.error.code, info.error.message.c_str());
  }
}