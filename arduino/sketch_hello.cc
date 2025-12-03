// Arduino Uno: Simple serial "Hello World" + LED control
// Protocol (newline-terminated):
//   HELLO      -> WORLD
//   LED ON     -> OK
//   LED OFF    -> OK

const int LED_PIN = 13;
void setup() {
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(9600);
  // Wait for USB serial on native USB boards; harmless on Uno
  while (!Serial) { ; }
  delay(500);                 // give host a moment after reset
  Serial.println("READY");    // handshake so Python knows we're alive
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim(); // remove \r and spaces

    if (line.equalsIgnoreCase("HELLO")) {
      Serial.println("WORLD");
    } else if (line.equalsIgnoreCase("LED ON")) {
      digitalWrite(LED_PIN, HIGH);
      Serial.println("OK");
    } else if (line.equalsIgnoreCase("LED OFF")) {
      digitalWrite(LED_PIN, LOW);
      Serial.println("OK");
    } else if (line.length() > 0) {
      Serial.println("ERR Unknown command");
    }
  }
}
