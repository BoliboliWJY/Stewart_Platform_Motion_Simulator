#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm(0x40);

// Match the current controller's pulse calibration and PWM frequency.
// Disconnect all six rods and support the upper platform before use.
// PCA9685 channel numbers must match the physical servo labels 0 through 5.
int angleToPulse(float angle) {
  return int(250.0f + angle / 180.0f * (1148.0f - 250.0f));
}

void setHornElevation(float elevation) {
  for (int ch = 0; ch < 6; ++ch) {
    float target = 90.0f + ((ch % 2 == 0) ? elevation : -elevation);
    pwm.setPWM(ch, 0, angleToPulse(target));
  }
}

void setup() {
  Serial.begin(38400);
  Wire.begin(21, 22);
  pwm.begin();
  pwm.setPWMFreq(120);
  delay(500);
  setHornElevation(0.0f);
  Serial.println("RODS DISCONNECTED ONLY. All channels commanded to 90 degrees.");
  Serial.println("n: all 90 degrees; t: even channels 95, odd channels 85.");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'n') {
      setHornElevation(0.0f);
      Serial.println("All channels: 90 degrees.");
    } else if (command == 't') {
      setHornElevation(5.0f);
      Serial.println("Even channels: 95; odd channels: 85. Check all horns rise.");
    }
  }
}
