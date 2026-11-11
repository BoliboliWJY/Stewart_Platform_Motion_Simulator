#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm(0x40);

int angleToPulse(int angle) {
  return map(angle, 0, 180, 250, 1148);
}

void setup() {
  Wire.begin(21, 22);
  pwm.begin();
  pwm.setPWMFreq(120);
  delay(500);

  // 将 0 到 5 号通道全部设定为 90°
  for (int ch = 0; ch < 6; ch++) {
    pwm.setPWM(ch, 0, angleToPulse(90));
  }
}

void loop() {}