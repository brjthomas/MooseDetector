"""
Generates a 2kHz square wave on a Raspberry Pi GPIO pin using hardware PWM.
Run with: python3 square_wave.py
Stop with: Ctrl+C
"""

import RPi.GPIO as GPIO
import time

# --- Configuration ---
GPIO_PIN = 18       # BCM pin number (pin 18 supports hardware PWM on Pi)
FREQUENCY_HZ = 2000 # 2 kHz
DUTY_CYCLE = 50     # 50% duty cycle = true square wave
# ---------------------

GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_PIN, GPIO.OUT)

pwm = GPIO.PWM(GPIO_PIN, FREQUENCY_HZ)
pwm.start(DUTY_CYCLE)

print(f"Generating {FREQUENCY_HZ}Hz square wave on GPIO{GPIO_PIN}. Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    pwm.stop()
    GPIO.cleanup()
    print("Stopped.")
