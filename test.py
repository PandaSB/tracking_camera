import time
import threading
from PCA9685 import PCA9685

pwm = PCA9685(address=0x40, debug=True,port=3)


def set_servo_angle(channel, angle):
    pulse = int(500 + ((angle / 180.0) * 2000))  # Convert angle to pulse width
    pwm.setServoPulse(channel, pulse)
    print(f"Set servo on channel {channel} to angle {angle} degrees (pulse: {pulse})")


if __name__ == "__main__":
    print ("Starting PCA9685 script")

    pwm.setPWMFreq(46)  # Set frequency to 50Hz
    print ("PCA9685 initialized and frequency set to 50Hz")

    for channel in range (15):
        angle = 90  # Set to middle position
        set_servo_angle(channel, angle)
    
    input("Press Enter to exit...") # The script will pause here
    
    for channel in range (15):
        angle = 45  # Set to middle position
        set_servo_angle(channel, angle)
    
    input("Press Enter to exit...") # The script will pause here
    
    for channel in range (15):
        angle = 135  # Set to middle position
        set_servo_angle(channel, angle)

    input("Press Enter to exit...") # The script will pause here
    
    for channel in range (15):
        angle = 90  # Set to middle position
        set_servo_angle(channel, angle)
    
    print("Exiting now. Goodbye!")
    exit(0)


