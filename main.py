import io
import cv2
import time
import threading
import ultralytics
from PCA9685 import PCA9685
from ultralytics import YOLO
from sixel import SixelWriter
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download


default_horizontal_angle = 120
default_vertical_angle = 110

pwm = PCA9685(address=0x40, debug=False,port=3)
horizontal_angle = default_horizontal_angle    # Default horizontal angle
vertical_angle = default_vertical_angle   # Default vertical angle


def set_servo_angle(channel, angle):
    pulse = int(500 + ((angle / 180.0) * 2000))  # Convert angle to pulse width
    pwm.setServoPulse(channel, pulse)
    #print(f"Set servo on channel {channel} to angle {angle} degrees (pulse: {pulse})")

def initpos():
    for channel in range(16):
        angle = 90  # Set to middle position
        set_servo_angle(channel, angle)
    print("All servos initialized to middle position (90 degrees).")

def horizontal_position(angle):
    if 0 <= angle <= 180:
        angle = 180 - angle  # Invert the angle for horizontal servos
        set_servo_angle(0, angle)
        print(f"Horizontal servo set to {angle} degrees.")
    else:
        print("Error: Angle must be between 0 and 180 degrees.")

def vertical_position(angle):
    if 0 <= angle <= 180:
        set_servo_angle(1, angle)
        print(f"Vertical servo set to {angle} degrees.")
    else:
        print("Error: Angle must be between 0 and 180 degrees.")

def set_position(horizontal_angle, vertical_angle):
    if 0 <= horizontal_angle <= 180 and 0 <= vertical_angle <= 180:
        horizontal_position(horizontal_angle)
        vertical_position( vertical_angle)
        print(f"Servos set to horizontal: {horizontal_angle} degrees, vertical: {vertical_angle} degrees.")
    else:
        print("Error: Angles must be between 0 and 180 degrees.")

def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)


def default_position():
    for channel in range(16):
        angle = 90  # Set to middle position
        set_servo_angle(channel, angle)
    print("All servos set to default position (90 degrees).")


if __name__ == "__main__":
    print("Starting control script")
    pwm.setPWMFreq(46)  # Set frequency to 50Hz
    initpos()
    set_position(horizontal_angle, vertical_angle)
    print("PCA9685 initialized and frequency set to 50Hz")
    ultralytics.checks()
    model = YOLO("yolo11n.pt")
    model.export(format="ncnn")
    ncnn_model = YOLO("yolo11n_ncnn_model")

    videoCap = cv2.VideoCapture(1)
    countloop = 0
    frame_counter = 0
    frame_skip_rate = 2
    while True:

        ret, frame = videoCap.read()
        frame = cv2.resize(frame, (320, 240))
        frame_counter += 1
        if not ret:
            exit("Failed to capture video frame. Please check the camera connection.")

        if frame_counter % frame_skip_rate == 0:
            print("Video frame captured successfully.")
            start_time = time.time()
            results = ncnn_model(frame)
            end_time = time.time()
            print(f"Inference time: {end_time - start_time:.4f} seconds")

            countloop += 1
            if countloop > 10:
                countloop = 0
                print("Resetting position to default.")
                horizontal_angle = default_horizontal_angle    # Default horizontal angle
                vertical_angle = default_vertical_angle   # Default vertical angle
                set_position(horizontal_angle, vertical_angle)


            for result in results:
            # get the classes names
                classes_names = result.names

                # iterate over each box
                for box in result.boxes:
                    # check if confidence is greater than 40 percent
                    if box.conf[0] > 0.4:
                        # get coordinates
                        [x1, y1, x2, y2] = box.xyxy[0]
                        # convert to int
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # get the class
                        cls = int(box.cls[0])

                        # get the class name
                        class_name = classes_names[cls]

                        # get the respective colour
                        colour = getColours(cls)

                        # draw the rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                        # put the class name and confidence on the image
                        cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                        # calculate the center of the box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        offset_x = center_x - (frame.shape[1] // 2)
                        offset_y = center_y - (frame.shape[0] // 2)

                        # draw the center point
                        cv2.circle(frame, (center_x, center_y), 5, colour, -1)

                        if (class_name == "person"):
                            # print the class name
                            print(f"Detected class: {class_name}")
                            # print the coordinates of the box
                            print(f"Coordinates: ({x1}, {y1}), ({x2}, {y2})")
                            # print the center coordinates
                            print(f"Center of {class_name}: ({center_x}, {center_y})")
                            # print the offset
                            print(f"Offset from center: ({offset_x}, {offset_y})")
                            if (offset_x > 10):
                                horizontal_angle = min(135, horizontal_angle + 2)
                            elif (offset_x < -10):
                                horizontal_angle = max(45, horizontal_angle - 2)
                            if (offset_y > 10):
                                vertical_angle = min(135, vertical_angle - 1)
                            elif (offset_y < -10):
                                vertical_angle = max(45, vertical_angle + 1)
                            set_position(horizontal_angle, vertical_angle)
                            print(f"Object detected: {class_name} at ({x1}, {y1}), ({x2}, {y2}) with confidence {box.conf[0]:.2f}.")
                            countloop = 0

                    
        filename = 'savedImage.jpg'

        # Using cv2.imwrite() method
        # Saving the image
        cv2.imwrite(filename, frame)
        im_bytes = cv2.imencode(".png",  frame,)[1].tobytes() 
        mem_file = io.BytesIO(im_bytes)
        w = SixelWriter()
        w.draw(mem_file)
    
    print("Exiting now. Goodbye!")
    exit(0)