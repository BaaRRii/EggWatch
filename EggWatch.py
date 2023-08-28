from camera import Camera
from counter import Counter
from sender import ErrorSender

import RPi.GPIO as GPIO
import time
import threading

# GPIOs that will receive the signals
PIN_A = 5 # -> Check errors
PIN_B = 6 # -> Start egg count

# GPIOs that will send error signals 
    # 0 0 -> System is off (no signal)
    # 0 1 -> Camara is not working
    # 1 0 -> Undefined / software problem
    # 1 1 -> Everything is OK
PIN_ERROR_0 = 7 
PIN_ERROR_1 = 1

# Define error sender
errorSender = ErrorSender([PIN_ERROR_0,PIN_ERROR_1])

# Setup pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_A, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(PIN_B, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Define devices IPs
# cameraIp = "192.168.1.10" # Default IP address for the camera
cameraIp = input("Camera IP:") # Default IP address for the camera
plcIp = "192.168.1.11" # Default IP address for the PLC
plcPort = 12345

# Define camera
camera = Camera(cameraIp)

# Define counter
counter = Counter(camera, (plcIp, plcPort))

# Function to check system status (camera)
def checkSystem(camera):
    # Check camera status
    try:
        camera.open()
        camera.readFrame()
        camera.close()
        print("Camera working correctly")
        
    except Exception as e:
        print("Error with camera")
        errorSender.sendError(True, False)
        
# Function to set error pins given a exception
def handleException(e):
    if str(e) == "Error with camera":
        cameraError = True
    else:
        undefinedError = True
        
    errorSender.sendError(cameraError, undefinedError)
    
# Function to monitor the signal of the egg count 
def monitor_pin_b():
    while True:
        if GPIO.input(PIN_B) == GPIO.LOW:
            print("Pin B is low. Stopping counting process.")
            counter.stop_process()
            break
        time.sleep(1)

# Main loop
while True:
    try:
        if GPIO.input(PIN_A) == GPIO.HIGH:
            print("Signal received: check errors")
            
            checkSystem(camera)
            time.sleep(1) # Debounce
            
        elif GPIO.input(PIN_B) == GPIO.HIGH:
            print("Signal received: start count")
            
            # Start monitoring PIN_B
            monitor_thread = threading.Thread(target=monitor_pin_b)
            monitor_thread.start()

            # Start counting process
            counter.process() # Count eggs

            # Join the monitoring thread
            monitor_thread.join()

            time.sleep(1) # Debounce
            
        else:
            print("Waiting signals...")
            time.sleep(1)
            
    # Ending program if KeyboardInterrupt
    except KeyboardInterrupt:
        break
    
    # Handle errors and send error signals
    except Exception as e:
        handleException(e)

print("Program ended")