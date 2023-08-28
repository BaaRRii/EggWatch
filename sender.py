import time
import RPi.GPIO as GPIO
import threading
class EggSignalSender:
	def __init__(self, pins):		
		self.D0 = pins[0]z
		self.D1 = pins[1]
		self.D2 = pins[2]
		self.D3 = pins[3]
		
		GPIO.setmode(GPIO.BCM)
		GPIO.setup(self.D0, GPIO.OUT)
		GPIO.setup(self.D1, GPIO.OUT)
		GPIO.setup(self.D2, GPIO.OUT)
		GPIO.setup(self.D3, GPIO.OUT)
		
		self.lock = threading.Lock()
		
	def codeMessage(self, zone):
		zone += 1
		binary = format(zone, '03b')
		return binary
		
	def sendMessage(self, codedMessage):
		with self.lock:
			print("D0", 1)
			print("D1", int(codedMessage[0]))
			print("D2", int(codedMessage[1]))
			print("D3", int(codedMessage[2]))
			
			GPIO.output(self.D0, 1)
			GPIO.output(self.D1, int(codedMessage[0]))
			GPIO.output(self.D2, int(codedMessage[1]))
			GPIO.output(self.D3, int(codedMessage[2]))
			
			time.sleep(0.15)
			
			GPIO.output(self.D0, 0)
			GPIO.output(self.D1, 0)
			GPIO.output(self.D2, 0)
			GPIO.output(self.D3, 0)
			
			time.sleep(0.15)
		
	def close(self):
		GPIO.cleanup()
		
		
class ErrorSender:
    def __init__(self, pins):
        self.E0 = pins[0]
        self.E1 = pins[1]
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.E0, GPIO.OUT)
        GPIO.setup(self.E1, GPIO.OUT)
        	
		# Initialise error GPIOs in 0 0
        GPIO.output(self.E0, GPIO.LOW)
        GPIO.output(self.E1, GPIO.LOW)
			        
    def sendError(self, cameraError, undefinedError):
        if not cameraError and not undefinedError:
            # Status OK 1 1
            GPIO.output(self.E0, 1)
            GPIO.output(self.E1, 1)
            time.sleep(0.15)
            
        elif cameraError:
            # Camera Error 0 1
            GPIO.output(self.E0, 0)
            GPIO.output(self.E1, 1)
            
            time.sleep(0.15)
            
        elif undefinedError:
            # Undefined Error 1 0
            GPIO.output(self.E0, 1)
            GPIO.output(self.E1, 0)
            time.sleep(0.15)
            
    def close(self):
        GPIO.cleanup()
    
# sender = Sender([17,27,22,23])
# coded = sender.codeMessage("A")
# thread = threading.Thread(target=sender.sendMessage, args=(coded,))
# thread.start()
# coded = sender.codeMessage("H")
# sender.sendMessage(coded)
# sender.close()
