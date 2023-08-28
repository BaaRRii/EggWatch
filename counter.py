import cv2
import configparser
import numpy as np
import time
import threading
from sender import EggSignalSender
from ethernetSender import EthernetSender
from tracker import EuclideanDistTracker
from collections import deque

class Counter:
    def __init__(self, camera, PLC):
        self.camera = camera
        self.tracker = EuclideanDistTracker()
        self.crossed_ids = deque(maxlen=50)
        self.crossed_ids_reverse = deque(maxlen=50)
        self.scaleFactor = 10
        self.counter = 0
        self.n_counter = 0
        self.payload = ""
        self.n_payload = 1
        self.stop = False
        
        self.sender = EggSignalSender([23,17,27,22])
        # plcIP = "127.0.0.1"
        plcIP = PLC[0]
        # plcPORT = 12345
        plcPORT = PLC[1]
        self.ethernetSender = EthernetSender(plcIP, plcPORT)
        
        # try:
        #     self.camera.open()
        # except Exception as e:
        #     raise e
        
    def loadConfig(self, filename="config.ini"):
        """
        Loads the configuration from a .ini file.

        Parameters:
        filename (str): The name of the configuration file to load. Defaults to "config.ini".

        Returns:
        None. However, instance variables are updated with the configuration values.
        """
        print("Loading data from:", filename)
        config = configparser.ConfigParser()
        config.read(filename)
        settings = config["SETTINGS"]

        self.h_low = int(settings.get('H_low'))
        self.s_low = int(settings.get('S_low'))
        self.v_low = int(settings.get('V_low'))
        self.h_high = int(settings.get('H_high'))
        self.s_high = int(settings.get('S_high'))
        self.v_high = int(settings.get('V_high'))
        self.threshold = int(settings.get('Threshold'))
        self.radius = int(settings.get('Radius_template'))
        self.zone_start = int(settings.get('Zone_Start'))
        self.zone_end = int(settings.get('Zone_End'))
        self.roi_start = int(settings.get('Roi_Start'))
        self.roi_end = int(settings.get('Roi_End'))
        self.n_zones = int(settings.get('N_zones'))
        
        # Adjust to frame width
        if self.camera.isOpened():
            frame = self.camera.readFrame()
            frame = self.scaleImage(frame)
            width = frame.shape[1]
            self.zone_start = int(self.zone_start/100 * width)
            self.zone_end = int(self.zone_end/100 * width)
        
        # Update variables
        self.zone_width = int((self.zone_end - self.zone_start)/self.n_zones)
        self.lower = np.array([self.h_low, self.s_low, self.v_low])
        self.upper = np.array([self.h_high, self.s_high, self.v_high])
        
        # Counter number
        info = config["INFO"]
        self.n_counter = int(info.get("n_counter"))
        
    def scaleImage(self, frame):
        """
        Resizes an image according to a given percentage defined in the instance.

        Parameters:
        frame (np.array): The image to be resized.

        Returns:
        frame (np.array): The resized image.
        """
        width = int(frame.shape[1] * self.scaleFactor / 100)
        height = int(frame.shape[0] * self.scaleFactor / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    
    def colorMask(self, roi):
        """
        Applies a color mask to the region of interest.

        Parameters:
        roi (np.array): The region of interest to which the mask will be applied.

        Returns:
        mask (np.array): The mask applied to the region of interest.
        """
        # Smoothing filter before applying the color mask
        blur = cv2.GaussianBlur(roi, (11, 11), 1)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        return mask
    
    def createTemplate(self):
        """
        Creates a template of an ellipse simulating an egg.

        Parameters:
        None. Uses the radius defined in the instance.

        Returns:
        norm_dist_template (np.array): The normalized template of the circle.
        """
        template = np.zeros((2*self.radius, 2*self.radius), dtype=np.uint8)

        center_coordinates = (self.radius, self.radius)
        axes_length = (int(self.radius/1.3), self.radius)
            
        template=cv2.ellipse(img=template, center=center_coordinates, axes=axes_length, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
        dist_template = cv2.distanceTransform(template, cv2.DIST_L2, 5)
        norm_dist_template = cv2.normalize(dist_template, None, 0, 1.0, cv2.NORM_MINMAX)
        return norm_dist_template
    
    def templateMatching(self, norm_dist_template, norm_dist_transform):
        """
        Performs template matching of a given template on a given image

        Args:
            norm_dist_template (np.array): template
            norm_dist_transform (np.array): image

        Returns:
            boxes (list): list of bounding boxes of the detections
        """
        res = cv2.matchTemplate(norm_dist_transform, norm_dist_template, cv2.TM_CCORR_NORMED)
        loc = np.where(res >= self.threshold/100)

        boxes = [[int(pt[0]), int(pt[1]), int(2*self.radius), int(2*self.radius)] for pt in zip(*loc[::-1])]

        scores = res[loc]

        indices = cv2.dnn.NMSBoxes(boxes, scores.tolist(), 0.5, 0.4)

        boxes = [boxes[i[0]] for i in indices]
        return boxes
    
    def stop_process(self):
        """
        Turns the stop flag to True
        
        Parameters:
        None.
        
        Returns:
        None.
        """
        self.stop = True
    
    def process(self):       
        """
        Starts the main processing of the video feed.

        Parameters:
        None.

        Returns:
        None.
        """ 
        self.loadConfig()
        norm_dist_template = self.createTemplate()
        start_time = time.time()
        
        try:
            self.camera.open()
        except Exception as e:
            raise e
            
        
        while self.camera.isOpened():
            try:
                if self.stop:
                    break
                frame = self.camera.readFrame()
                ############################################################
                # Image scaling
                ############################################################
                frame = self.scaleImage(frame)
                roi = frame[self.roi_start:self.roi_end, :].copy()

                ############################################################
                # Color mask
                ############################################################
                mask = self.colorMask(roi)

                ############################################################
                # Mask adjustment
                ############################################################
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                roi = cv2.bitwise_and(roi, roi, mask=mask)

                ############################################################
                # Distance transform and matching
                ############################################################
                # 0s margin for the borders
                mask = cv2.copyMakeBorder(mask, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)

                dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                norm_dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
                # cv2.imshow("", norm_dist_transform)

                boxes = self.templateMatching(norm_dist_template, norm_dist_transform)
                
                for box in boxes:
                    cv2.rectangle(frame[self.roi_start:self.roi_end, :], (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 1)

                ids = self.tracker.update(boxes)
                
                ############################################################
                # Count the bounding boxes that go through the line
                ############################################################

                # Calculate the midpoint of the region of interest (ROI)
                mid_height = (self.roi_end - self.roi_start) // 2

                # Loop through all identified objects (eggs) to determine their direction
                for cid in ids:
                    x, y, w, h, id = cid

                    # Display the unique ID on the frame for visualization purposes
                    cv2.putText(frame[self.roi_start:self.roi_end, :], str(id), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    
                    # Fetch the current and previous center positions of the egg for direction calculation
                    current_center = self.tracker.center_points[id]
                    prev_center = self.tracker.prev_center_points.get(id, (10000, 10000))
                    
                    # Determine the direction of movement of the egg
                    if current_center[1] > prev_center[1]:
                        direction = "Descending"
                    else:
                        direction = "Ascending"
                    
                    # Calculate the boundaries for the egg to determine if it crosses the midline
                    lower_boundary = current_center[1] + h//2                        
                    upper_boundary = current_center[1] - h//2      
                    
                    if upper_boundary < mid_height and lower_boundary > mid_height:
                        center = x+w/2
                        # Calculate the zone in which the egg is currently located
                        zone = int((center-self.zone_start) // self.zone_width)
                        
                        # Handle egg counting depending on its direction
                        if direction == "Ascending" and id not in self.crossed_ids:
                            self.crossed_ids.append(id)
                            self.counter += 1
                            print("{}{}".format(self.counter, chr(zone+65)))
                            
                            # Send the counting data to the external system
                            coded = self.sender.codeMessage(zone)
                            thread = threading.Thread(target=self.sender.sendMessage, args=(coded,))
                            thread.start()
                            
                            payload = f"{chr(zone+65)}{self.counter:06d}{int(time.time() - start_time)}"
                            self.payload += payload
                            
                        elif direction == "Descending" and id not in self.crossed_ids_reverse:
                            self.crossed_ids_reverse.append(id)
                            if id in self.crossed_ids:
                                self.crossed_ids.remove(id)
                            self.counter -= 1
                            print("{}{}".format(self.counter, chr(zone+65)))
                            
                            # Send the corrected counting data to the external system
                            coded = self.sender.codeMessage(-1)
                            thread = threading.Thread(target=self.sender.sendMessage, args=(coded,))
                            thread.start()
                            
                            payload = f"{chr(zone+65)}{self.counter:06d}{int(time.time() - start_time)}"
                            self.payload += payload

                # Check if 5 seconds have passed and then send the data
                current_time = time.time()
                if current_time - start_time >= 5:
                    header = f"{self.n_counter:02d}{self.n_payload:04d}"
                    print("Sending: ", header+self.payload)
                    self.ethernetSender.sendMessage(header+self.payload)
                    self.payload = ""
                    self.n_payload += 1
                    start_time = current_time

                cv2.imshow("Frame", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            except Exception as e:
                print(e)
                raise Exception("Problem while executing counter")

        self.camera.close()
        self.ethernetSender.close()
        # self.sender.close()
        cv2.destroyAllWindows()
