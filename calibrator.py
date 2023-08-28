import cv2
import numpy as np
from camera import Camera
import configparser

cameraIP = input("Camera ip:")
camera = Camera(cameraIP)
camera.open()

def onMouse(event, x, y, flags, params):
    global selected_field
    global slider_pos
    global color_config
    global trackbars_input_fields
    global trackbars_selected_field
    global stop
    if event == cv2.EVENT_LBUTTONDOWN:
        if 150 < x < 200 and 600 < y < 650: 
            stop = True
            
        elif 350 < x < 450 and 500 < y < 550:
            color_config = np.array([1, 61, 125, 180, 255, 255])
            slider_pos = (color_config / max_values_color * 255).astype(int)
            trackbars_input_fields = color_config.astype(str)
            
        elif 475 < x < 575 and 500 < y < 550:
            color_config = np.array([1, 40, 150, 180, 255, 255])
            slider_pos = (color_config / max_values_color * 255).astype(int)
            trackbars_input_fields = color_config.astype(str)
            
        for i, (startX, startY, endX, endY) in enumerate(input_boxes):
            if startX < x < endX and startY < y < endY:
                selected_field = i
                
        for i, (startX, startY, endX, endY) in enumerate(trackbars_input_boxes):
            if startX < x < endX and startY < y < endY:
                trackbars_selected_field = i
                
        for i, (startX, startY, endX, endY) in enumerate(trackbars):
            if startX <= x <= endX and startY <= y <= endY:
                slider_pos[i] = x - 400
                color_config = ((slider_pos / 255) * max_values_color).astype(int)
                trackbars_input_fields = color_config.astype(str)
            
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        for i, (startX, startY, endX, endY) in enumerate(trackbars):
            if startX <= x <= endX and startY <= y <= endY:
                slider_pos[i] = x - 400
                color_config = ((slider_pos / 255) * max_values_color).astype(int)
                trackbars_input_fields = color_config.astype(str)
                
                
def colorMask(color_config, frame):
    blur = cv2.GaussianBlur(frame, (11, 11), 1)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, color_config[:3], color_config[3:])
    return mask

def adjustMask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    mask = cv2.copyMakeBorder(mask, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
    return mask

def createTemplate(input_fields):
    template = np.zeros((2*int(input_fields[1]), 2*int(input_fields[1])), dtype=np.uint8)
    center_coordinates = (int(input_fields[1]), int(input_fields[1]))
    axes_length = (int(int(input_fields[1])/1.3), int(input_fields[1]))
    template=cv2.ellipse(img=template, center=center_coordinates, axes=axes_length, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
    dist_template = cv2.distanceTransform(template, cv2.DIST_L2, 5)
    norm_dist_template = cv2.normalize(dist_template, None, 0, 1.0, cv2.NORM_MINMAX)
    return norm_dist_template

def scaleImage(frame):
    width = int(frame.shape[1] * 25 / 100)
    height = int(frame.shape[0] * 25 / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def drawDetections(input_fields, result, res):
    loc = np.where(res >= int(input_fields[2])/100)

    boxes = [[int(pt[0]), int(pt[1]), int(2*int(input_fields[1])), int(2*int(input_fields[1]))] for pt in zip(*loc[::-1])]

    scores = res[loc]
            
    # Filter boxes (eliminate duplicate detections for the same object)
    indices = cv2.dnn.NMSBoxes(boxes, scores.tolist(), 0.5, 0.4)

    boxes = [boxes[i[0]] for i in indices]
            
    # Draw boxes
    for box in boxes:
        cv2.rectangle(result, (box[0]-5, box[1]-5), (box[0]-5+box[2], box[1]+box[3]-5), (0, 0, 255), 1)
        
def saveData(input_fields, trackbars_input_fields):    
    config = configparser.ConfigParser()

    config["SETTINGS"] = {
        'H_low': trackbars_input_fields[0],
        'S_low': trackbars_input_fields[1],
        'V_low': trackbars_input_fields[2],
        'H_high': trackbars_input_fields[3],
        'S_high': trackbars_input_fields[4],
        'V_high': trackbars_input_fields[5],
        'Threshold': input_fields[2],
        'Radius_template': int(int(input_fields[1]) / 2),
        'Zone_Start': input_fields[4],
        'Zone_End': input_fields[5],
        'N_zones': input_fields[3],
        'Roi_start': 10,
        'Roi_end': 80,
    }

    config["INFO"] = {
        'N_counter': input_fields[0],
    }

    with open("config.ini", 'w') as configfile:
        config.write(configfile)
                    
stop = False
                
input_fields = ["0", "30", "80", "3", "0", "100"]
selected_field = -1

input_boxes = [(170, 150 + i * 75, 270, 200 + i * 75) for i in range(6)]
input_labels = ["Counter Number", "Egg size", "Threshold (%)", "Zones", "Zone Start (%)", "Zone End (%)"]

slider_pos = np.array([0]*6)

color_config = np.array([0]*6) # default values
max_values_color = np.array([180, 255, 255, 180, 255, 255]) 

trackbars_selected_field = -1
trackbars = [(400, 150 + i * 50, 655, 170 + i * 50) for i in range(6)]
trackbars_labels = ["H_Low", "S_Low", "V_Low", "H_High", "S_High", "V_High"]
trackbars_input_fields = ["0"]*6
trackbars_input_boxes = [(675, 150 + i * 50, 725, 170 + i * 50) for i in range(6)]

# TODO:
#   - Calculate template radius etc through parameters

while camera.isOpened() and not stop:
    frame = camera.readFrame()
    frame = scaleImage(frame)
    result = frame.copy()
    
    control_panel = np.zeros((750, 750, 3), dtype="uint8")
    cv2.rectangle(control_panel, (0, 0), (750, 100), (255,255,255), -1)
    cv2.putText(control_panel, "CONFIGURATION", (150, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 10)
    
    cv2.rectangle(control_panel, (150, 600), (200, 650), (255, 255, 255), -1)
    cv2.putText(control_panel, "OK", (160, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Input boxes for measures
    for i, (startX, startY, endX, endY) in enumerate(input_boxes):
        cv2.rectangle(control_panel, (startX, startY), (endX, endY), (255, 255, 255), -1)
        cv2.putText(control_panel, input_fields[i], (startX + 10, startY + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(control_panel, input_labels[i], (startX - 130, startY + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Trackbars
    for i, (startX, startY, endX, endY) in enumerate(trackbars):
        cv2.rectangle(control_panel, (startX, startY), (endX, endY), (255, 255, 255), 2) 
        cv2.circle(control_panel, (startX + slider_pos[i], startY+10), 10, (0, 0, 255), -1)
        cv2.putText(control_panel, trackbars_labels[i], (startX - 80, startY + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
        
    # Inputs for trackbars
    for i, (startX, startY, endX, endY) in enumerate(trackbars_input_boxes):
        cv2.rectangle(control_panel, (startX, startY), (endX, endY), (255, 255, 255), -1)
        cv2.putText(control_panel, trackbars_input_fields[i], (startX+5, startY + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        
    cv2.rectangle(control_panel, (350, 500), (450, 550), (63,133,205), -1)
    cv2.putText(control_panel, "BROWN", (375, 530), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    
    cv2.rectangle(control_panel, (475, 500), (575, 550), (255,255,255), -1)
    cv2.putText(control_panel, "WHITE", (500, 530), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
    
    cv2.imshow("Control panel", control_panel)
    cv2.setMouseCallback("Control panel", onMouse)
    
    # Process image to see results
    mask = colorMask(color_config, frame)
    
    mask = adjustMask(mask)

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    norm_dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow("distance", norm_dist_transform)
    
    if input_fields[1] != "0" and input_fields[1] != "":
        norm_dist_template = createTemplate(input_fields)
        res = cv2.matchTemplate(norm_dist_transform, norm_dist_template, cv2.TM_CCORR_NORMED)
        
        if input_fields[2] != "0" and input_fields[2] != "":
            drawDetections(input_fields, result, res)
            
    width = frame.shape[1]
    height = frame.shape[0]
    
    if input_fields[3] != "" and input_fields[4] != "" and input_fields[5] != "":
        zone_start = width * int(input_fields[4])/100
        zone_end = width * int(input_fields[5])/100
        zone_width = (zone_end - zone_start) / int(input_fields[3])
        
        for zone in range(int(input_fields[3])+1):
            cv2.line(result, (int(zone_start + zone * zone_width), 0), (int(zone_start + zone * zone_width), height), (255,255,0), 1)

    # cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
    cv2.imshow("Result", result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # Exit
        break
    
    elif selected_field != -1 and chr(key).isdigit(): # Write in selected field
        input_fields[selected_field] += chr(key)
    
    elif key == 8 and selected_field != -1: # Backspace
        input_fields[selected_field] = input_fields[selected_field][:-1]
        
    elif trackbars_selected_field != -1 and chr(key).isdigit(): # Write in selected field
        trackbars_input_fields[trackbars_selected_field] += chr(key)
        color_config = np.array(trackbars_input_fields).astype(int)
        slider_pos = (color_config / max_values_color * 255).astype(int)
    
    elif key == 8 and trackbars_selected_field != -1: # Backspace
        trackbars_input_fields[trackbars_selected_field] = trackbars_input_fields[trackbars_selected_field][:-1]
        if trackbars_input_fields[trackbars_selected_field] != "":
            color_config = np.array(trackbars_input_fields).astype(int)
            slider_pos = (color_config / max_values_color * 255).astype(int)
        
    elif key == 13: # Enter
        selected_field = -1
        trackbars_selected_field = -1

print(trackbars_labels, trackbars_input_fields)
print(input_labels, input_fields)

saveData(input_fields, trackbars_input_fields)  

print("The new configuration has been saved correctly.")
cv2.destroyAllWindows()
camera.close()