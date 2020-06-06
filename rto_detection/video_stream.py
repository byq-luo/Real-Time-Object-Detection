
""" 
Implementing Real Time Object Detection and Counting
using YOLO and SORT(Simple Online and Realtime Tracker)
on Django 
"""
# import the necessary packages
import numpy as np
import imutils
import time
import cv2
import os
import pandas as pd
import json
from .sort import simple_sort 

data=[]
label_data =[]
vehicle_type =[]

def stream_video():
    tracker = simple_sort.Sort()
    memory = {}
     
    line_to_count = [(0, 600), (13200, 500)]
    counter = 0
    
    # Return true if line segments AB and CD intersect
    def intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    # load the COCO class labels  
    labelsPath = "yolo/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = "yolo/yolov3.weights"
    configPath =  "yolo/yolov3.cfg"

    # load YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNet(weightsPath, configPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    # set vs to cv2.VideoCapture(0) if you use your webcame 
    vs = cv2.VideoCapture("overpass.mp4")
    writer = None
    (W, H) = (None, None)

    frameIndex = 0

    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    # loop over frames from the video file stream
    while True:
    # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
       
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > float(0.5):
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        dets = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x+w, y+h, confidences[i]])
                

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)
        boxes = []
        indexIDs = []
        previous = memory.copy()
        memory = {}

        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                 
                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                cv2.rectangle(frame, (x, y), (w, h), color, 2)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                    p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                    # line to count as the vehicle passed
                    cv2.line(frame, line_to_count[0], line_to_count[1], (0, 0, 255), 2)
                    line_meet = intersect(p0, p1, line_to_count[0], line_to_count[1])
                        
                    if line_meet:
                        counter += 1
                        label_data.append(LABELS[classIDs[i]])
                        
                text = "{}".format(LABELS[classIDs[i]])
                
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                i += 1 

        # put rectangle with black baground for readability
        cv2.rectangle(frame, (0, 0), (1300, 200), (0,0,0), -1)
         # draw counter
        cv2.putText(frame, str("total car:"), (100,90), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2)
        cv2.putText(frame, str(counter), (430,90), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 2)
        cv2.putText(frame, str("vehicle type:"), (100,170), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
            
        if label_data:
            # put the type of vehicle on the frame
            cv2.putText(frame, str(label_data[counter-1]), (500,170), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 255), 2)
        
   
        # saves image file 
        # cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)
        
        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output", fourcc, 30,
                (frame.shape[1], frame.shape[0]), True)

            # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    elap * total))

        # increase frame index
        frameIndex += 1

        #maximum fram index
        if frameIndex >= total:
            print("[INFO] cleaning up...")
            writer.release()
            vs.release()
            exit()
        imgencode=cv2.imencode('.jpg',frame)[1]
        stringData=imgencode.tostring()
        # return generator object 
        yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
         
    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()
 
 


 

 


 
 

    
