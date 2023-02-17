import cv2
import numpy as np
from urllib.request import urlopen
import time

prev_time = 0
FPS = 60

url = "http://192.168.0.199:8080/?action=stream" # Your video streaming url. don't try with this url!!
stream = urlopen(url)
buffer = b''


YOLO_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []
with open("sample.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

while True:
    buffer += stream.read(4096) # read by buffer 
    head = buffer.find(b'\xff\xd8')
    end = buffer.find(b'\xff\xd9')
    if head > -1 and end > -1:
        jpg = buffer[head:end+2]
        buffer = buffer[end+2:]
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        current_time = time.time() - prev_time
        # ret, frame = img

        if  (current_time > 1/ FPS):
            prev_time = time.time()


        h, w, c = img.shape

        # YOLO 입력
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0),
        True, crop=False)
        YOLO_net.setInput(blob)
        outs = YOLO_net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:

            for detection in out:

                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    dw = int(detection[2] * w)
                    dh = int(detection[3] * h)
                    # Rectangle coordinate
                    x = int(center_x - dw / 2)
                    y = int(center_y - dh / 2)
                    boxes.append([x, y, dw, dh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)
              
                
        
        cv2.imshow("YOLOv3", img)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                score = confidences[i]
                color = colors[i]
                
                # 경계상자와 클래스 정보 이미지에 입력
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                cv2.putText(img, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

            cv2.imshow("???",img)



      

    key = cv2.waitKey(1)
    if key == 27:
        # if you push the ESC key, 
        break

cv2.destroyAllWindows()
