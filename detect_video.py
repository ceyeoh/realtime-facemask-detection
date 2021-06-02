import numpy as np
import cv2 
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from PIL import Image
import time

import torch
import torch.nn as nn
from torchvision import transforms

def detect(frame, facenet, masknet):
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    facenet.setInput(blob)
    detections = facenet.forward()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    
    faces = []
    locs = []
    preds = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = transform(Image.fromarray(np.uint8(face))).unsqueeze(dim=0)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces)>0:
        stackface = torch.stack(faces).view(-1, 3, 224, 224)
        with torch.no_grad():
            out = masknet.forward(stackface)
        preds.extend(out)
        preds = [i.unsqueeze(dim=0) for i in preds]
    return (locs, preds)
    
prototxt = 'face_detector/deploy.prototxt'
weights = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
facenet = cv2.dnn.readNet(prototxt, weights)

masknet = torch.load('models/resnet18.pth', map_location=torch.device('cpu'))
masknet.eval()

vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    fps = FPS().start()
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    (locs, preds) = detect(frame, facenet, masknet)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        _, predicted = torch.max(pred.data, 1)
        label = 'Mask' if predicted.item()==0 else 'No Mask'
        color = (0, 200, 0) if label=='Mask' else (0, 0, 200)
        label = f'{label} {(nn.Softmax(1)(pred).squeeze()[predicted].item()*100):.2f}%'
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    fps.update()
    fps.stop()
    text = f'FPS: {fps.fps():.2f}'
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 0, 0), 2)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()