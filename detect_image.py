import numpy as np
import cv2
import imutils
import argparse
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

def detect():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='path to input image')
    args = vars(ap.parse_args())
    
    prototxt = 'face_detector/deploy.prototxt'
    weights = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
    cvnet = cv2.dnn.readNet(prototxt, weights)

    net = torch.load('models/resnet18.pth', map_location=torch.device('cpu'))
    net.eval()

    image = cv2.imread(args['image'])
    h, w = image.shape[:2]
    if h>800:
        image = imutils.resize(image, height=800)
    elif w>1000:
        image = imutils.resize(image, width=1000)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    cvnet.setInput(blob)
    detections = cvnet.forward()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = transform(Image.fromarray(np.uint8(face))).unsqueeze(dim=0)
            with torch.no_grad():
                out = net.forward(face)
            _, predicted = torch.max(out.data, 1)
            label = 'Mask' if predicted.item()==0 else 'No Mask'
            color = (0, 255, 0) if label=='Mask' else (0, 0, 255)
            label = f'{label} {(nn.Softmax(1)(out).squeeze()[predicted].item()*100):.2f}%'
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    cv2.imshow("Output", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    detect()