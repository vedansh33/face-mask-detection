import numpy as np
import cv2

import torch

import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
from IPython import display

from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torchvision


model = torchvision.models.mobilenet_v2(pretrained=True)
classifier = nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, 2), nn.Sigmoid())
model.fc = classifier

state_dict = torch.load("D:/Downloads/face_mask_path.pth",map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
trans = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


labels = ['Mask', 'No Mask']
labelColor = [(10, 255, 0), (10, 0, 255)]

cap = cv2.VideoCapture(0)

# MTCNN for detecting the presence of faces
mtcnn = MTCNN(keep_all=True, device=device)


model.eval()
while True:
    ret, frame = cap.read()
    if ret == False:
        pass

    img_ = frame.copy()
    boxes, _ = mtcnn.detect(img_)
    # Using PIL to draw boxes
    '''frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)'''
    '''
    try:
        for x1,y1,x2,y2 in boxes:
            frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
            roi = img_[int(y1):int(y2) , int(x1):int(x2)]
    except TypeError as e:
        pass'''

    try:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            x1, y1 = max(x1, 0), max(y1, 0)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            face = img_[int(y1 - 30):int(y2 + 30), int(x1 - 30):int(x2 + 30)]

            in_img = trans(face)
            in_img = in_img.unsqueeze(0)


            out = model(in_img)
            prob = torch.exp(out)
            a = list(prob.squeeze())
            predicted = a.index(max(a))
            textSize = cv2.getTextSize(labels[predicted], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            textX = x1 + (x2 - x1) // 2 - textSize[0] // 2
            cv2.putText(frame, labels[predicted], (int(textX), y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        labelColor[predicted], 2)
    except (TypeError, ValueError) as e:
        pass

    cv2.imshow('Mask detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()





