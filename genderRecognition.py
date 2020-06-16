from fastai.vision import *
from fastai.metrics import error_rate
import cv2
import threading

path = Path('./')
learn = load_learner(path)

camera = cv2.VideoCapture(0)

cv2.namedWindow("test")

while True:
    _, img = camera.read()
    cv2.imshow("test", img)
    img = Image(torch.from_numpy(img))
    _,_,outputs = learn.predict(img)
    gender = ("Male" if outputs[0]>= outputs[1] else "Female")
    print(f"Looks like they're {gender} - P(Male)={round(outputs[0],3)}, P(Female)={round(outputs[1],3)}")
camera.release()
