from fastai.vision import *
from fastai.metrics import error_rate
import cv2
import threading


path = Path('./')
learn = load_learner(path)

camera = cv2.VideoCapture(0)

cv2.namedWindow("test")

while True:
    ret_val, img = camera.read()
    cv2.imshow("test", img)
    image = Image(pil2tensor(img, dtype=np.float32).div_(255))
    #print(image)
    _,_,outputs = learn.predict(image)
    output.normalise()
    gender = ("Male" if outputs[0]>= outputs[1] else "Female")
    print(f"Looks like they're {gender} - P(Male)={outputs[0]}, P(Female)={outputs[1]}")
    cv2.waitKey(33)
camera.release()
