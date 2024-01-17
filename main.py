import cv2
import mediapipe as mp
import time
import os
from ultralytics import YOLO
import moviepy.video.io.ImageSequenceClip
model = YOLO("yolov8n.pt")
class posedetectors():
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()


    def findpose(self,img,draw=True):
        imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgrgb)  

    def getposition(self,img,draw=True): 
     lmlist=[]   
     if self.results.pose_landmarks:  
       for id,lm in enumerate(self.results.pose_landmarks.landmark):
           h,w,c=img.shape
           cx,cy=int(lm.x*w),int(lm.y*h)
           #lmlist.append([id,cx,cy])
           lmlist.append([cx,cy])
           #cv2.circle(img,(cx,cy),3,(0,0,255),cv2.FILLED)
           #print(id,lm) 
     return lmlist

def avragex(listx):
           x=listx[0][0]+listx[3][0]+listx[6][0]+listx[9][0]+listx[10][0]
           return int(x/5)
def avragey(listx):
           y=listx[0][1]+listx[3][1]+listx[6][1]+listx[9][1]+listx[10][1]
           return int(y/5)  
       
       
def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def main():
    detector=posedetectors()
    cap=cv2.VideoCapture("face.mp4")
    while True: 
        success,img=cap.read() 
        results = predict(model,img)
        detector.findpose(img)
        lmlist=detector.getposition(img)
        cv2.circle(img,(avragex(lmlist),avragey(lmlist)),3,(0,0,255),cv2.FILLED) 
        cv2.imshow("image",img)
        if cv2.waitKey(1) == ord('q'):
            break
        
if __name__=="__main__":
    main()