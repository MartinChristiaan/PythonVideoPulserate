import numpy as np
import argparse
import cv2
import time
import os
import matplotlib.pyplot as plt
import scipy.io as sio
from util.opencv_util import *
from rPPG_Extracter import *
import csv
from rPPG_lukas_Extracter import *
data_path = "C:\\Users\\marti\\Downloads\\Data\\mixed_motion"
vl = VideoLoader(data_path + "\\bmp\\" )
fs = 20

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


with open(data_path + '\\reference.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
pulse_ref = np.array([float(row[1]) for row in data if is_number(row[1])])
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
#ex = rPPG_Lukas_Extracter()
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
old_frame,_,_ = vl.load_frame() 
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = np.zeros((4,1,2),dtype = np.float32)
p0[0,0,0] = 300
p0[0,0,1] = 300
p0[1,0,0] = 380
p0[1,0,1] = 280
p0[2,0,0] = 460
p0[2,0,1] = 300
p0[3,0,0] = 380
p0[3,0,1] = 140

#cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
print(p0.shape)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    frame,_,_ = vl.load_frame()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1
    good_old = p0
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        frame = cv2.line(frame, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    #img = cv2.add(frame,mask)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()





#mat_dict = {"rPPG" : rPPG,"ref_pulse_rate" : pulse_ref}
#sio.savemat("mixed_motion_CSK_FOREHEAD",mat_dict)

t = np.arange(rPPG.shape[1])/fs

plt.figure()
plt.plot(t,1/np.array(timestamps),'-r')
plt.xlabel("Time (sec)")
plt.ylabel("FPS")


plt.figure()
plt.plot(t,rPPG[0],'-r')
plt.plot(t,rPPG[1],'-g')
plt.plot(t,rPPG[2],'-b')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude ")

plt.figure()
plt.plot(t,pulse_ref[0:t.shape[0]],'-r')
plt.xlabel("Time (sec)")
plt.ylabel("Pulse_rate (BPM) ")
plt.grid(1)
plt.show()
