#%%

import numpy as np
import cv2
import matplotlib.pyplot as plt
import subprocess
import os
import wave
import scipy.io.wavfile
from pydub import AudioSegment

		
#%%

nCamera = 11

target_fn = 800
target_file = 'result/181129_{}-019.mp4'.format(str(nCamera).zfill(2))
cap = cv2.VideoCapture(target_file)
cap.set(cv2.CAP_PROP_POS_FRAMES,target_fn)
target_nFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        target_frame = frame

plt.imshow(cv2.cvtColor(target_frame,cv2.COLOR_BGR2RGB))


#%%

filename = 'mp4/181129_{}.MP4'.format(str(nCamera).zfill(2))
cap = cv2.VideoCapture(filename)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
nFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(fps)
print(w,h)
print(nFrame)


cnt = 20000
cap.set(cv2.CAP_PROP_POS_FRAMES,cnt)
minErr = 99999999
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        err = np.linalg.norm(target_frame.astype(float)-frame.astype(float))
        if minErr > err:
            minErr = err
            minCnt = cnt
            minFrame = frame
            print minCnt, minErr

        if cnt % 100 == 0:
            print cnt,
            
        cnt = cnt+1
        if cnt > 25000:
            break
    else:
        print("something wrong")
        break


print("done")
cap.release()

print minCnt, minErr


#%%
filename = 'mp4/181129_{}.MP4'.format(str(nCamera).zfill(2))
cap = cv2.VideoCapture(filename)
cap.set(cv2.CAP_PROP_POS_FRAMES,minCnt)
fps = cap.get(cv2.CAP_PROP_FPS)

if(cap.isOpened()):
    ret, frame = cap.read()
        
plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
print minCnt

#%%

plt.imshow(cv2.cvtColor(target_frame,cv2.COLOR_BGR2RGB))

#%%

s_frame = minCnt - target_fn
s_secs = float(s_frame)/25
s_mins_ = int(s_secs / 60)
s_secs_ = s_secs - s_mins_*60
print s_mins_, s_secs_

f_secs = float(target_nFrame + s_frame)/25
f_mins_ = int(f_secs / 60)
f_secs_ = f_secs - f_mins_*60
print f_mins_, f_secs_

print s_secs, f_secs
print (f_secs - s_secs) * fps
# 129724.0

#%%

rate, data = scipy.io.wavfile.read('wav/181129_{}.wav'.format(str(nCamera).zfill(2)))
data = data[int(s_secs * rate):int(f_secs * rate),:]
scipy.io.wavfile.write('result/181129_{}-019.wav'.format(str(nCamera).zfill(2)),rate,data)



#%%
