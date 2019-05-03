import sys
import os
import copy
import subprocess
import numpy as np
import scipy
import wave
import time
import cv2
import scipy.io.wavfile

fullpath_src = sys.argv[1]
fullpath_frames = sys.argv[2]
path_tgt = sys.argv[3]

frames = np.loadtxt(sys.argv[2])
print("frames : ", frames)

filename_src, ext_src = os.path.splitext(os.path.basename(fullpath_src))
if ext_src.lower() == ".mp4":
    cap = cv2.VideoCapture(fullpath_src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    capSize = (w, h)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    nFrame = 0
    ls_nFrameEnd = []
    for j in range(len(frames)):
        nFrameEnd = frames[j]
        strNum = '-%03d' % j
        strFilenameOutput = os.path.join(path_tgt, filename_src + strNum + ".mp4")
        print strFilenameOutput,
        out = cv2.VideoWriter(strFilenameOutput, fourcc, fps, capSize)

        while(cap.isOpened() and nFrame < nFrameEnd):
            ret, frame = cap.read()
            if ret == True:
                nFrame = nFrame + 1
                out.write(frame)
            else:
                break

        out.release()
        
        test = cv2.VideoCapture(strFilenameOutput)
        print test.get(cv2.CAP_PROP_FRAME_COUNT)

    j = len(frames)
    strNum = '-%03d' % j
    strFilenameOutput = os.path.join(path_tgt, filename_src + strNum + ".mp4")
    print strFilenameOutput,
    out = cv2.VideoWriter(strFilenameOutput, fourcc, fps, capSize)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            nFrame = nFrame + 1
            out.write(frame)
        else:
            break

    out.release()
    test = cv2.VideoCapture(strFilenameOutput)
    print test.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

elif ext_src.lower() == ".wav":
    rate, data = scipy.io.wavfile.read(fullpath_src)
    if len(data.shape) == 1:
        data = data[:,np.newaxis]
    s_frame = 0
    for j in range(len(frames)):
        f_frame = frames[j]
        strNum = '-%03d' % j
        strFilenameOutput = os.path.join(path_tgt, filename_src + strNum + ".wav")
        print(strFilenameOutput)

        s_secs = float(s_frame)/25
        f_secs = float(f_frame)/25

        data_output = data[int(s_secs * rate):int(f_secs * rate),:]
        scipy.io.wavfile.write(strFilenameOutput, rate, data_output)
        s_frame = f_frame

    
    strNum = '-%03d' % len(frames)
    strFilenameOutput = os.path.join(path_tgt, filename_src + strNum + ".wav")
    print(strFilenameOutput)

    s_secs = float(s_frame)/25

    data_output = data[int(s_secs * rate):,:]
    scipy.io.wavfile.write(strFilenameOutput, rate, data_output)
    
