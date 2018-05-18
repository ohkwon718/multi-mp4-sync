import subprocess


command = "ffmpeg -i ./mp4/test.mp4 -ab 160k -ac 2 -ar 44100 -vn test.wav"

subprocess.call(command, shell=True)

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys


spf = wave.open('./wav/test.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')

print signal
print signal.shape
print spf.getnchannels()

# #If Stereo
# if spf.getnchannels() == 2:
#     print 'Just mono files'
#     sys.exit(0)

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import wave

file = 'test.wav'

wav_file = wave.open(file,'r')
#Extract Raw Audio from Wav File
signal = wav_file.readframes(-1)
signal = np.fromstring(signal, 'Int16')

#Split the data into channels 
channels = [[] for channel in range(wav_file.getnchannels())]
for index, datum in enumerate(signal):
    channels[index%len(channels)].append(datum)

#Get time from indices
fs = wav_file.getframerate()
Time=np.linspace(0, len(signal)/len(channels)/fs, num=len(signal)/len(channels))

#Plot
plt.figure(1)
plt.title('Signal Wave...')
for channel in channels:
    plt.plot(Time,channel)
plt.show()	