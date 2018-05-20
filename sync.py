import sys
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import wave

from PyQt4 import QtGui
from PyQt4.QtCore import QTimer

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import random
import cv2



class Window(QtGui.QDialog):
	def __init__(self, parent=None):
		super(Window, self).__init__(parent)
		self.setWindowTitle("Multi-mp4-Sync")
		self.setAcceptDrops(True)

		self.figure = Figure()
		self.canvas = FigureCanvas(self.figure)
		self.toolbar = NavigationToolbar(self.canvas, self)

		# Just some button connected to `plot` method
		self.btnPlot = QtGui.QPushButton('Plot')
		self.btnPlot.clicked.connect(self.plot)

		self.btnPlay = QtGui.QPushButton('play')
		self.btnPlay.clicked.connect(self.play)
		
		self.btnStop = QtGui.QPushButton('stop')
		self.btnStop.clicked.connect(self.stop)
	
		self.btnLoad = QtGui.QPushButton('Load')
		self.btnLoad.clicked.connect(self.load)
				

		# set the layout
		layout = QtGui.QVBoxLayout()
		layout.addWidget(self.toolbar)
		layout.addWidget(self.canvas)
		layout.addWidget(self.btnPlot)
		layout.addWidget(self.btnLoad)
		layout.addWidget(self.btnPlay)
		layout.addWidget(self.btnStop)
		self.setLayout(layout)

		self.lsFileMp4 = []
		self.lsFileWav = []
		self.lsDataWav = []

	def dragEnterEvent(self, event):
		if event.mimeData().hasUrls():
			event.accept()
		else:
			event.ignore()

	def dropEvent(self, event):
		lsUrl = [unicode(u.toLocalFile()) for u in event.mimeData().urls()]
		for url in lsUrl:
			self.loadMp4(url)
		
	def loadMp4(self, url):
		if url in self.lsFileMp4:
			return
		strBase = os.path.basename(url)
		strFilename, strExtension = os.path.splitext(strBase)
		if strExtension.lower() != ".mp4":
			return
		strFileWav = "./wav/" + strFilename + ".wav"
		command = "ffmpeg -y -i " + url + " -ac 1 -vn "+ strFileWav
		subprocess.call(command, shell=True)
		if os.path.isfile(strFileWav):

			self.lsFileMp4.append(url)
			self.lsFileWav.append(strFileWav)

	def plot(self):
		ax = self.figure.add_subplot(111)
		ax.clear()
		for strFileWav in self.lsFileWav:
			wav = wave.open(strFileWav,'r')
			numCh = wav.getnchannels()
			signal = wav.readframes(-1)
			signal = np.fromstring(signal, 'Int16')
			signal = signal.reshape(-1, numCh)
			fr = wav.getframerate()
			print fr
			numSignal = signal.shape[0]
			tEnd = numSignal/fr
			Time = np.linspace(0, tEnd, num=numSignal)
			
			ax.plot(Time,signal)
		self.canvas.draw()

	# def plot(self):
	# 	data = [random.random() for i in range(10)]
	# 	ax = self.figure.add_subplot(111)
	# 	ax.clear()
	# 	ax.plot(data, '*-')
	# 	self.canvas.draw()

	def play(self):
		self.timer = QTimer()
		self.timer.timeout.connect(self.tick)
		self.timer.start(50)


	def stop(self):
		self.timer.stop()

	def load(self):
		self.cap = cv2.VideoCapture('mp4/cam4.MP4')
		nFrame = 0
		while(cap.isOpened()):
		    ret, frame = cap.read()
		    if ret == True:
		    	print "frame ",nFrame
		    	nFrame = nFrame + 1
		    	cv2.imshow('Frame',frame)
		     
		        if cv2.waitKey(25) & 0xFF == ord('q'):
		           break

		    else: 
		        break

	def tick(self):
		print 't'


if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)

	main = Window()
	main.show()

	sys.exit(app.exec_())


# # import subprocess
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import wave
# # import sys

# # command = "ffmpeg -y -i ./mp4/test.mp4 -ac 1 -vn ./wav/test.wav"
# # subprocess.call(command, shell=True)

# # filename = './wav/test.wav'

# # wav_file = wave.open(filename,'r')
# # numCh = wav_file.getnchannels()
# # #Extract Raw Audio from Wav File
# # signal = wav_file.readframes(-1)
# # signal = np.fromstring(signal, 'Int16')
# # signal = signal.reshape(-1, numCh)

# # #Get time from indices
# # fr = wav_file.getframerate()
# # numSignal = signal.shape[0]
# # tEnd = numSignal/fr
# # Time = np.linspace(0, tEnd, num=numSignal)

# # #Plot
# # plt.figure(1)
# # plt.title('Signal Wave...')
# # plt.plot(Time,signal)
# # plt.show()	








# # import sys
# # from PyQt4.QtCore import *
# # from PyQt4.QtGui import *

# # class sliderdemo(QWidget):
# #    def __init__(self, parent = None):
# #       super(sliderdemo, self).__init__(parent)

# #       layout = QVBoxLayout()
# #       self.l1 = QLabel("Hello")
# #       self.l1.setAlignment(Qt.AlignCenter)
# #       layout.addWidget(self.l1)
		
# #       self.sl = QSlider(Qt.Horizontal)
# #       self.sl.setMinimum(10)
# #       self.sl.setMaximum(30)
# #       self.sl.setValue(20)
# #       self.sl.setTickPosition(QSlider.TicksBelow)
# #       self.sl.setTickInterval(5)
		
# #       layout.addWidget(self.sl)
# #       self.sl.valueChanged.connect(self.valuechange)
# #       self.setLayout(layout)
# #       self.setWindowTitle("SpinBox demo")

# #    def valuechange(self):
# #       size = self.sl.value()
# #       self.l1.setFont(QFont("Arial",size))
		
# # def main():
# #    app = QApplication(sys.argv)
# #    ex = sliderdemo()
# #    ex.show()
# #    sys.exit(app.exec_())
	
# # if __name__ == '__main__':
# #    main()






# # import sys
# # from PyQt4 import QtCore, QtGui
# # from PyQt4.phonon import Phonon
# # app = QtGui.QApplication(sys.argv)
# # vp = Phonon.VideoPlayer()
# # media = Phonon.MediaSource('./mp4/test.mp4')
# # vp.load(media)
# # vp.play()
# # vp.show()
# # sys.exit(app.exec_())



