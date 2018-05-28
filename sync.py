
# from __future__ import print_function

# import matplotlib.pyplot as plt
# import numpy as np
# t = np.arange(10)
# plt.plot(t, np.sin(t))
# print("Please click")
# x = plt.ginput(3)	
# print("clicked", x)
# plt.show()

import sys
import os
import copy
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy
import wave

from PyQt4 import QtGui
from PyQt4.QtCore import QTimer, QEvent, Qt

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import random
import cv2

from util.signal import Signal


class Window(QtGui.QDialog):
	def __init__(self, parent=None):
		super(Window, self).__init__(parent)
		self.setWindowTitle("Multi-mp4-Sync")
		self.setAcceptDrops(True)

		self.figure = Figure()
		self.canvas = FigureCanvas(self.figure)
		self.toolbar = NavigationToolbar(self.canvas, self)

		# Just some button connected to `plot` method
		self.btnSync = QtGui.QPushButton('Sync')
		self.btnSync.clicked.connect(self.sync)
		
		# self.btnSync = QtGui.QPushButton('Segmentation')
		# self.btnSync.clicked.connect(self.Segmentation)

		self.btnPlay = QtGui.QPushButton('play')
		self.btnPlay.clicked.connect(self.play)
		
		self.btnStop = QtGui.QPushButton('stop')
		self.btnStop.clicked.connect(self.stop)
	
		self.btnLoad = QtGui.QPushButton('Load')
		self.btnLoad.clicked.connect(self.load)

		self.btnFuse = QtGui.QPushButton('Fuse')
		self.btnFuse.clicked.connect(self.fuse)

		self.btnClick = QtGui.QPushButton('Click')
		self.btnClick.clicked.connect(self.click)

		
		self.edt = QtGui.QPlainTextEdit()
		self.edt.setDisabled(True)
		self.edt.setMaximumBlockCount(10)
		
			
		
		

		self.listFile = QtGui.QListWidget()
		self.listFile.installEventFilter(self)
		self.listFile.setFixedWidth(100)
		
		
		

		layout = QtGui.QGridLayout()

		layout.addWidget(self.toolbar,0,0,1,3)
		layout.addWidget(self.canvas,1,0,1,3)
		layout.addWidget(self.btnSync,2,0,1,1)
		layout.addWidget(self.btnFuse,3,0,1,1)
		layout.addWidget(self.btnClick,4,0,1,1)
		layout.addWidget(self.listFile,2,1,3,1)
		layout.addWidget(self.edt,2,2,3,1)


		# layout.addWidget(self.btnLoad)
		# layout.addWidget(self.btnPlay)
		# layout.addWidget(self.btnStop)
		self.setLayout(layout)
		self.lsMp4 = []
		self.bClick = False
		self.lsSplitPosition = []
		self.ax = self.figure.add_subplot(111)
		
	def eventFilter(self, obj, event):
		if event.type() == QEvent.KeyPress and obj == self.listFile:
			if event.key() == Qt.Key_Delete:
				listItems=self.listFile.selectedItems()
				if not listItems: return        
				for item in listItems:
					self.listFile.takeItem(self.listFile.row(item))
					for mp4 in self.lsMp4:
						if mp4['name'] == item.text():
							self.lsMp4.remove(mp4)
							break
				self.plot()			
			return super(Window, self).eventFilter(obj, event)
		else:
			return super(Window, self).eventFilter(obj, event)

	def dragEnterEvent(self, event):
		if event.mimeData().hasUrls():
			event.accept()
		else:
			event.ignore()

	def dropEvent(self, event):
		lsUrl = [unicode(u.toLocalFile()) for u in event.mimeData().urls()]
		for url in lsUrl:
			mp4 = self.loadMp4(url)
			if mp4:
				self.lsMp4.append(mp4)
				item = QtGui.QListWidgetItem(mp4['name'])
				self.listFile .addItem(item)
		self.keyPlot = 'raw'	
		self.plot()


	def loadMp4(self, url):
		if url in [mp4['mp4-file'] for mp4 in self.lsMp4]:
			return
		strBase = os.path.basename(url)
		strFilename, strExtension = os.path.splitext(strBase)
		if strExtension.lower() != ".mp4":
			return
		strFileWav = os.path.join("wav", strFilename + ".wav")
		command = "ffmpeg -n -i " + url + " -ac 1 -vn "+ strFileWav
		# subprocess.call(command, shell=True)
		if os.path.isfile(strFileWav):
			wavfile = wave.open(strFileWav,'r')
			numCh = wavfile.getnchannels()
			wav = np.fromstring( wavfile.readframes(-1) , 'Int16' ).reshape(-1, numCh).mean(1)
			fr = float(wavfile.getframerate())
			raw = Signal(x=wav, f = fr)
			mp4 = {'mp4-file':url, 'wav-file':strFileWav, 'raw':raw, 'name':strFilename}
			return mp4
		return

	def plot(self):
		key = self.keyPlot
		if key == None:
			return		
		self.ax.clear()

		lsLegend = []
		for mp4 in self.lsMp4:
			legend, = self.ax.plot(mp4[key].t[::100], mp4[key].x[::100], label=mp4['name'])
			lsLegend.append(legend)
		
		self.ax.legend(handles=lsLegend)
		self.ax.set_xlabel('t(sec)')
		self.canvas.draw()
		

	def sync(self):
		# rise unit to avoid big prime number which leads very slow fft
		for mp4 in self.lsMp4:
			mp4['zp'] = mp4['raw'].riseUnit()
		self.getTimeShift('zp','sync')
		for mp4 in self.lsMp4:
			mp4['sync'] = mp4['sync'].removePadding()
		self.keyPlot = 'sync'
		self.plot()				
	

	def getTimeShift(self, keyIn, keyOut):
		# find base signal - longest one
		lsT = [mp4[keyIn].T for mp4 in self.lsMp4]
		TMax = max(lsT)
		idxBase = lsT.index(TMax)
		signalBase = self.lsMp4[idxBase][keyIn]
		wavBase = signalBase.x
		tBase = signalBase.t

		# FFT squared base signal
		# square is for highlighting peaks
		print 'FFT base'
		fftBase = scipy.fft(wavBase * wavBase)

		lsTimeShift = []
		lsSampleShift = []
		for mp4 in self.lsMp4:
			# FFT squared signal
			# square is for highlighting peaks
			print 'FFT ' + mp4['name']
			wav = np.interp(tBase, mp4[keyIn].t, mp4[keyIn].x, left=0, right=0)
			fftWav = scipy.fft(wav * wav)
			
			# get correlation function based on FFT (conjugate of convolution)
			corr = scipy.ifft(fftBase * scipy.conj(fftWav))

			# peak point of correlation
			idxPeak = np.argmax(np.abs(corr))
			mp4['nShift'] = idxPeak

			# for negative shift case
			if mp4['nShift'] > tBase.size/2:
				mp4['nShift'] = mp4['nShift'] - tBase.size
			lsSampleShift.append(mp4['nShift'])
		
		# allign minimum shifts to zero
		minSampleShift = min(lsSampleShift)

		for mp4 in self.lsMp4:
			mp4['nShift'] = mp4['nShift'] - minSampleShift
			mp4[keyOut] = mp4[keyIn].shiftSample(mp4['nShift'])
			del mp4['nShift']

	def fuse(self):
		# find base signal - longest one
		lsT = [mp4['sync'].T for mp4 in self.lsMp4]
		TMax = max(lsT)
		idxBase = lsT.index(TMax)
		signalBase = self.lsMp4[idxBase]['sync']
		wavBase = signalBase.x
		tBase = signalBase.t

		lsWav = []
		for mp4 in self.lsMp4:
			wav = np.interp(tBase, mp4['sync'].t, mp4['sync'].x, left=0, right=0)
			lsWav.append(wav)
		lsWav = np.array(lsWav)
		wavMul = np.sum(lsWav, axis=0)
		
		# ax = self.figure.add_subplot(111)
		self.ax.clear()
		lsLegend = []
		self.ax.plot(tBase, wavMul, label='multiplied')

		self.ax.set_xlabel('t(sec)')
		self.canvas.draw()
		self.bClick = True
		self.plotted = np.array([mp4['sync'].t, mp4['sync'].x])

	def click(self):
		if self.bClick:
			self.edt.appendPlainText("Click point")
			x = self.figure.ginput(1)
			self.edt.appendPlainText(str(x))

			self.ax.set_ylim(self.ax.get_ylim()) 
			rect = patches.Rectangle((x[0][0]-1,-40000),3,80000, facecolor='r', ec='none', zorder=10)
			self.ax.add_patch(rect)
			self.ax.plot(x[0][0],x[0][1],'go')
			self.canvas.draw()
			self.lsSplitPosition.append(x[0][0])
			self.edt.appendPlainText(" ".join(str(x) for x in self.lsSplitPosition))

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
		    ret, frame = self.cap.read()
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







