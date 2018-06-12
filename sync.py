
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
		w = 1280; h = 720
		self.resize(w, h)
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
			step = 200
			legend, = self.ax.plot(mp4[key].t[::step], mp4[key].x[::step], label=mp4['name'])
			lsLegend.append(legend)
		
		self.ax.legend(handles=lsLegend)
		self.ax.set_xlabel('t(sec)')
		self.canvas.draw()

	def sync(self):
		self.getTimeShift('raw','sync')
		self.keyPlot = 'sync'
		self.ax.clear()
		self.plot()
		self.edt.appendPlainText("Sync Done")
	

	def getTimeShift(self, keyIn, keyOut, nBatchSize = 8000000):
		# find base signal - longest one
		lsT = [mp4[keyIn].T for mp4 in self.lsMp4]
		tMax = max(lsT)
		idxBase = lsT.index(tMax)
		signalBase = self.lsMp4[idxBase][keyIn]
		# print signalBase.f # 48000.0
		
		wavBase = signalBase.x
		tBase = signalBase.t
		nBase = tBase.size
		
		numBatch = np.ceil(float(nBase)/nBatchSize).astype(int)
		numMp4 = len(self.lsMp4)
		npCorr = np.zeros((numBatch, numMp4, nBatchSize))
		
		for i in range(numBatch):
			print 'batch %d / %d'%(i,numBatch)
			idxS = i * nBatchSize
			idxE = min(idxS + nBatchSize, nBase)
			wavBaseBatch = wavBase[idxS:idxE]
			tBaseBatch = tBase[idxS:idxE]
			
			# FFT squared base signal
			# square is better for highlighting peaks

			
			wavBaseBatch = np.pad(wavBaseBatch, (0, nBatchSize-wavBaseBatch.size), 'constant')
			print 'FFT base', wavBaseBatch.size
			fftBaseBatch = scipy.fft(wavBaseBatch * wavBaseBatch)
			
			for j in range(numMp4):
				mp4 = self.lsMp4[j]
				# FFT squared signal
				# square is for highlighting peaks
				
				wav = np.interp(tBaseBatch, mp4[keyIn].t, mp4[keyIn].x, left=0, right=0)
				wav = np.pad(wav, (0, nBatchSize-wav.size), 'constant')
				print 'FFT ' + mp4['name'], wav.size
				fftWav = scipy.fft(wav * wav)
				
				# get correlation function based on FFT (conjugate of convolution)
				corr = np.abs(scipy.ifft(fftBaseBatch * scipy.conj(fftWav)))
				npCorr[i,j,:] = corr/corr.max() + 1.0
				# npCorr[i,j,:] = np.abs(scipy.ifft(fftBaseBatch * scipy.conj(fftWav)))
				
		
		npCorrProd = npCorr.prod(axis=0)
		idxPeak = np.argmax(npCorrProd, axis=1)
		idxPeak[np.where(idxPeak > nBatchSize/2)] = idxPeak[np.where(idxPeak > nBatchSize/2)] - nBatchSize
		
		# allign minimum shifts to zero
		sampleShift = idxPeak - min(idxPeak)

		for j in range(numMp4):
			self.lsMp4[j][keyOut] = self.lsMp4[j][keyIn].shiftSample(sampleShift[j])

		print 'Done'


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

			xmin, xmax = self.ax.get_xlim()
			ymin, ymax = self.ax.get_ylim()
			sx, sy = self.ax.get_size_inches()
			ax = (xmax - xmin)/sx
			ay = (ymax - ymin)/sy
			npA = np.array([ax,ay]).reshape(2,1)

			rx = (xmax - xmin)/10
			idxX = int(x[0][0]/self.plotted[0,-1] * self.plotted.shape[1])
			subset = self.plotted[:,idxX-rx:idxX+rx]
			
			npX = np.array(x[0]).reshape(2,1)
			diff = npA * (subset - npX)
			dist = diff[0]*diff[0] + diff[1]*diff[1]
			idxMin = np.argmin(dist)
			X = subsex[:,idxMin]
			
			
			# x[0][0]
			# x[0][1]

			# self.plotted
			# rect = patches.Rectangle((x[0][0]-1,-40000),3,80000, facecolor='r', ec='none', zorder=10)
			self.ax.add_patch(rect)
			# self.ax.plot(x[0][0],x[0][1],'go')
			self.ax.plot(X[0],X[1],'go')
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







