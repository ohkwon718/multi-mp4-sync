import sys
import os
import copy
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import scipy
import wave

from PyQt4 import QtGui
from PyQt4.QtCore import QTimer

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

		self.btnRemove = QtGui.QPushButton('Remove')
		self.btnRemove.clicked.connect(self.remove)


		self.listFile = QtGui.QListWidget()


		# set the layout
		# layout = QtGui.QVBoxLayout()
		layout = QtGui.QGridLayout()

		layout.addWidget(self.toolbar,0,0,1,3)
		layout.addWidget(self.canvas,1,0,1,3)
		layout.addWidget(self.btnSync,2,0)
		layout.addWidget(self.btnRemove,2,1)
		layout.addWidget(self.listFile,2,2)

		
		# layout.addWidget(self.btnLoad)
		# layout.addWidget(self.btnPlay)
		# layout.addWidget(self.btnStop)
		self.setLayout(layout)
		self.lsMp4 = []
		

	def remove(self):
		listItems=self.listFile.selectedItems()
		if not listItems: return        
		for item in listItems:
			self.listFile.takeItem(self.listFile.row(item))
			for mp4 in self.lsMp4:
				if mp4['name'] == item.text():
					self.lsMp4.remove(mp4)
					break
		print self.lsMp4
		self.plot()


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

	def plotOrig(self):
		self.keyPlot = 'raw'
		self.plot()

	def plot(self):
		key = self.keyPlot
		if key == None:
			return
		ax = self.figure.add_subplot(111)
		ax.clear()
		lsLegend = []
		for mp4 in self.lsMp4:
			legend, = ax.plot(mp4[key].t[::100], mp4[key].x[::100], label=mp4['name'])
			lsLegend.append(legend)
		ax.legend(handles=lsLegend)
		ax.set_xlabel('t(sec)')
		self.canvas.draw()

	def sync(self):
		for mp4 in self.lsMp4:
			mp4['zp'] = mp4['raw'].riseUnit()
		
		# self.getTimeShift(lsMp4ZP)
		# self.keyPlot = 'zp'
		self.plot()				
			
	# def getTimeShift(self, lsMp4):
	# 	# find base signal - longest one
	# 	lsTEnd = [mp4['time-end'] for mp4 in lsMp4]
	# 	tEndMax = max(lsTEnd)
	# 	idxBase = lsTEnd.index(tEndMax)
	# 	wavBase = lsMp4[idxBase]['wav-data']

	# 	# FFT squared base signal
	# 	# square is for highlighting peaks
	# 	print 'FFT base'
	# 	fftBase = scipy.fft(wavBase * wavBase)
	# 	tBase = lsMp4[idxBase]['time']
		

	# 	lsTimeShift = []
	# 	lsSampleShift = []
	# 	for mp4 in lsMp4:
	# 		# FFT squared signal
	# 		# square is for highlighting peaks
	# 		print 'FFT ' + mp4['name']
	# 		wav = np.interp(tBase, mp4['time'], mp4['wav-data'], left=0, right=0)
	# 		fftWav = scipy.fft(wav * wav)
			
	# 		# get correlation function based on FFT (conjugate of convolution)
	# 		corr = scipy.ifft(fftBase * scipy.conj(fftWav))

	# 		# peak point of correlation
	# 		idxPeak = np.argmax(np.abs(corr))
	# 		mp4['sample-shift'] = idxPeak

	# 		# for negative shift case
	# 		if mp4['sample-shift'] > tBase.size/2:
	# 			mp4['sample-shift'] = mp4['sample-shift'] - tBase.size
	# 		lsSampleShift.append(mp4['sample-shift'])
		
	# 	# allign minimum shifts to zero
	# 	minSampleShift = min(lsSampleShift)
	# 	for mp4 in lsMp4:
	# 		mp4['sample-shift'] = mp4['sample-shift'] - minSampleShift
	# 		mp4['time-shift'] = tBase[mp4['sample-shift']]

	# 		# remove padding
	# 		mp4['wav-data'] = mp4['wav-data'][:-mp4['padding']]
	# 		mp4['time'] 	= mp4['time'][:-mp4['padding']]
	# 		mp4['time-end'] = mp4['time'][-1]
	# 		mp4['padding'] = 0

	# 		# shift time without padding
	# 		mp4['time'] = mp4['time'] + mp4['time-shift']
			
	# 		# padding zero from 0 sec
	# 		lenTarget = mp4['time'].size + mp4['sample-shift']
	# 		tEndTarget = mp4['time-end'] + mp4['time-shift']
	# 		tNew = np.linspace(0, tEndTarget, num = lenTarget)
	# 		wavNew = np.interp(tNew, mp4['time'], mp4['wav-data'], left=0, right=0)

	# 		mp4['time'] = tNew
	# 		mp4['wav-data'] = wavNew
	# 		mp4['time-end'] = tEndTarget
	# 		print (tNew[1]-tNew[0])*mp4['framerate']
				
			

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







