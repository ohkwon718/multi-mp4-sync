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
		self.btnPlot.clicked.connect(self.plotOrig)

		self.btnSync = QtGui.QPushButton('Sync')
		self.btnSync.clicked.connect(self.sync)
		

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
		layout.addWidget(self.btnSync)
		# layout.addWidget(self.btnLoad)
		# layout.addWidget(self.btnPlay)
		# layout.addWidget(self.btnStop)
		self.setLayout(layout)
		self.lsMp4 = []

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
		self.plot(self.lsMp4)

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
			wav = wavfile.readframes(-1)
			wav = np.fromstring(wav, 'Int16')
			wav = wav.reshape(-1, numCh)
			wav = wav.mean(1)
			fr = float(wavfile.getframerate())
			lenSignal = wav.shape[0]
			tEnd = (lenSignal-1.0)/fr
			t = np.linspace(0, tEnd, num=lenSignal)
			mp4 = {'mp4-file':url, 'wav-file':strFileWav, 'wav-data':wav, 'time':t, 'time-end':tEnd,
					'framerate':fr, 'name':strFilename, 'padding':0, 'sample-shift':0, 'time-shift':0.}
			return mp4
		return

	def plotOrig(self):
		self.plot(self.lsMp4)

	def plot(self, lsMp4):
		ax = self.figure.add_subplot(111)
		ax.clear()
		lsLegend = []
		for mp4 in lsMp4:
			legend, = ax.plot(mp4['time'][::100],mp4['wav-data'][::100], label=mp4['name'])
			lsLegend.append(legend)
		ax.legend(handles=lsLegend)
		ax.set_xlabel('t(sec)')
		self.canvas.draw()

	def sync(self):
		lsMp4ZP = self.zeropadding(self.lsMp4)
		self.getTimeShift(lsMp4ZP)
		self.plot(lsMp4ZP)

	def zeropadding(self, lsMp4):
		# to avoid big prime number which lead very slow fft
		lsMp4New = copy.deepcopy(lsMp4)
		for mp4 in lsMp4New:
			lenWav = mp4['time'].size
			dec = len(str(lenWav))
			if dec > 2:
				lenTarget = (int(lenWav/np.power(10,dec-2))+1) * np.power(10,dec-2)
				tEndTarget = (lenTarget-1.0)/mp4['framerate']
				tNew = np.linspace(0, tEndTarget, num=lenTarget)
				wavNew = np.interp(tNew, mp4['time'], mp4['wav-data'], left=0, right=0)
				mp4['padding'] = lenTarget - lenWav
				mp4['time'] = tNew
				mp4['wav-data'] = wavNew
				mp4['time-end'] = tEndTarget
		return lsMp4New
	
	def getTimeShift(self, lsMp4):
		# find base signal - longest one
		lsTEnd = [mp4['time-end'] for mp4 in lsMp4]
		tEndMax = max(lsTEnd)
		idxBase = lsTEnd.index(tEndMax)
		wavBase = lsMp4[idxBase]['wav-data']

		# FFT squared base signal
		# square is for highlighting peaks
		print 'FFT base'
		fftBase = scipy.fft(wavBase * wavBase)
		tBase = lsMp4[idxBase]['time']
		

		lsTimeShift = []
		lsSampleShift = []
		for mp4 in lsMp4:
			# FFT squared signal
			# square is for highlighting peaks
			print 'FFT ' + mp4['name']
			wav = np.interp(tBase, mp4['time'], mp4['wav-data'], left=0, right=0)
			fftWav = scipy.fft(wav * wav)
			
			# get correlation function based on FFT (conjugate of convolution)
			corr = scipy.ifft(fftBase * scipy.conj(fftWav))

			# peak point of correlation
			idxPeak = np.argmax(np.abs(corr))
			mp4['sample-shift'] = idxPeak

			# for negative shift case
			if mp4['sample-shift'] > tBase.size/2:
				mp4['sample-shift'] = mp4['sample-shift'] - tBase.size
			lsSampleShift.append(mp4['sample-shift'])
		
		# allign minimum shifts to zero
		minSampleShift = min(lsSampleShift)
		for mp4 in lsMp4:
			mp4['sample-shift'] = mp4['sample-shift'] - minSampleShift
			mp4['time-shift'] = tBase[mp4['sample-shift']]

			# remove padding
			mp4['wav-data'] = mp4['wav-data'][:-mp4['padding']]
			mp4['time'] 	= mp4['time'][:-mp4['padding']]
			mp4['time-end'] = mp4['time'][-1]
			mp4['padding'] = 0

			# shift time without padding
			mp4['time'] = mp4['time'] + mp4['time-shift']
			
			# padding zero from 0 sec
			lenTarget = mp4['time'].size + mp4['sample-shift']
			tEndTarget = mp4['time-end'] + mp4['time-shift']
			tNew = np.linspace(0, tEndTarget, num = lenTarget)
			wavNew = np.interp(tNew, mp4['time'], mp4['wav-data'], left=0, right=0)

			mp4['time'] = tNew
			mp4['wav-data'] = wavNew
			mp4['time-end'] = tEndTarget
			print (tNew[1]-tNew[0])*mp4['framerate']
				
			


		


	# def sync(self):
	# 	self.zeropadding()
	# 	self.getTimeShift()
		# lsTEnd = [mp4['time-end'] for mp4 in self.lsMp4]
		# tEndMax = max(lsTEnd)
		# idxBase = lsTEnd.index(tEndMax)
		# idxBase = 3
		# wavBase = self.lsMp4[idxBase]['wav-data']
		# fsBase = self.lsMp4[idxBase]['framerate']
		# print 'FFT base'
		# print (wavBase * wavBase).shape
		# fftBase = scipy.fft(wavBase * wavBase)
		# print 'FFT base done'
		# tBase = self.lsMp4[idxBase]['time']
		
		# lsWav = []
		# for mp4 in self.lsMp4:
		# 	print 'FFT ' + mp4['name']
		# 	wav = np.interp(tBase, mp4['time'], mp4['wav-data'], left=0, right=0)
		# 	fftWav = scipy.fft(wav * wav)
		# 	corr = scipy.ifft(fftBase * scipy.conj(fftWav))
		# 	mp4['time-shift'] = tBase[np.argmax(np.abs(corr))]
		# 	if mp4['time-shift'] > tBase[-1]/2:
		# 		mp4['time-shift'] = mp4['time-shift'] - tBase.size/fsBase
		# 	print mp4['time-shift']
		# 	mp4['time'] = mp4['time'] + mp4['time-shift']	
		# self.plot()

	# def sync(self):
	# 	lsTEnd = [mp4['time-end'] for mp4 in self.lsMp4]
	# 	tEndMax = max(lsTEnd)
	# 	idxBase = lsTEnd.index(tEndMax)
	# 	wavBase = self.lsMp4[idxBase]['wav-data']

	# 	fftBase = scipy.fft(wavBase * wavBase)
	# 	tBase = self.lsMp4[idxBase]['time']
		
	# 	lsWav = []
	# 	for mp4 in self.lsMp4:
	# 		print 'FFT ' + mp4['name']
	# 		wav = np.interp(tBase, mp4['time'], mp4['wav-data'], left=0, right=0)
	# 		fftWav = scipy.fft(wav * wav)
	# 		corr = scipy.ifft(fftBase * scipy.conj(fftWav))
	# 		mp4['time-shift'] = tBase[np.argmax(np.abs(corr))]
	# 		mp4['time'] = mp4['time'] + mp4['time-shift']	
	# 	self.plot()

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







