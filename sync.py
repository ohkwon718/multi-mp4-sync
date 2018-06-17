
# from __future__ import print_function

import sys
import os
import copy
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy
import wave
import pyaudio
import struct

from PyQt4 import QtGui
from PyQt4.QtCore import QTimer, QEvent, Qt

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import random
import cv2

from util.signal import MySignal


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

		self.setLayout(layout)
		self.lsMp4 = []
		self.dictWav = {}
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
		self.keyPlot = 'wav'	
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
			sigWav = MySignal(x=wav, f = fr)
			mp4 = {'mp4-file':url, 'wav-file':strFileWav, 'wav':sigWav, 'name':strFilename}
			return mp4
		return

	def plot(self):
		key = self.keyPlot
		if key == None:
			return		
		self.ax.clear()

		lsLegend = []
		for mp4 in self.lsMp4:
			step = 100
			legend, = self.ax.plot(mp4[key].getTimeAxis()[::step], mp4[key].x[::step], label=mp4['name'])
			lsLegend.append(legend)
		
		self.ax.legend(handles=lsLegend)
		self.ax.set_xlabel('t(sec)')
		self.canvas.draw()

	def sync(self):
		self.getTimeShift()
		self.keyPlot = 'wav'
		self.plot()
		self.edt.appendPlainText("Sync Done")
	

	def getTimeShift(self, nChunkSize = 4000000):
		# find base signal - longest one
		lsT = [mp4['wav'].getTEnd() for mp4 in self.lsMp4]
		tMax = max(lsT)
		idxBase = lsT.index(tMax)
		signalBase = self.lsMp4[idxBase]['wav']
		# print signalBase.f # 48000.0
		
		wavBase = signalBase.x
		tBase = signalBase.getTimeAxis()
		nBase = signalBase.getLength()
		
		numChunk = np.ceil(float(nBase)/nChunkSize).astype(int)
		numMp4 = len(self.lsMp4)
		npCorrProd = np.ones((numMp4, nChunkSize))

		for i in range(numChunk):
			print 'Chunk %d / %d'%(i,numChunk)
			idxS = i * nChunkSize
			idxE = min(idxS + nChunkSize, nBase)
			wavBaseChunk = wavBase[idxS:idxE]
			tBaseChunk = tBase[idxS:idxE]
			
			# FFT squared base signal
			# square is better for highlighting peaks
			wavBaseChunk = np.pad(wavBaseChunk, (0, nChunkSize-wavBaseChunk.size), 'constant')
			print 'FFT base', wavBaseChunk.size
			fftBaseChunk = scipy.fft(wavBaseChunk * wavBaseChunk)
			
			for j in range(numMp4):
				mp4 = self.lsMp4[j]
				# FFT squared signal
				# square is for highlighting peaks
				
				wav = np.interp(tBaseChunk, mp4['wav'].getTimeAxis(), mp4['wav'].x, left=0, right=0)
				wav = np.pad(wav, (0, nChunkSize-wav.size), 'constant')
				print 'FFT ' + mp4['name'], wav.size
				fftWav = scipy.fft(wav * wav)
				
				# get correlation function based on FFT (conjugate of convolution)
				corr = np.abs(scipy.ifft(fftBaseChunk * scipy.conj(fftWav)))
				# add offset to reduce effects from zero corr
				npCorrProd[j] = npCorrProd[j] * (corr/(corr.max()+1) + 1.0)
		
		# npCorrProd = npCorr.prod(axis=0)
		idxPeak = np.argmax(npCorrProd, axis=1)
		idxPeak[np.where(idxPeak > nChunkSize/2)] = idxPeak[np.where(idxPeak > nChunkSize/2)] - nChunkSize
		
		# allign minimum shifts to zero
		sampleShift = idxPeak - min(idxPeak)

		for j in range(numMp4):
			self.lsMp4[j]['wav'].shiftSample(sampleShift[j])
		print 'Done'

	def fuse(self):
		fBase = 48000.0
		
		# find base signal - longest one
		lsT = [mp4['wav'].getTEnd() for mp4 in self.lsMp4]
		TMax = max(lsT)
		tBase = np.arange(0,TMax, 1.0/fBase)
		
		numMp4 = len(self.lsMp4)
		wavMean = np.zeros(tBase.size)
		for i in range(numMp4):
			wav = np.interp(tBase, self.lsMp4[i]['wav'].getTimeAxis(), self.lsMp4[i]['wav'].x, left=0, right=0)
			wavMean = wavMean + wav
		
		wavMean = (wavMean / numMp4).astype(int)
		self.dictWav['fuse'] = MySignal(x=wavMean, f = fBase) 
		self.ax.clear()
		self.ax.plot(tBase[::100], wavMean[::100], label='mean')
		self.ax.set_xlabel('t(sec)')
		self.canvas.draw()
		self.bClick = True
		self.edt.appendPlainText("Fuse Done")
		
	def click(self):
		if self.bClick:
			self.edt.appendPlainText("Click point")
			X_clicked = self.figure.ginput(1)[0]
			print X_clicked
			self.edt.appendPlainText(str(X_clicked))
			self.ax.set_ylim(self.ax.get_ylim()) 
			
			x_plotted = self.dictWav['fuse'].x
			t_plotted = self.dictWav['fuse'].getTimeAxis()
			lenPlotted = self.dictWav['fuse'].getLength()

			xmin, xmax = self.ax.get_xlim()
			ymin, ymax = self.ax.get_ylim()
			sx, sy = self.figure.get_size_inches()
			npScale = np.array([float(sx)/(xmax - xmin), float(sy)/(ymax - ymin)]).reshape(2,1)
			x_range = int(lenPlotted/10)
			idxX = int(X_clicked[0] / t_plotted[-1] * lenPlotted)
			idxFrom = max(idxX-x_range, 0)
			idxTo = min(idxX+x_range, lenPlotted-1)
			subset = np.vstack([t_plotted[idxFrom:idxTo], x_plotted[idxFrom:idxTo]])
			npX = np.array(X_clicked).reshape(2,1)
			diff = npScale * (subset - npX)
			dist = diff[0]*diff[0] + diff[1]*diff[1]
			print idxX-x_range, idxX+x_range
			print dist
			idxMin = np.argmin(dist)
			X = subset[:,idxMin]


			f = self.dictWav['fuse'].f

			p = pyaudio.PyAudio()
			stream = p.open(format = p.get_format_from_width(2), channels = 1, rate = int(f), output = True)
			play = subset[1,int(idxMin - 1.5 * f):int(idxMin + 1.5 * f)]

			chunk = 1024
			sig=play[0:chunk]

			inc = 0;
			data= 0;
			while data != '':
				data = struct.pack("%dh"%(len(sig)), *list(sig))   
				stream.write(data)
				inc=inc+chunk
				sig=play[inc:inc+chunk]
			 

			if QtGui.QMessageBox.question(self,'', "Is it the cutting point?", 
				QtGui.QMessageBox.Yes | QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes:

				rect = patches.Rectangle((X[0]-1.5,-40000),3,80000, facecolor='r', ec='none', zorder=10)
				self.ax.add_patch(rect)
				self.ax.plot(X[0],X[1],'go')
				self.canvas.draw()
				self.lsSplitPosition.append(X[0])
				self.edt.appendPlainText(" ".join(str(x) for x in self.lsSplitPosition))


	# def click(self):
	# 	if self.bClick:
	# 		self.edt.appendPlainText("Click point")
	# 		x, y = self.figure.ginput(1)[0]
	# 		self.edt.appendPlainText(str((x,y)))
	# 		self.ax.set_ylim(self.ax.get_ylim()) 
			
	# 		xPlotted = self.dictWav['fuse'].x
	# 		tPlotted = self.dictWav['fuse'].getTimeAxis()
	# 		lPlotted = self.dictWav['fuse'].getLength()

	# 		xmin, xmax = self.ax.get_xlim()
	# 		ymin, ymax = self.ax.get_ylim()
	# 		sx, sy = self.figure.get_size_inches()
	# 		ax = (xmax - xmin)/sx
	# 		ay = (ymax - ymin)/sy
	# 		npA = np.array([ax,ay]).reshape(2,1)
	# 		rx = int(lPlotted/20)
	# 		idxX = int(x / tPlotted[-1] * lPlotted)
	# 		# subset = self.plotted[:,idxX-rx:idxX+rx]
	# 		subset = np.vstack([tPlotted[idxX-rx:idxX+rx],xPlotted[idxX-rx:idxX+rx]])
			
	# 		npX = np.array([x,y]).reshape(2,1)
	# 		diff = 1.0/npA * (subset - npX)
	# 		dist = diff[0]*diff[0] + diff[1]*diff[1]
	# 		idxMin = np.argmin(dist)
	# 		X = subset[:,idxMin]


	# 		rect = patches.Rectangle((X[0]-1.5,-40000),3,80000, facecolor='r', ec='none', zorder=10)
	# 		f = (lPlotted - 1)/(tPlotted[-1] - tPlotted[0])
	# 		print f

	# 		# p = pyaudio.PyAudio()
	# 		# stream = p.open(format = p.get_format_from_width(2), channels = 1, rate = int(f), output = True)
	# 		# play = subset[:,int(idxMin - 1.5 * f):int(idxMin + 1.5 * f)]
	# 		# chunk = 1024
	# 		# sig=play[0:chunk]
	# 		# inc = 0;
	# 		# data=0;
	# 		# while data != '':
	# 		# 	data = struct.pack("%dh"%(len(sig)), *list(sig))   
	# 		# 	stream.write(data)
	# 		# 	inc=inc+chunk
	# 		# 	sig=signal[inc:inc+chunk]
    

	# 		self.ax.add_patch(rect)
	# 		# self.ax.plot(X[0],X[1],'go')
	# 		self.canvas.draw()
	# 		self.lsSplitPosition.append(X[0])
	# 		self.edt.appendPlainText(" ".join(str(x) for x in self.lsSplitPosition))


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
















