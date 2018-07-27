
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

		self.btnGenerate = QtGui.QPushButton('Generate')
		self.btnGenerate.clicked.connect(self.generate)
		
		self.edt = QtGui.QPlainTextEdit()
		self.edt.setDisabled(True)
		self.edt.setMaximumBlockCount(10)
		
			
		self.listFile = QtGui.QListWidget()
		self.listFile.installEventFilter(self)
		self.listFile.setFixedWidth(100)
		

		self.menuBar = QtGui.QMenuBar(self)		
		fileMenu = self.menuBar.addMenu('File')
		exit_action = QtGui.QAction('Exit', self)
		exit_action.triggered.connect(exit)
		fileMenu.addAction(exit_action)
		# editMenu = mainMenu.addMenu('Edit')
		# viewMenu = mainMenu.addMenu('View')
		# searchMenu = mainMenu.addMenu('Search')
		# toolsMenu = mainMenu.addMenu('Tools')
		# helpMenu = mainMenu.addMenu('Help')

		layout = QtGui.QGridLayout()

		layout.addWidget(self.menuBar,0,0,1,3)
		layout.addWidget(self.toolbar,1,0,1,3)
		layout.addWidget(self.canvas,2,0,1,3)
		layout.addWidget(self.btnSync,3,0,1,1)
		layout.addWidget(self.btnFuse,4,0,1,1)
		layout.addWidget(self.btnClick,5,0,1,1)
		layout.addWidget(self.btnGenerate,6,0,1,1)
		layout.addWidget(self.listFile,3,1,4,1)
		layout.addWidget(self.edt,3,2,4,1)
		layout.move(100,100)
		# layout.addWidget(self.toolbar,0,0,1,3)
		# layout.addWidget(self.canvas,1,0,1,3)
		# layout.addWidget(self.btnSync,2,0,1,1)
		# layout.addWidget(self.btnFuse,3,0,1,1)
		# layout.addWidget(self.btnClick,4,0,1,1)
		# layout.addWidget(self.btnGenerate,5,0,1,1)
		# layout.addWidget(self.listFile,2,1,4,1)
		# layout.addWidget(self.edt,2,2,4,1)

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
		sys.stdout.write('\a')
		sys.stdout.flush()

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
		for j in range(numMp4):
			self.lsMp4[j]['time-shift'] = self.lsMp4[j]['wav'].t0
			self.lsMp4[j]['time-end'] = self.lsMp4[j]['wav'].getTEnd()
			del self.lsMp4[j]['wav']
		sys.stdout.write('\a')
		sys.stdout.flush()	

	def click(self):
		if self.bClick:
			X, play = self.getClickedPoint()
			self.playSound(play, f = self.dictWav['fuse'].f)
			
			if QtGui.QMessageBox.question(self,'', "Is it the cutting point?", 
				QtGui.QMessageBox.Yes | QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes:

				rect = patches.Rectangle((X[0]-1.5,-40000),3,80000, facecolor='r', ec='none', zorder=10)
				self.ax.add_patch(rect)
				self.ax.plot(X[0],X[1],'go')
				self.canvas.draw()
				self.lsSplitPosition.append(X[0])
				self.lsSplitPosition.sort()
				self.edt.appendPlainText(" ".join(str(x) for x in self.lsSplitPosition))


	def getClickedPoint(self):
		self.ax.set_xlim(self.ax.get_xlim()) 
		self.ax.set_ylim(self.ax.get_ylim()) 

		self.edt.appendPlainText("Click point")
		X_clicked = self.figure.ginput(1)[0]
		self.edt.appendPlainText(str(X_clicked))

		x_plotted = self.dictWav['fuse'].x
		t_plotted = self.dictWav['fuse'].getTimeAxis()
		len_plotted = self.dictWav['fuse'].getLength()

		xmin, xmax = self.ax.get_xlim()
		ymin, ymax = self.ax.get_ylim()
		sx, sy = self.figure.get_size_inches()
		npScale = np.array([float(sx)/(xmax - xmin), float(sy)/(ymax - ymin)]).reshape(2,1)

		x_range = int(len_plotted/10)
		idxX = int(X_clicked[0] / t_plotted[-1] * len_plotted)

		idxFrom = max(idxX-x_range, 0)
		idxTo = min(idxX+x_range, len_plotted-1)
		subset = np.vstack([t_plotted[idxFrom:idxTo], x_plotted[idxFrom:idxTo]])
		npX = np.array(X_clicked).reshape(2,1)
		diff = npScale * (subset - npX)
		dist = diff[0]*diff[0] + diff[1]*diff[1]
		
		idxMin = np.argmin(dist)
		X = subset[:,idxMin]
		f_plotted = self.dictWav['fuse'].f
		play = subset[1,int(idxMin - 1.5 * f_plotted):int(idxMin + 1.5 * f_plotted)]
		return X, play

	def playSound(self, play, f):
		p = pyaudio.PyAudio()
		stream = p.open(format = p.get_format_from_width(2), channels = 1, rate = int(f), output = True)
		nChunkSize = 1024
		
		numChunk = np.ceil(float(play.size)/nChunkSize).astype(int)
		for i in range(numChunk):
			idxS = i * nChunkSize
			idxE = min(idxS + nChunkSize, play.size)
			wavChunk = play[idxS:idxE]
			data = struct.pack("%dh"%(len(wavChunk)), *list(wavChunk))   
			stream.write(data)
	
	def generate(self):
		self.generateSegmentedVideos()



	def generateSegmentedVideos(self):
		tEndMax = max([mp4['time-end'] for mp4 in self.lsMp4])
		for mp4 in self.lsMp4:
			cap = cv2.VideoCapture(mp4['mp4-file'])
			fps = cap.get(cv2.CAP_PROP_FPS)
			w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
			h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
			capSize = (w, h)
			fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
			strFilename, strExtension = os.path.splitext(os.path.basename(mp4['mp4-file']))

			nFrame = 0
			
			for j in range(len(self.lsSplitPosition)):
				t = self.lsSplitPosition[j]
				# nFrameEnd = int(fps * (t - mp4['time-shift']))
				nFrameEnd = int(fps * t) - int(fps * mp4['time-shift'])
				
				strNum = '-%03d' % j
				strFilenameOutput = os.path.join("result", strFilename + strNum + ".mp4")
				print strFilenameOutput,
				out = cv2.VideoWriter(strFilenameOutput, fourcc, fps, capSize)

				if j == 0:
					# print 'black %d frames'%np.ceil(fps * mp4['time-shift'])
					# for _ in range(np.ceil(fps * mp4['time-shift'])):
					print 'black %d frames'%int(fps * mp4['time-shift']),
					for _ in range(int(fps * mp4['time-shift'])):
						out.write(np.zeros((h,w,3), np.uint8))
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
			
			j = len(self.lsSplitPosition)
			strNum = '-%03d' % j
			strFilenameOutput = os.path.join("result", strFilename + strNum + ".mp4")
			print strFilenameOutput,
			out = cv2.VideoWriter(strFilenameOutput, fourcc, fps, capSize)
			
			while(cap.isOpened()):
				ret, frame = cap.read()
				if ret == True:
					nFrame = nFrame + 1
					out.write(frame)
				else:
					break
			print 'black %d frames'%int(fps * mp4['time-shift']),
			for _ in range( int(np.ceil(fps * tEndMax)) - ( nFrame + int(fps * mp4['time-shift']) ) ):
				out.write(np.zeros((h,w,3), np.uint8))
			out.release()
			test = cv2.VideoCapture(strFilenameOutput)
			print test.get(cv2.CAP_PROP_FRAME_COUNT)
			cap.release()

		sys.stdout.write('\a')
		sys.stdout.flush()
				

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
















