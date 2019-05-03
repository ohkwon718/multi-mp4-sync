
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
import time

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

		self.menuBar = QtGui.QMenuBar(self)		
		self.menuBar.setNativeMenuBar(False)
		menuFile = self.menuBar.addMenu('File')

		actOpen = QtGui.QAction('Open', self)
		actOpen.setShortcut("Ctrl+O")
		actOpen.triggered.connect(self.openFiles)
		menuFile.addAction(actOpen)

		actExit = QtGui.QAction('Exit', self)
		actExit.setShortcut("Ctrl+Q")
		actExit.triggered.connect(exit)
		menuFile.addAction(actExit)

		self.figure = Figure()
		self.canvas = FigureCanvas(self.figure)
		self.toolbar = NavigationToolbar(self.canvas, self)

		self.btnSync = QtGui.QPushButton('Sync')
		self.btnSync.clicked.connect(self.sync)

		self.btnFuse = QtGui.QPushButton('Fuse')
		self.btnFuse.clicked.connect(self.fuse)

		self.btnClick = QtGui.QPushButton('Click')
		self.btnClick.clicked.connect(self.click)

		self.btnGenerate = QtGui.QPushButton('Generate')
		self.btnGenerate.clicked.connect(self.generate)

		self.cbBlank = QtGui.QCheckBox("Insert Blank")

		layoutControl = QtGui.QGridLayout()
		layoutControl.addWidget(self.btnSync,0,0,1,1)
		layoutControl.addWidget(self.btnFuse,1,0,1,1)
		layoutControl.addWidget(self.btnClick,2,0,1,1)
		layoutControl.addWidget(self.btnGenerate,3,0,1,1)
		layoutControl.addWidget(self.cbBlank,4,0,1,1)
		
		self.edt = QtGui.QPlainTextEdit()
		self.edt.setDisabled(True)
		self.edt.setMaximumBlockCount(10)
					
		self.listFile = QtGui.QListWidget()
		self.listFile.installEventFilter(self)
		self.listFile.setFixedWidth(100)

		layout = QtGui.QGridLayout()
		layout.addWidget(self.menuBar,0,0,1,3)
		layout.addWidget(self.toolbar,1,0,1,3)
		layout.addWidget(self.canvas,2,0,1,3)
		layout.addLayout(layoutControl,3,0,1,1)
		layout.addWidget(self.listFile,3,1,1,1)
		layout.addWidget(self.edt,3,2,1,1)

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


	def openFiles(self):
		dlg = QtGui.QFileDialog()
		dlg.setFileMode(QtGui.QFileDialog.ExistingFiles)
		dlg.setDirectory(os.getcwd())
		dlg.setFilter("Text files (*.mp4)")
		

		if dlg.exec_():
			lsUrl = dlg.selectedFiles()
			for url in lsUrl:
				mp4 = self.loadMp4(str(url))
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
			
			sec_cut = 3000
			fr = float(wavfile.getframerate())
			wav = np.fromstring( wavfile.readframes(-1) , 'Int16' )
			t_end = wav.size / (numCh * fr)
			wav = wav[:int(numCh * sec_cut * fr)].reshape(-1, numCh).mean(1)
			
			sigWav = MySignal(x=wav, f = fr)
			mp4 = {'mp4-file':url, 'wav-file':strFileWav, 'wav':sigWav, 'name':strFilename, 'time-end':t_end}
			print url, "loaded"
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

	def getTimeShift(self, nChunkSize = 8000000):
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
			fftBaseChunk = scipy.fft(wavBaseChunk * wavBaseChunk * wavBaseChunk * wavBaseChunk)
			
			for j in range(numMp4):
				mp4 = self.lsMp4[j]
				# FFT squared signal
				# square is for highlighting peaks
				
				wav = np.interp(tBaseChunk, mp4['wav'].getTimeAxis(), mp4['wav'].x, left=0, right=0)
				wav = np.pad(wav, (0, nChunkSize-wav.size), 'constant')
				print 'FFT ' + mp4['name'], wav.size
				fftWav = scipy.fft(wav * wav * wav * wav)
				
				# get correlation function based on FFT (conjugate of convolution)
				corr = np.abs(scipy.ifft(fftBaseChunk * scipy.conj(fftWav)))
				# add offset to reduce effects from zero corr
				npCorrProd[j] = npCorrProd[j] * (corr/(corr.max()+1) + 1.0)
		
		# npCorrProd = npCorr.prod(axis=0)
		idxPeak = np.argmax(npCorrProd, axis=1)
		idxPeak[np.where(idxPeak > nChunkSize/2)] = idxPeak[np.where(idxPeak > nChunkSize/2)] - nChunkSize
		
		# allign minimum shifts to zero
		sampleShift = idxPeak - min(idxPeak)

		f = open(os.path.join("result","sync.txt"), "a")		
		for j in range(numMp4):
			self.lsMp4[j]['wav'].shiftSample(sampleShift[j])
			self.lsMp4[j]['time-end'] += float(sampleShift[j])/self.lsMp4[j]['wav'].f
			f.write(self.lsMp4[j]['mp4-file'] + ' ' + str(sampleShift[j]) + '\n')
		f.close()


		print 'Done'
		

	def fuse(self):
		fBase = 48000.0
		
		# find base signal - longest one
		# lsT = [mp4['wav'].getTEnd() for mp4 in self.lsMp4]
		lsT = [mp4['time-end'] for mp4 in self.lsMp4]
		TMax = max(lsT)
		tBase = np.arange(0,TMax, 1.0/fBase)
		
		numMp4 = len(self.lsMp4)
		wavMean = np.zeros(tBase.size).astype(int)
		for mp4 in self.lsMp4:
			mp4['time-shift'] = mp4['wav'].t0
			# mp4['time-end'] = mp4['wav'].getTEnd()
			del mp4['wav']

		for mp4 in self.lsMp4:
			print mp4['wav-file']
			wavfile = wave.open(mp4['wav-file'],'r')
			numCh = wavfile.getnchannels()
			wav = np.fromstring( wavfile.readframes(-1) , 'Int16' ).reshape(-1, numCh).mean(1)
			fr = float(wavfile.getframerate())
			sigWav = MySignal(x=wav, f = fr, t0 = mp4['time-shift'])
			
			wav = np.interp(tBase, sigWav.getTimeAxis(), sigWav.x, left=0, right=0)
			wavMean = wavMean + wav
			del wav
		
		wavMean = (wavMean / numMp4).astype(int)
		self.dictWav['fuse'] = MySignal(x=wavMean, f = fBase) 
		self.ax.clear()
		self.ax.plot(tBase[::100], wavMean[::100], label='mean')
		self.ax.set_xlabel('t(sec)')
		self.canvas.draw()
		self.bClick = True
		self.edt.appendPlainText("Fuse Done")
			
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
		np.savetxt(os.path.join("result","click.txt"), np.array(self.lsSplitPosition), fmt='%f')


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
		for mp4 in self.lsMp4:
			cap = cv2.VideoCapture(mp4['mp4-file'])
			fps = cap.get(cv2.CAP_PROP_FPS)
			strFilename, _ = os.path.splitext(os.path.basename(mp4['mp4-file']))
			cap.release()
			ls_nFrameEnd = [int(fps * t) - int(fps * mp4['time-shift']) for t in self.lsSplitPosition]
			np.savetxt(os.path.join("result", strFilename + "_frames.txt"), np.array(ls_nFrameEnd), fmt='%d')
		self.edt.appendPlainText("Done")




if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)

	main = Window()
	main.show()

	sys.exit(app.exec_())
















