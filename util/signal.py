
import sys
import os
import numpy as np
import scipy

class MySignal:
	# x, t0, f
	def __init__(self, *args, **kwargs):
		self.x = kwargs['x'] 
		self.f = kwargs['f'] 

		if 't0' in kwargs.keys():
			self.t0 = kwargs['t0']
		else:
			self.t0 = 0

	def getLength(self):
		return self.x.size

	def getTEnd(self):
		return self.t0 + (self.x.size - 1.0)/self.f

	def getTimeAxis(self):
		return np.linspace(self.t0, self.getTEnd(), num=self.x.size)

	def shiftSample(self, nShift):
		self.t0 = self.t0 + float(nShift)/self.f
		
