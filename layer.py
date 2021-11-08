import caffe
import numpy as np
import sys
caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

class AddLayer(caffe.Layer):
	def setup(self, bottom, top):
		pass
	def reshape(self, bottom, top):
		top[0].reshape(bottom[0].num, bottom[0].channels, bottom[0].height, bottom[0].width)
	def forward(self, bottom, top):
		top[0].data[...] = 10 * bottom[0].data
	def backward(self, bottom, top):
		pass

class ReluALayer(caffe.Layer):
	"""
	Compute the Convolution in Python.
	"""
	def setup(self, bottom, top):
		# check input pair
		pass

	def reshape(self, bottom, top):
		# top[0].reshape(*bottom[0].data.shape)
		# copy shape from bottom:
		top[0].reshape(bottom[0].num, bottom[0].channels, bottom[0].height, bottom[0].width)
		
		# top[0].reshape(10, 64, 224, 224)
		# change shape:
		# top[0].reshape(1)

	def forward(self, bottom, top):
		# perform computation
		# top[0].data[...] = ...
		top[0].data[...] = bottom[0].data
		print(top[0].data[0][0][0])
		#top[1].data[...] = bottom[1].data
		print(bottom[1].data[0][0][0])
		temp = bottom[0].data - bottom[1].data
		top[0].data[temp < 0] = 0
		print(top[0].data[0][0][0])
		#top[1].data[temp < 0] = 0
		#print top[1].data[0][0][0]

	def backward(self, top, propagate_down, bottom):
		pass

class ReluBLayer(caffe.Layer):
	"""
	Compute the Convolution in Python.
	"""
	def setup(self, bottom, top):
		# check input pair
		pass

	def reshape(self, bottom, top):
		# top[0].reshape(*bottom[0].data.shape)
		# copy shape from bottom:
		top[0].reshape(bottom[0].num, bottom[0].channels, bottom[0].height, bottom[0].width)
		
		# top[0].reshape(10, 64, 224, 224)
		# change shape:
		# top[0].reshape(1)

	def forward(self, bottom, top):
		# perform computation
		# top[0].data[...] = ...
		top[0].data[...] = bottom[1].data
		print(top[0].data[0][0][0])
		#top[1].data[...] = bottom[1].data
		print(bottom[0].data[0][0][0])
		temp = bottom[0].data - bottom[1].data
		top[0].data[temp < 0] = 0
		print(top[0].data[0][0][0])
		#top[1].data[temp < 0] = 0
		#print top[1].data[0][0][0]

	def backward(self, top, propagate_down, bottom):
		pass

class ReluCLayer(caffe.Layer):
	"""
	Compute the Convolution in Python.
	"""
	def setup(self, bottom, top):
		# check input pair
		top[0].reshape(bottom[0].num, bottom[0].channels, 1, 1)

	def reshape(self, bottom, top):
		#top[0].reshape(*bottom[0].data.shape)
		# copy shape from bottom:
		
		pass
		# top[0].reshape(10, 64, 224, 224)
		# change shape:
		# top[0].reshape(1)

	def forward(self, bottom, top):
		# perform computation
		# top[0].data[...] = ...
		top[0].data[...] = bottom[0].data
		print(top[0].data[0][0][0])
		#top[1].data[...] = bottom[1].data
		print(bottom[1].data[0][0][0])
		temp = bottom[0].data - bottom[1].data
		top[0].data[temp < 0] = 0
		print(top[0].data[0][0][0])
		#top[1].data[temp < 0] = 0
		#print top[1].data[0][0][0]

	def backward(self, top, propagate_down, bottom):
		pass

class ReluDLayer(caffe.Layer):
	"""
	Compute the Convolution in Python.
	"""
	def setup(self, bottom, top):
		# check input pair
		pass

	def reshape(self, bottom, top):
		top[0].reshape(*bottom[0].data.shape)
		# copy shape from bottom:
		#top[0].reshape(bottom[0].num, bottom[0].channels, bottom[0].height, bottom[0].width)
		#pass
		# top[0].reshape(10, 64, 224, 224)
		# change shape:
		# top[0].reshape(1)

	def forward(self, bottom, top):
		# perform computation
		# top[0].data[...] = ...
		top[0].data[...] = bottom[1].data
		print(top[0].data[0][0][0])
		#top[1].data[...] = bottom[1].data
		print(bottom[0].data[0][0][0])
		temp = bottom[0].data - bottom[1].data
		top[0].data[temp < 0] = 0
		print(top[0].data[0][0][0])
		#top[1].data[temp < 0] = 0
		#print top[1].data[0][0][0]

	def backward(self, top, propagate_down, bottom):
		pass

class MaxALayer(caffe.Layer):
	def setup(self, bottom, top):
		pass

	def reshape(self, bottom, top):
		top[0].reshape(bottom[0].num, bottom[0].channels, int(bottom[0].height/2), int(bottom[0].width/2))
		

	def forward(self, bottom, top):
		bn = bottom[0].num
		bc = bottom[0].channels
		bh = bottom[0].height
		bw = bottom[0].width
		tn = bn
		tc = bc
		th = int(bh / 2)
		tw = int(bw / 2)
		temp_max = 0
		for n in range(tn):
			for c in range(tc):
				#temp_b = bottom[0].data[n, c]
				#temp_b = bottom_temp[n, c]
				temp_b1 = bottom[0].data[n, c]
				temp_b2 = bottom[1].data[n, c]
				temp_b = temp_b1 - temp_b2

				#temp_t = np.zeros(th * tw)
				temp_t1 = np.zeros((th, tw))
				temp_t2 = np.zeros((th, tw))
				for h in range(th):
					for w in range(tw):
						### Scheme 1 ###
						#temp_t[h*th+w] = np.max(temp_b[h*2:h*2+2, w*2:w*2+2])
				#top[0].data[n, c] = temp_t.reshape(th, tw)
						
						### Scheme 2 ###
						#temp_t[h, w] = np.max(temp_b[h*2:h*2+2, w*2:w*2+2])
				#top[0].data[n, c] = temp_t

						position = np.where(temp_b == np.max(temp_b[h*2:h*2+2, w*2:w*2+2]))
						p_x = position[0][0]
						p_y = position[1][0]
						temp_t1[h, w] = temp_b1[p_x, p_y]
						temp_t2[h, w] = temp_b2[p_x, p_y]
				top[0].data[n, c] = temp_t1
				

class MaxBLayer(caffe.Layer):
	def setup(self, bottom, top):
		pass

	def reshape(self, bottom, top):
		top[0].reshape(bottom[0].num, bottom[0].channels, int(bottom[0].height/2), int(bottom[0].width/2))
		

	def forward(self, bottom, top):
		bn = bottom[0].num
		bc = bottom[0].channels
		bh = bottom[0].height
		bw = bottom[0].width
		tn = bn
		tc = bc
		th = int(bh / 2)
		tw = int(bw / 2)
		temp_max = 0
		for n in range(tn):
			for c in range(tc):
				#temp_b = bottom[0].data[n, c]
				#temp_b = bottom_temp[n, c]
				temp_b1 = bottom[0].data[n, c]
				temp_b2 = bottom[1].data[n, c]
				temp_b = temp_b1 - temp_b2

				#temp_t = np.zeros(th * tw)
				temp_t1 = np.zeros((th, tw))
				temp_t2 = np.zeros((th, tw))
				for h in range(th):
					for w in range(tw):
						### Scheme 1 ###
						#temp_t[h*th+w] = np.max(temp_b[h*2:h*2+2, w*2:w*2+2])
				#top[0].data[n, c] = temp_t.reshape(th, tw)
						
						### Scheme 2 ###
						#temp_t[h, w] = np.max(temp_b[h*2:h*2+2, w*2:w*2+2])
				#top[0].data[n, c] = temp_t

						position = np.where(temp_b == np.max(temp_b[h*2:h*2+2, w*2:w*2+2]))
						p_x = position[0][0]
						p_y = position[1][0]
						temp_t1[h, w] = temp_b1[p_x, p_y]
						temp_t2[h, w] = temp_b2[p_x, p_y]
				top[0].data[n, c] = temp_t2
				

	def backward(self, top, propagate_down, bottom):
		pass
