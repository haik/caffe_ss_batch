import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_cpu()

def f42(output2):
	model_def = 'deploy4_2.prototxt'
	model_weights = 'without_biases.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['pool1_2'].data[...] = output2

	net.forward()

	output = net.blobs['conv2_2'].data

	return output