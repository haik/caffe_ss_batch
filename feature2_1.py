import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_cpu()

def f21(conv1_output, conv2_output):
	model_def = 'deploy2_1.prototxt'
	model_weights = 'with_biases.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['conv1_1'].data[...] = conv1_output
	net.blobs['conv1_2'].data[...] = conv2_output

	net.forward()

	output = net.blobs['relu1_1'].data
	
	return output