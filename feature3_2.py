import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_cpu()

def f32(relu1_output, relu2_output):
	model_def = 'deploy3_2.prototxt'
	model_weights = 'without_biases.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['relu1_1'].data[...] = relu1_output
	net.blobs['relu1_2'].data[...] = relu2_output

	net.forward()

	output = net.blobs['pool1_2'].data

	return output