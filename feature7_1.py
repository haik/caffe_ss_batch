import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_cpu()

def f71(output1):
	model_def = 'deploy7_1.prototxt'
	model_weights = 'with_biases.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	net.blobs['pool2_1'].data[...] = output1

	net.forward()

	output = net.blobs['fc1_1'].data

	return output