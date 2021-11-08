import numpy as np
import pylab
import pickle as p
import time

import sys
caffe_root = '../caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_cpu()

def f11(image):
	model_def = 'deploy1_1.prototxt'
	model_weights = 'with_biases.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	# # mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
	# # mu = mu.mean(1).mean(1)
	# # print 'mean-subtracted values:', zip('BGR', mu)

	# transformer = caffe.io.Transformer({'data_1': net.blobs['data_1'].data.shape})
	# transformer.set_transpose('data_1', (2,0,1))
	# # transformer.set_mean('data', mu)
	# transformer.set_raw_scale('data_1', 255)
	# #transformer.set_channel_swap('data', (2,1,0))

	# #net.blobs['data'].reshape(50,
	# #						  3,
	# # 						  28, 28)

	# ### image = image1 - image2 ###
	# #f = file('image1.data')
	# #image = p.load(f)
	# ###############################

	# ### image1 = image + image2 ###
	# image = caffe.io.load_image(img, color=False)
	# # print(image.shape)
	# # f = file('image1.data')
	# # image2 = p.load(f)
	# with open('image1.data', 'rb') as f:
	# 	image2 = p.load(f, encoding='latin1')
	# t1 = time.time()
	# image = image + image2
	# t2 = time.time()
	# # print(t2 - t1)
	# ###############################

	# net.blobs['data_1'].data[...] = transformer.preprocess('data_1', image)
	net.blobs['data_1'].data[...] = image

	#pylab.imshow(transformed_image.transpose(1, 0, 2).reshape(28, 1*28), cmap='gray'); pylab.axis('off')
	#pylab.show()

	net.forward()

	output = net.blobs['conv1_1'].data

	return output