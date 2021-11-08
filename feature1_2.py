import numpy as np
import pylab
import pickle as p

import sys
caffe_root = '../caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_cpu()

def f12(image):
	model_def = 'deploy1_2.prototxt'
	model_weights = 'without_biases.caffemodel'

	net = caffe.Net(model_def,
		            model_weights,
		            caffe.TEST)

	# # mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
	# # mu = mu.mean(1).mean(1)
	# # print 'mean-subtracted values:', zip('BGR', mu)

	# transformer = caffe.io.Transformer({'data_2': net.blobs['data_2'].data.shape})
	# transformer.set_transpose('data_2', (2,0,1))
	# # transformer.set_mean('data', mu)
	# transformer.set_raw_scale('data_2', 255)
	# #transformer.set_channel_swap('data', (2,1,0))

	# #net.blobs['data'].reshape(50,
	# #						  3,
	# # 						  28, 28)

	# ### image = image1 - image2 ###
	# #image0 = caffe.io.load_image('mnist/test/8/07805.png', color=False)
	# #f = file('image1.data')
	# #image1 = p.load(f)
	# #image = image1 - image0
	# ###############################

	# ### image1 = image + image2 ### 
	# # f = file('image1.data')
	# # image = p.load(f)
	# with open('image1.data', 'rb') as f:
	# 	image = p.load(f, encoding='latin1')
	# ###############################

	# net.blobs['data_2'].data[...] = transformer.preprocess('data_2', image)
	net.blobs['data_2'].data[...] = image

	#pylab.imshow(transformed_image.transpose(1, 0, 2).reshape(28, 1*28), cmap='gray'); pylab.axis('off')
	#pylab.show()
	
	net.forward()

	output = net.blobs['conv1_2'].data

	return output