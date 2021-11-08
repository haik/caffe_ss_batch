import numpy as np
import os
import cv2

import feature1_1, feature1_2, feature2_1, feature2_2, feature3_1, feature3_2
import feature4_1, feature4_2, feature5_1, feature5_2, feature6_1, feature6_2
import feature7_1, feature7_2, feature8_1, feature8_2, feature9_1, feature9_2

dir = 'mnist/test/9'
filelist = []
filenames = os.listdir(dir)
for fn in filenames:
	fullfilename = os.path.join(dir,fn)
	filelist.append(fullfilename)

result = []

for i in range(0, 10):	# len(filelist)
	img = filelist[i]
	image = cv2.imread(img, 0)
	if image.shape != [28, 28]:
		image0 = cv2.resize(image, (28, 28))
		image = image0.reshape(28, 28, -1)
	else:
		image = image.reshape(28, 28, -1)
	image = 1.0 - image/255.0

	# image1 = np.random.random((28, 28, 1))
	image1 = np.random.randint(2**10, size=(28,28,1))
	image2 = image1 - image
	print(np.max(image1), np.min(image1))
	print(np.max(image2), np.min(image2))

	image1 = image1.transpose(2,0,1)
	image2 = image2.transpose(2,0,1)

	conv1_1 = feature1_1.f11(image1)
	conv1_2 = feature1_2.f12(image2)
	relu1_1 = feature2_1.f21(conv1_1, conv1_2)
	relu1_2 = feature2_2.f22(conv1_1, conv1_2)
	pool1_1 = feature3_1.f31(relu1_1, relu1_2)
	pool1_2 = feature3_2.f32(relu1_1, relu1_2)
	conv2_1 = feature4_1.f41(pool1_1)
	conv2_2 = feature4_2.f42(pool1_2)
	relu2_1 = feature5_1.f51(conv2_1, conv2_2)
	relu2_2 = feature5_2.f52(conv2_1, conv2_2)
	pool2_1 = feature6_1.f61(relu2_1, relu2_2)
	pool2_2 = feature6_2.f62(relu2_1, relu2_2)
	fc1_1 = feature7_1.f71(pool2_1)
	fc1_2 = feature7_2.f72(pool2_2)
	relu3_1 = feature8_1.f81(fc1_1, fc1_2)
	relu3_2 = feature8_2.f82(fc1_1, fc1_2)
	score_1 = feature9_1.f91(relu3_1)
	score_2 = feature9_2.f92(relu3_2)

	score = score_1 - score_2
	result.append(score.argmax())
print(result)