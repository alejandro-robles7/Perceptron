from helper import *

'''
Homework1: perceptron classifier
'''
def sign(x):
	return 1 if x > 0 else -1

#-------------- Implement your code Below -------------#

def show_images(data):
	'''
	This function is used for plot image and save it.

	Args:
	data: Two images from train data with shape (2, 16, 16). The shape represents total 2
	      images and each image has size 16 by 16. 

	Returns:
		Do not return any arguments, just save the images you plot for your report.
	'''
	plt.imsave('five.png', data[0])
	plt.imsave('one.png', data[1])


def show_features(data, label):
	'''
	This function is used for plot a 2-D scatter plot of the features and save it. 

	Args:
	data: train features with shape (1561, 2). The shape represents total 1561 samples and 
	      each sample has 2 features.
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	
	Returns:
	Do not return any arguments, just save the 2-D scatter plot of the features you plot for your report.
	'''
	fig = plt.figure()
	alpha = 0.5

	for sample_label, sample in zip(label, data):
		if sample_label < 0:
			marker, color = "+", "blue"
		else:
			marker, color = "*", "red"
		plt.scatter(sample[0], sample[1], marker=marker, c=color, alpha=alpha)

	plt.xlabel("Symmetry")
	plt.ylabel("Average Intensity")
	plt.legend(('5', '1'))
	fig.savefig('features.png')


def perceptron(data, label, max_iter, learning_rate):
	'''
	The perceptron classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average internsity)
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update
	
	Returns:
		w: the seperater with shape (1, 3). You must initilize it with w = np.zeros((1,d))
	'''
	w = np.zeros(3)
	for iterations in range(max_iter):
		for x, y in zip(data, label):
			if np.dot(x, w) * y <= 0:
				w += learning_rate * x * y
	return w


def show_result(data, label, w):
	'''
	This function is used for plot the test data with the separators and save it.
	
	Args:
	data: test features with shape (424, 2). The shape represents total 424 samples and 
	      each sample has 2 features.
	label: test data's label with shape (424,1). 
		   1 for digit number 1 and -1 for digit number 5.
	
	Returns:
	Do not return any arguments, just save the image you plot for your report.
	'''
	x = np.arange(-1.0, 0.01)
	fig = plt.figure()
	plt.ylim(-1, 0.3)
	plt.xlim(-1, 0.1)
	alpha = 0.5

	for sample_label, sample in zip(label, data):
		if sample_label < 0:
			marker, color = "+", "blue"
		else:
			marker, color = "*", "red"
		plt.scatter(sample[0], sample[1], marker=marker, c=color, alpha=alpha)
	separator_line = -(w[1] / w[2]) * x - (w[0] / w[2])
	plt.plot(x, separator_line)
	plt.xlabel("Symmetry")
	plt.ylabel("Average Intensity")
	fig.savefig('results.png')


#-------------- Implement your code above ------------#
def accuracy_perceptron(data, label, w):
	n, _ = data.shape
	mistakes = 0
	for i in range(n):
		if sign(np.dot(data[i,:],np.transpose(w))) != label[i]:
			mistakes += 1
	return (n-mistakes)/n


def test_perceptron(max_iter, learning_rate):
	#get data
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	#train perceptron
	w = perceptron(train_data, train_label, max_iter, learning_rate)
	train_acc = accuracy_perceptron(train_data, train_label, w)	
	#test perceptron model
	test_acc = accuracy_perceptron(test_data, test_label, w)
	return w, train_acc, test_acc


