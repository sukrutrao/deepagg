import numpy as np
import random
import scipy.misc
import theano
import theano.tensor as T
import lasagne
import sys

def majority_voting(input_data,num_classes,num_dimensions=2):
	"""
	Get result of majority voting
	"""
	# currently supporting only 2D
	num_participants = len(input_data)
	if num_participants > 0:
		num_questions = len(input_data[0])
	else:
		return np.empty(0)
	proposals = []
	input_data = input_data.astype(int)
#	sys.exit(0)
	for i in range(0,num_questions):
		class_counts = np.zeros(num_classes,dtype=int)
		for j in range(0,num_participants):
			class_counts[input_data[j][i]] += 1
		proposals.append(np.argmax(class_counts))
	proposals = np.array(proposals)
	return proposals
	
def augment_set(train_X,train_y,num_p,num_q,factor):
	"""
	Perform augmentation
	"""
	if (len(train_X) <= num_p or len(train_y) <= num_q):
		print "Yielding"
		yield train_X, train_y
	else:
		total_p = len(train_X)
		total_q = len(train_y)
		augmentation_limit = min(factor,scipy.misc.comb(total_p,num_p)*scipy.misc.comb(total_q,num_q))
		for i in range(0,augmentation_limit):
			random_p = random.sample(range(0,total_p),num_p)
			random_q = random.sample(range(0,total_q),num_q)
			s_train_X = []
			s_train_y = []
			for j in range(0,len(random_p)):
				s_train_X_element = []
				for k in range(0,len(random_q)):
					s_train_X_element.append(train_X[j][k])
				s_train_X.append(s_train_X_element)
			for j in range(0,len(random_q)):
				s_train_y.append(train_y[j])
			s_train_X = np.array(s_train_X)
			s_train_y = np.array(s_train_y)
			yield s_train_X, s_train_y
	
def split_train_and_val(total_train_X,total_train_y,validate_split=0.2):
	"""
	Split into train and validation splits
	"""
	total_p = len(total_train_X)
	total_q = len(total_train_y)
	val_p_count = int(validate_split*total_p)
	val_q_count = int(validate_split*total_q)
	val_p = random.sample(range(0,total_p),val_p_count)
	val_q = random.sample(range(0,total_q),val_q_count)
	train_X = []
	train_y = []
	val_X = []
	val_y = []
	print np.shape(total_train_X)
#	sys.exit(0)
	for i in range(0,total_p):
		if i in val_p:
			val_X_element = []
			for j in range(0,total_q):
				if j in val_q:
					val_X_element.append(total_train_X[i][j])
			val_X.append(val_X_element)
		else:
			train_X_element = []
			for j in range(0,total_q):
				if j not in val_q:
					train_X_element.append(total_train_X[i][j])
			train_X.append(train_X_element)
	for i in range(0,total_q):
		if i in val_q:
			val_y.append(total_train_y[i])
		else:
			train_y.append(total_train_y[i])
	train_X = np.array(train_X)
	train_y = np.array(train_y)
	val_X = np.array(val_X)
	val_y = np.array(val_y)
	return train_X, train_y, val_X, val_y
	
def flatten_2D_list(input_list):
	result_list = []
	for i in range(0,len(input_list)):
		for j in range(0,len(input_list[i])):
			result_list.append(input_list[i][j])
	return result_list
