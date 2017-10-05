import numpy as np
import theano
import theano.tensor as T
import lasagne

def majority_voting(input_data,num_classes,num_dimensions=2):
	# currently supporting only 2D
	num_participants = len(input_data)
	if num_participants > 0:
		num_questions = len(input_data[0])
	else:
		return np.empty(0)
	proposals = []
	for i in range(0,num_questions):
		class_counts = np.zeros(num_classes,dtype=int)
		for j in range(0,num_participants):
			class_counts[input_data[j][i]] += 1
		proposals.append(np.argmax(class_counts))
	proposals = np.array(proposals)
	return proposals
	
def augment_set(train_X,train_y,num_p,num_q,factor):
	#TODO
	pass
	
def split_train_and_val(total_train_X,total_train_y,validate_split=0.2):
	#TODO
	pass
