import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys

class Block2:
	"""
	Define the probabilistic Block2 of DeepAgg
	"""
	
	def __init__(self,number_of_people,num_classes=2):
		"""
		number_of_people - participant count
		num_classes - 2 for binary, etc.
		"""
		self.number_of_people = number_of_people
		self.num_classes = num_classes
		
	def fit(self,ability_vector):
		"""
		Store the ability vector in the network
		"""
		print self.number_of_people
		print np.shape(ability_vector)
	#	sys.exit(0)
		assert self.number_of_people == len(ability_vector)
		self.ability_vector = ability_vector
		
	def predict_element(self,test_X_element):
		"""
		Get prediction for a given question
		"""
		assert len(test_X_element) == self.number_of_people
		predictions = []
		for i in range(0,self.num_classes):
			this_class_prob = 1
			for j in range(0,self.number_of_people):
				if test_X_element[j] == i:
					this_class_prob = this_class_prob*self.ability_vector[j]
				elif test_X_element[j] >= 0:
					this_class_prob = this_class_prob*(1-self.ability_vector[j])
					this_class_prob = this_class_prob/(self.num_classes-1)
				else:
					this_class_prob = this_class_prob*0.5 # check!
			predictions.append(this_class_prob)
		predictions = np.array(predictions)
		class_predicted = np.argmax(predictions)
		return class_predicted
		
	def predict_single(self,test_X):
		"""
		Get prediction for a given matrix of answers
		PROBABLY WONT WORK as only a single ability vector is stored
		"""
		predictions = []
		for i in range(0,len(test_X)):
			predictions.append(predict_element(test_X[i]))
		predictions = np.array(predictions)
		return predictions
		
	def predict_multiple(self,test_X):
		"""
		Get prediction for 3D matrix
		PROBABLY WONT WORK as only a single ability vector is stored
		"""
		predictions = []
		for i in range(0,len(test_X)):
			predictions.append(predict_single(test_X[i]))
		predictions = np.array(predictions)
		return predictions
		
if __name__ == "__main__":
	block2 = Block2(3,2)
	block2.fit([0.2,0.5,0.1])
	block2.predict_element([0,0,1])
	print "Defined Block2 of DeepAgg"
		
