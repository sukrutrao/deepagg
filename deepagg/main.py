import numpy as np
import block1
import block2
import feature 
import loader 
import utils

class EM2D:
	"""  
	Class for performing the DeepAgg EM on a 2D matrix
	In a multi-answer context, this could mean a matrix of 
	persons vs options, with binary choices for each value
	"""
	
	def __init__(self,num_p,num_q,k_ability,k_difficulty,num_classes):
		"""  
		Create blocks and initialize members
		"""
		self.loader = loader.Loader()
		self.num_classes = num_classes
		self.num_participants = num_p 
		self.num_questions = num_q 
		self.k_ability = k_ability
		self.k_difficulty = k_difficulty
		self.feature = feature.FeatureRepresenter(self.num_participants,self.num_questions,
					k_ability,k_difficulty,2)
		self.block1 = block1.Block1(k_ability+k_difficulty+2)
		self.block2 = block2.Block2(self.num_participants,self.num_classes)		
		
	def train_block_1(self,train_csv,weights_name='block1_weights.npy',multiplicative_factor=10,
					num_epochs=10,batch_size=20,learning_rate=0.01,momentum=0.9,validate_split=0.2):
		"""  
		Train the neural network corresponding to block1
		"""
		# get the train data as a 2D matrix and train labels as a vector
		train_X, train_y = self.loader.get_data(train_csv)
		total_train_X = [], total_train_y = []
		for s_train_X, s_train_y in utils.augment_set(train_X,train_y,self.num_participants,
						self.num_questions,multiplicative_factor):
			# generate the phi features using this data
			self.feature.generate_features_2d(s_train_X,s_train_y)
			# get the phi features and the labels in flattened vectors of dimensions p*q
			f_train_X, f_train_y = self.feature.get_features_2d()
			total_train_X.append(f_train_X)
			total_train_y.append(f_train_y)
		# split it into train and validation splits
		train_X, train_y, val_X, val_y = utils.split_train_and_val(total_train_X,total_train_y,validate_split)
		# train the network
		self.block1.train(train_X,train_y,val_X,val_y,num_epochs,batch_size,learning_rate,momentum)
		# save the weights
		self.block1.save_weights(weights_name)
		
	def predict(self,test_csv,iteration_type='fixed',num_iterations=10):
		"""  
		Predict the answer for a set of questions
		The test file can contain more than p people and q questions
		This needs to take subsets before predicting
		This will return an array of proposed answers
		"""
		# currently assume test.csv contains tests of the correct size
		input_data = self.loader.get_data(test_csv)
		num_participants = len(input_data)
		if num_participants > 0:
			num_questions = len(input_data[0])
		else:
			return
		proposals = utils.majority_voting(input_data,self.num_classes,2)
		for i in range(0,num_iterations):
			abilities = self.cor(input_data,proposals)
			proposals = self.ref(abilities,input_data)
		return proposals
		
	def cor(self,input_data,proposals):
		self.feature.generate_features_2d(input_data,proposals)
		test_X, test_y = self.feature.get_features_2d()
		abilities = self.block1.predict(test_X)
		return abilities
		
	def ref(self,abilities,input_data):
		proposals = []
		ability_matrix = []
		counter = 0
		for i in range(0,self.num_participants):
			for j in range(0,self.num_questions):
				ability_matrix.append(abilities[counter])
				counter += 1
		ability_matrix = np.array(ability_matrix)
		for j in range(0,self.num_questions):
			ability_vector = ability_matrix[:,j]
			input_vector = input_data[:,j]
			self.block2.fit(ability_vector)
			# TODO use single function in block2 insteaf of iterating here
			prediction = self.block2.predict_element(input_vector)
			proposals.append(prediction)
		proposals = np.array(proposals)
		return proposals
		
		
