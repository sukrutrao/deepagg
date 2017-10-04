import numpy as np 
import theano
import theano.tensor as T
import lasagne

class FeatureRepresenter:
	
	def __init__(self,k_ability=3,k_difficulty=3,num_dimensions=2,num_p,num_q):
		self.k_ability = k_ability
		self.k_difficulty = k_difficulty
		self.num_dimensions = num_dimensions
		self.num_participants = num_p 
		self.num_questions = num_q
		
	def get_average_ability_2d():
		abilities = []
		for i in range(0,self.num_participants):
			correct_count = 0
			for j in range(0,self.num_questions):
				if self.current_proposals[j] == self.input_data[i][j]:
					correct_count += 1
			ability = float(correct_count)/self.num_questions
			abilities.append((ability,i))
		return abilities
	
	def get_average_difficulty_2d():
		difficulties = []
		for i in range(0,self.num_questions):
			correct_count = 0
			for j in range(0,self.num_participants):
				if self.current_proposals[j] == self.input_data[i][j]:
					correct_count += 1
			difficulty = 1-float(correct_count)/self.num_participants
			difficulties.append((difficulty,i))
		return difficulties
		
	def get_buckets(self,attribute,k):
		sorted_attribute = attribute.sort(key=lambda x:x[0])
		attribute_step_size = float(len(attribute))/k
		for i in range(0,k):
			current_bucket = sorted_attribute[i*attribute_step_size:min((i+1)*attribute_step_size,len(attribute))]
			yield current_bucket			
		
	def get_features(self,input_data,current_proposals):
		assert len(np.shape(input_data)) == self.num_dimensions
		assert len(current_proposals) == self.num_questions
		self.input_data = input_data
		self.current_proposals = current_proposals
		abilities = self.get_average
