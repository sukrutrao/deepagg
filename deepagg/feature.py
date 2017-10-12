import numpy as np 
import theano
import theano.tensor as T
import lasagne

class FeatureRepresenter:
	"""
	Get the feature representation of the data for giving
	as input to Block1
	"""
	
	def __init__(self,num_p,num_q,k_ability=3,k_difficulty=3,num_dimensions=2):
		"""
		k_ability - number of buckets for ability
		k_difficulty - number of buckets for difficulty
		num_p - rows in the matrix
		num_q - columns in the matrix
		num_dimensions - 2 for 2D matrix, etc.
		"""
		self.k_ability = k_ability
		self.k_difficulty = k_difficulty
		self.num_dimensions = num_dimensions
		self.num_participants = num_p 
		self.num_questions = num_q
		
	def get_average_ability_2d(self):
		"""
		Get average ability in 2D case
		"""
		abilities = []
		for i in range(0,self.num_participants):
			correct_count = 0
			for j in range(0,self.num_questions):
				if self.current_proposals[j] == self.input_data[i][j]:
					correct_count += 1
			ability = float(correct_count)/self.num_questions
			abilities.append((ability,i))
		abilities = np.array(abilities)
		return abilities
	
	def get_average_difficulty_2d(self):
		"""
		Get average difficulty in 2D case
		"""
		difficulties = []
		for i in range(0,self.num_questions):
			correct_count = 0
			for j in range(0,self.num_participants):
				if self.current_proposals[j] == self.input_data[i][j]:
					correct_count += 1
			difficulty = 1-float(correct_count)/self.num_participants
			difficulties.append((difficulty,i))
		difficulties = np.array(difficulties)
		return difficulties
		
	def get_buckets(self,attribute,k):
		"""
		Get k buckets for the attribute list given in ascending order
		"""
		attribute = attribute.tolist()
		attribute.sort(key=lambda x:x[0])
		attribute_step_size = len(attribute)/k
		for i in range(0,k):
			current_bucket = attribute[i*attribute_step_size:min((i+1)*attribute_step_size,len(attribute))]
			for j in range(0,len(current_bucket)):
				current_bucket[j][1] = int(current_bucket[j][1])
			yield current_bucket	
			
	def get_ability_in_bucket(self,ability_bucket):
		"""
		Get the ability in the bucket ability_bucket
		"""
		abilities = []
		for i in range(0,self.num_participants):
			correct_count = 0	
			for j in range(0,len(ability_bucket)):
				if self.current_proposals[ability_bucket[j][1]] == self.input_data[i][ability_bucket[j][1]]:
					correct_count += 1
			ability = float(correct_count)/len(ability_bucket)
			abilities.append((ability,i))
		abilities = np.array(abilities)
		return abilities
		
	def get_difficulty_in_bucket(self,difficulty_bucket):
		"""
		Get the difficulty in the bucket difficulty_bucket
		"""
		difficulties = []
		for i in range(0,self.num_questions):
			correct_count = 0	
			for j in range(0,len(difficulty_bucket)):
				if self.current_proposals[i] == self.input_data[difficulty_bucket[j][1]][i]:
					correct_count += 1
			difficulty = 1-float(correct_count)/len(difficulty_bucket)
			difficulties.append((difficulty,i))
		difficulties = np.array(difficulties)
		return difficulties
		
	def generate_features_2d(self,input_data,current_proposals):
		"""
		Generate the feature input given the data and current proposed
		answer sheet
		"""
		assert len(np.shape(input_data)) == self.num_dimensions
		assert len(current_proposals) == self.num_questions
		self.input_data = input_data
		self.current_proposals = current_proposals
		self.abilities = self.get_average_ability_2d()
		self.difficulties = self.get_average_difficulty_2d()
		abilities_bucket_wise = []
		difficulties_bucket_wise = []
		for ability_bucket in self.get_buckets(self.abilities[:],self.k_ability):
			abilities_bucket_wise.append(self.get_ability_in_bucket(ability_bucket))
		for difficulty_bucket in self.get_buckets(self.difficulties[:],self.k_difficulty):
			difficulties_bucket_wise.append(self.get_difficulty_in_bucket(difficulty_bucket))
		self.abilities_bucket_wise = np.array(abilities_bucket_wise)
		self.difficulties_bucket_wise = np.array(difficulties_bucket_wise)				
		
	def get_features_2d_element(self,person,question):
		"""
		Get a particular value in the generated feature matrix
		"""
		feature_list = []
		feature_list.append(self.abilities[person][0])
		for i in range(0,self.k_ability):
			feature_list.append(self.abilities_bucket_wise[i][person][0])
		feature_list.append(self.difficulties[question][0])
		for i in range(0,self.k_difficulty):
			feature_list.append(self.difficulties_bucket_wise[i][question][0])
		feature_list = np.array(feature_list)
		return feature_list
		
	def get_features_2d(self):
		"""
		Get the feature list by flattring the feature matrix
		"""
		features_list = []
		labels_list = []
		for i in range(0,self.num_participants):
			for j in range(0,self.num_questions):
				features_list.append(self.get_features_2d_element(i,j))
				labels_list.append(self.current_proposals[j])
		features_list = np.array(features_list)
		labels_list = np.array(labels_list)
		return features_list, labels_list
		
		
if __name__ == "__main__":
	feature = FeatureRepresenter(3,3,3,3,2)
	feature.generate_features_2d([[1,0,0],[0,1,0],[1,1,0]],[1,0,1])
	print "Defined Feature Creator for DeepAgg"
