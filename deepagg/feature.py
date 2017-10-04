import numpy as np 
import theano
import theano.tensor as T
import lasagne

class FeatureRepresenter:
	
	def __init__(self,num_p,num_q,k_ability=3,k_difficulty=3,num_dimensions=2):
		self.k_ability = k_ability
		self.k_difficulty = k_difficulty
		self.num_dimensions = num_dimensions
		self.num_participants = num_p 
		self.num_questions = num_q
		
	def get_average_ability_2d(self):
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
		attribute = attribute.tolist()
		attribute.sort(key=lambda x:x[0])
		attribute_step_size = len(attribute)/k
		for i in range(0,k):
			current_bucket = attribute[i*attribute_step_size:min((i+1)*attribute_step_size,len(attribute))]
			for j in range(0,len(current_bucket)):
				current_bucket[j][1] = int(current_bucket[j][1])
			yield current_bucket	
			
	def get_ability_in_bucket(self,ability_bucket):
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
		
	def get_features_2d(self,person,question):
		feature_list = []
		feature_list.append(self.abilities[person][0])
		for i in range(0,self.k_ability):
			feature_list.append(self.abilities_bucket_wise[i][person][0])
		feature_list.append(self.difficulties[question][0])
		for i in range(0,self.k_difficulty):
			feature_list.append(self.difficulties_bucket_wise[i][question][0])
		feature_list = np.array(feature_list)
		return feature_list
		
if __name__ == "__main__":
	feature = FeatureRepresenter(3,3,3,3,2)
	feature.generate_features_2d([[1,0,0],[0,1,0],[1,1,0]],[1,0,1])
	print "Defined Feature Creator for DeepAgg"