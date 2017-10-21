import numpy as np 
import csv

class Loader:
	"""
	To load the data
	"""
	
	def __init__(self):
		pass
			
	def get_data(self, crowd_csv, gt_csv=None, order="question"):
		assert(order == "question") # temporary, later also allow for "participant"
		data = []
		with open(crowd_csv, "r") as crowd_data:
			crowd_reader = csv.reader(crowd_data, delimiter=",")
			for row in crowd_reader:
				data.append(row)
		data = np.array(data, dtype=int)
		print np.shape(data)
		questions, persons = self.get_counts(data)
		print questions, persons
		result_data = np.zeros((persons, questions))
		for i in range(0, len(data)):
			question = data[i][0]
			person = data[i][1]
			result_data[person][question] = data[i][2]
		print result_data
		if gt_csv == None:
			return result_data
		answers = []
		with open(gt_csv, "r") as gt_data:
			gt_reader = csv.reader(gt_data, delimiter=",")
			for row in gt_reader:
				answers.append(row)
		answers = np.array(answers, dtype=int)
		print answers
		assert(len(answers) == questions)
		gt = np.zeros(questions)
		for i in range(0, questions):
			question = answers[i][0]
			gt[question] = answers[i][1]
		print gt
		return result_data, gt
		
	def get_counts(self, data):
		shape = np.shape(data)
		assert(len(shape) == 2)
		questions_product = shape[0]
		first_q = data[0][0]
		persons = 0
		for i in range(0, len(data)):
			if data[i][0] == first_q:
				persons += 1
		questions = questions_product/persons
		assert(questions*persons == questions_product)
		return questions, persons


if __name__ == "__main__":	
	loader = Loader()
	loader.get_data('/home/home/Sukrut/crowdsourced-data-simulator/data.csv','/home/home/Sukrut/crowdsourced-data-simulator/gt.csv')


