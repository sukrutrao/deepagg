import numpy as np 
import csv
import sys

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
		questions, persons = self.get_counts(data)
		result_data = np.zeros((persons, questions))
		for i in range(0, len(data)):
			question = data[i][0]
			person = data[i][1]
			result_data[person][question] = data[i][2]
		if gt_csv == None:
			return result_data
		answers = []
		with open(gt_csv, "r") as gt_data:
			gt_reader = csv.reader(gt_data, delimiter=",")
			for row in gt_reader:
				answers.append(row)
		answers = np.array(answers, dtype=int)
		assert(len(answers) == questions)
		gt = np.zeros(questions)
		for i in range(0, questions):
			question = answers[i][0]
			gt[question] = answers[i][1]
		return result_data, gt
		
	def get_data_3D(self, crowd_csv, gt_csv=None, order="question"):
		assert(order == "question") # temporary, later also allow for "participant"
		data = []
		with open(crowd_csv, "r") as crowd_data:
			crowd_reader = csv.reader(crowd_data, delimiter=",")
			for row in crowd_reader:
				data.append(row)
		data = np.array(data, dtype=int)
		questions, persons = self.get_counts(data)
		options = len(data[0])-2
		result_data = np.zeros((persons, questions, options))
		for i in range(0, len(data)):
			question = data[i][0]
			person = data[i][1]
			for j in range(0,options):
				result_data[person][question][j] = data[i][2+j]
		if gt_csv == None:
			return result_data
		answers = []
		with open(gt_csv, "r") as gt_data:
			gt_reader = csv.reader(gt_data, delimiter=",")
			for row in gt_reader:
				answers.append(row)
		answers = np.array(answers, dtype=int)
		assert(len(answers) == questions)
		gt = np.zeros((questions,options),dtype=np.int)
		for i in range(0, questions):
			question = answers[i][0]
			for j in range(0,options):
				gt[question][j] = answers[i][1+j]
		return result_data, gt.astype(np.int)
		
	def get_gt(self,gt_csv):
		answers = []
		with open(gt_csv, "r") as gt_data:
			gt_reader = csv.reader(gt_data, delimiter=",")
			for row in gt_reader:
				answers.append(row)
		answers = np.array(answers, dtype=int)
		gt = np.zeros(len(answers))
		for i in range(0, len(answers)):
			question = answers[i][0]
			gt[question] = answers[i][1]
		return gt
		
	def get_gt_3D(self,gt_csv):
		answers = []
		with open(gt_csv, "r") as gt_data:
			gt_reader = csv.reader(gt_data, delimiter=",")
			for row in gt_reader:
				answers.append(row)
		answers = np.array(answers, dtype=int)
		questions = len(answers)
		options = len(answers[0])-1
		gt = np.zeros((questions,options),dtype=np.int)
		for i in range(0, questions):
			question = answers[i][0]
			for j in range(0,options):
				gt[question][j] = answers[i][1+j]
		return gt
		
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
	loader.get_data_3D('data.csv','gt.csv')


