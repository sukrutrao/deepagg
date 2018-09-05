import numpy as np
import block1
import block2
import feature
import loader
import utils
import sys


class EM2D:
    """  
    Class for performing the DeepAgg EM on a 2D matrix
    In a multi-answer context, this could mean a matrix of 
    persons vs options, with binary choices for each value
    """

    def __init__(self, num_p, num_q, k_ability, k_difficulty, num_classes):
        """  
        Create blocks and initialize members
        """
        self.loader = loader.Loader()
        self.num_classes = num_classes
        self.num_participants = num_p
        self.num_questions = num_q
        self.k_ability = k_ability
        self.k_difficulty = k_difficulty
        self.feature = feature.FeatureRepresenter(self.num_participants, self.num_questions,
                                                  k_ability, k_difficulty, 2)
        self.block1 = block1.Block1(k_ability + k_difficulty + 2)
        self.block2 = block2.Block2(self.num_participants, self.num_classes)

    def train_block_1(self, train_csv, gt_csv, weights_name='block1_weights.npy', multiplicative_factor=10,
                      num_epochs=10, batch_size=20, learning_rate=0.01, momentum=0.9, validate_split=0.2):
        """  
        Train the neural network corresponding to block1
        """
        # get the train data as a 2D matrix and train labels as a vector
        train_X, train_y = self.loader.get_data(train_csv, gt_csv)
        total_train_X = []
        total_train_y = []
        # get the phi features and the labels in flattened vectors of
        # dimensions p*q
        for s_train_X, s_train_y in utils.augment_set(train_X, train_y, self.num_participants,
                                                      self.num_questions, multiplicative_factor):
            # generate the phi features using this data
            self.feature.generate_features_2d(s_train_X, s_train_y)
            # get the phi features and the labels in flattened vectors of
            # dimensions p*q
            f_train_X, f_train_y = self.feature.get_features_2d()
            total_train_X.append(f_train_X)
            total_train_y.append(f_train_y)
        # split it into train and validation splits
        total_train_X = utils.flatten_2D_list(total_train_X)
        total_train_y = utils.flatten_2D_list(total_train_y)
        train_X = total_train_X
        val_X = total_train_X
        train_y = total_train_y
        val_y = total_train_y
        # train the network
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        val_X = np.array(val_X)
        val_y = np.array(val_y)
        train_y = train_y.astype(int)
        val_y = val_y.astype(int)
        self.block1.train(train_X, train_y, val_X, val_y,
                          num_epochs, batch_size, learning_rate, momentum)
        # save the weights
        self.block1.save_weights(weights_name)

    def predict(self, test_csv, iteration_type='fixed', num_iterations=10):
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
        proposals = utils.majority_voting(input_data, self.num_classes, 2)
        self.majority_voting_props = proposals
        for i in range(0, num_iterations):
            abilities = self.cor(input_data, proposals)
            proposals = self.ref(abilities, input_data)
        return proposals

    def predict_and_evaluate(self, test_csv, gt_csv, iteration_type='fixed', num_iterations=10):
        proposals = self.predict(test_csv, iteration_type, num_iterations)
        proposals_mj = self.majority_voting_props
        gt = self.loader.get_gt(gt_csv)
        self.print_results(proposals, gt)
        self.print_results(proposals_mj, gt)

    def print_results(self, proposals, gt):
        assert(len(proposals) == len(gt))
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(0, len(gt)):
            if gt[i] == 0:
                if proposals[i] == 0:
                    tn += 1
                else:
                    fp += 1
            else:
                if proposals[i] == 0:
                    fn += 1
                else:
                    tp += 1
        total = tp + fp + fn + tn
        print "TP = ", tp
        print "TN = ", tn
        print "FP = ", fp
        print "FN = ", fn
        print "Total = ", total
        print "Sensitivity = ", (float(tp) / (tp + fn))
        print "Specificity = ", (float(tn) / (tn + fp))
        print "Accuracy = ", (float(tp + tn) / (total))

    def cor(self, input_data, proposals):
        """
        The E step
        """
        self.feature.generate_features_2d(input_data, proposals)
        test_X, test_y = self.feature.get_features_2d()
        abilities = self.block1.predict(test_X)
        return abilities

    def ref(self, abilities, input_data):
        """
        The M step
        """
        proposals = []
        ability_matrix = []
        counter = 0
        for i in range(0, self.num_participants):
            ability_element = []
            for j in range(0, self.num_questions):
                ability_element.append(abilities[counter])
                counter += 1
            ability_matrix.append(ability_element)
        ability_matrix = np.array(ability_matrix)
        for j in range(0, self.num_questions):
            ability_vector = ability_matrix[:, j, 0]
            input_vector = input_data[:, j]
            self.block2.fit(ability_vector)
            # TODO use single function in block2 instead of iterating here
            prediction = self.block2.predict_element(input_vector)
            proposals.append(prediction)
        proposals = np.array(proposals)
        return proposals


class EM3D:

    def __init__(self, num_p, num_q, num_opt, k_ability, k_qdifficulty, k_odifficulty):
        """  
        Create blocks and initialize members
        """
        self.loader = loader.Loader()
        self.num_classes = 2
        self.num_participants = num_p
        self.num_questions = num_q
        self.num_options = num_opt
        self.k_ability = k_ability
        self.k_qdifficulty = k_qdifficulty
        self.k_odifficulty = k_odifficulty
        self.feature = feature.FeatureRepresenter_3D(self.num_participants, self.num_questions, self.num_options,
                                                     k_ability, k_qdifficulty, k_odifficulty)
        self.block1 = block1.Block1(k_ability + k_odifficulty + 2)
        self.block2 = block2.Block2(self.num_participants, self.num_classes)

    def train_block_1(self, train_csv, gt_csv, weights_name='block1_weights.npy', multiplicative_factor=10,
                      num_epochs=10, batch_size=20, learning_rate=0.01, momentum=0.9, validate_split=0.2):
        """  
        Train the neural network corresponding to block1
        """
        # get the train data as a 2D matrix and train labels as a vector
        train_X, train_y = self.loader.get_data_3D(train_csv, gt_csv)
        total_train_X = []
        total_train_y = []
        # get the phi features and the labels in flattened vectors of
        # dimensions p*q
        for s_train_X, s_train_y in utils.augment_set(train_X, train_y, self.num_participants,
                                                      self.num_questions, multiplicative_factor):
            # generate the phi features using this data
            self.feature.generate_features_3d(s_train_X, s_train_y)
            # get the phi features and the labels in flattened vectors of
            # dimensions p*q
            f_train_X, f_train_y = self.feature.get_features_3d()
            total_train_X.append(f_train_X)
            total_train_y.append(f_train_y)
        # split it into train and validation splits
        total_train_X = utils.flatten_2D_list(total_train_X)
        total_train_y = utils.flatten_2D_list(total_train_y)
        train_X = total_train_X
        val_X = total_train_X
        train_y = total_train_y
        val_y = total_train_y
        # train the network
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        val_X = np.array(val_X)
        val_y = np.array(val_y)
        train_y = train_y.astype(int)
        val_y = val_y.astype(int)
        self.block1.train(train_X, train_y, val_X, val_y,
                          num_epochs, batch_size, learning_rate, momentum)
        # save the weights
        self.block1.save_weights(weights_name)

    def predict(self, test_csv, iteration_type='fixed', num_iterations=10):
        """  
        Predict the answer for a set of questions
        The test file can contain more than p people and q questions
        This needs to take subsets before predicting
        This will return an array of proposed answers
        """
        # currently assume test.csv contains tests of the correct size
        input_data = self.loader.get_data_3D(test_csv)
        num_participants = len(input_data)
        if num_participants > 0:
            num_questions = len(input_data[0])
        else:
            return
        proposals = utils.majority_voting(input_data, self.num_classes, 3)
        self.majority_voting_props = proposals
        for i in range(0, num_iterations):
            abilities = self.cor(input_data, proposals)
            proposals = self.ref(abilities, input_data)
        return proposals

    def predict_and_evaluate(self, test_csv, gt_csv, iteration_type='fixed', num_iterations=10):
        proposals = self.predict(test_csv, iteration_type, num_iterations)
        proposals_mj = self.majority_voting_props
        gt = self.loader.get_gt_3D(gt_csv)
        self.print_results(proposals, gt)
        self.print_results(proposals_mj, gt)

    def print_results(self, proposals, gt):
        assert(len(proposals) == len(gt))
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        avg_question_accuracy = 0
        avg_options_correct = 0
        for i in range(0, len(gt)):
            all_correct = True
            this_question_correct = 0
            for j in range(0, len(gt[0])):
                if gt[i][j] == 0:
                    if proposals[i][j] == 0:
                        tn += 1
                        this_question_correct += 1
                    else:
                        fp += 1
                        all_correct = False
                else:
                    if proposals[i][j] == 0:
                        fn += 1
                        all_correct = False
                    else:
                        tp += 1
                        this_question_correct += 1
            avg_options_correct += float(this_question_correct) / len(gt[0])
            if(all_correct):
                avg_question_accuracy += 1
        avg_question_accuracy /= float(len(gt))
        avg_options_correct /= float(len(gt))
        total = tp + fp + fn + tn
        print "TP = ", tp
        print "TN = ", tn
        print "FP = ", fp
        print "FN = ", fn
        print "Total = ", total
        print "Sensitivity = ", (float(tp) / (tp + fn))
        print "Specificity = ", (float(tn) / (tn + fp))
        print "Accuracy = ", (float(tp + tn) / (total))
        print "Average Question Accuracy = ", avg_question_accuracy
        print "Average Options Correct = ", avg_options_correct

    def cor(self, input_data, proposals):
        """
        The E step
        """
        self.feature.generate_features_3d(input_data, proposals)
        test_X, test_y = self.feature.get_features_3d()
        abilities = self.block1.predict(test_X)
        return abilities

    def ref(self, abilities, input_data):
        """
        The M step
        """
        proposals = []
        ability_matrix = []
        counter = 0
        for i in range(0, self.num_participants):
            ability_question = []
            for j in range(0, self.num_questions):
                ability_option = []
                for k in range(0, self.num_options):
                    ability_option.append(abilities[counter])
                    counter += 1
                ability_question.append(ability_option)
            ability_matrix.append(ability_question)
        ability_matrix = np.array(ability_matrix)
        for j in range(0, self.num_questions):
            proposal_element = []
            for k in range(0, self.num_options):
                ability_vector = ability_matrix[:, j, k]
                input_vector = input_data[:, j, k]
                self.block2.fit(ability_vector)
                # TODO use single function in block2 instead of iterating here
                prediction = self.block2.predict_element(input_vector)
                proposal_element.append(prediction)
            proposals.append(proposal_element)
        proposals = np.array(proposals)
        return proposals
