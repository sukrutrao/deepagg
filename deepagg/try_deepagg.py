import main

em2d = main.EM2D(10,600,3,3,2)
em2d.train_block_1('../../crowdsourced-data-simulator/data.csv','../../crowdsourced-data-simulator/gt.csv',num_epochs=600)
predictions = em2d.predict_and_evaluate('../../crowdsourced-data-simulator/test_data.csv','../../crowdsourced-data-simulator/test_gt.csv',num_iterations=1)
#print predictions


