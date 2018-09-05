import os

from argparse import ArgumentParser
from deepagg import main


def run(args):
    solve_type = args.type
    mode = args.mode
    crowd_path = args.crowd_annotations_path
    gt_path = args.ground_truths_path
    assert os.path.exists(crowd_path), crowd_path + " does not exist!"
    assert os.path.exists(gt_path), gt_path + " does not exist!"

    num_p = args.num_annotators
    num_q = args.num_questions
    num_opts = args.num_options
    k_ability = args.k_ability
    k_difficulty = args.k_difficulty

    if solve_type == '2D':
        model = main.EM2D(num_p, num_q, k_ability, k_difficulty, num_opts)
    else:
        k_odifficulty = args.k_odifficulty
        model = main.EM3D(num_p, num_q, num_opts, k_ability,
                          k_difficulty, k_odifficulty)

    if mode == 'train':
        assert os.path.exists(
            args.save_path), args.save_path + " does not exist!"
        output_path = os.path.join(args.save_path, args.save_name)
        model.train_block_1(crowd_path, gt_path, weights_name=output_path, multiplicative_factor=args.multiplicative_factor,
                            num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.lr, momentum=args.momentum, validate_split=args.validate_split)
    else:
        load_path = args.load_path
        assert os.path.exists(load_path), load_path + " does not exist!"
        model.block1.load_weights(args.load_path)
        predictions = model.predict_and_evaluate(
            crowd_path, gt, num_iterations=args.num_iterations)
        print("Predictions:")
        print(predictions)

if __name__ == '__main__':
    parser = ArgumentParser(description='DeepAgg')
    parser.add_argument('--crowd_annotations_path', type=str, required=True,
                        help='Path to crowdsourced annotations.')
    parser.add_argument('--ground_truths_path', type=str, required=False,
                        help='Path to ground truths. Used only in test mode.')
    parser.add_argument('--type', default='2D', type=str,
                        choices=['2D', '3D'], help='Type of aggregation - 2D or 3D.')
    parser.add_argument('--mode', default='train', type=str,
                        choices=['train', 'test'], help='Mode - train or test')
    parser.add_argument('--num_annotators', type=int,
                        required=True, help='Number of annotators.')
    parser.add_argument('--num_questions', type=int,
                        required=True, help='Number of questions.')
    parser.add_argument('--num_options', type=int,
                        required=True, help='Number of classes.')
    parser.add_argument('--k_ability', type=int,
                        required=True, help='Number of annotator ability buckets.')
    parser.add_argument('--k_difficulty', type=int,
                        required=True, help='Number of question difficulty buckets')
    parser.add_argument('--k_odifficulty', default=3, type=int,
                        required=False, help='Number of option difficulty buckets. Used only for 3D type. Default is 3.')
    parser.add_argument('--num_epochs', default=10, type=int, required=False,
                        help='Number of training epochs. Used only in the train mode. Default is 10.')
    parser.add_argument('--multiplicative_factor', default=10, type=float, required=False,
                        help='Factor by which training data is augmented. Used only in the train mode. Default is 10.')
    parser.add_argument('--batch_size', default=20, type=int, required=False,
                        help='Number of batches. Used only in the train mode. Default is 20.')
    parser.add_argument('--lr', default=0.01, type=float, required=False,
                        help='Learning Rate. Used only in the train mode. Default is 0.01.')
    parser.add_argument('--momentum', default=0.9, type=float, required=False,
                        help='Momentum factor. Used only in the train mode. Default is 0.9.')
    parser.add_argument('--validate_split', default=0.2, type=float, required=False,
                        help='Fraction of data used for validation. Used only in the train mode. Default is 0.2.')
    parser.add_argument('--num_iterations', default=10, type=int, required=False,
                        help='Number of iterations for aggregation. Used only in the test mode. Default is 10.')
    parser.add_argument('--save_path', default='.', type=str, required=False,
                        help='Path to save trained weights. Used only in the train mode. The default path is the deepagg directory.')
    parser.add_argument('--save_name', default='block1_weights.npy', type=str, required=False,
                        help='File name for saving weights. Used only in the train mode. Default is "block1_weights.npy".')
    parser.add_argument('--load_path', default='./block1_weights.npy', type=str, required=False,
                        help='Load path for saved weights. Used only in the test mode. Default is "deepagg/block1_weights.npy".')
    args = parser.parse_args()
    run(args)
