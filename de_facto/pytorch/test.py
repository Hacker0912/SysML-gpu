import argparse

from algs.logistic_regression import SCDOptimizer
from data.synthesized import Dataset, load_data

parser = argparse.ArgumentParser(description='GPU benckmarks')
parser.add_argument('--lr', type=float, default=0.01, 
					help='learning rate for opt')
parser.add_argument('--enable-gpu', type=bool, default=False, 
					help='wether/or not enable GPU')
parser.add_argument('--load-in-memory', type=bool, default=False, 
					help='weather load the entire dataset into GPU memory')
parser.add_argument('--max-steps', type=int, default=1000, 
					help='max iterations we run for opt algs')
args = parser.parse_args()

if __name__ == "__main__":
    data = load_data('./data/T_float')
    dataset = Dataset(data)
    args = {'lr':args.lr, 
    		'max_steps':args.max_steps,
    		'enable_gpu':args.enable_gpu, 
    		'load_memory':args.load_in_memory}
    lr_trainer = SCDOptimizer(**args)
    lr_trainer.minimize(dataset)