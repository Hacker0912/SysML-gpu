import argparse

from algs.logistic_regression import SCDOptimizer
from data.synthesized import Dataset, load_data

parser = argparse.ArgumentParser(description='GPU benckmarks')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for opt')
parser.add_argument('--enable-gpu', type=bool, default=False, help='wether/or not enable GPU')
args = parser.parse_args()

if __name__ == "__main__":
    data = load_data('./data/T_float')
    dataset = Dataset(data)
    lr_trainer = SCDOptimizer(lr=args.lr, enable_gpu=args.enable_gpu)
    lr_trainer.minimize(dataset)