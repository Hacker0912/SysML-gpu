import argparse
import importlib
import time

import algs

import torch
from data.synthesized import Dataset, load_data
from data.synthesized_pytorch import SynthesizedDataset

parser = argparse.ArgumentParser(description='GPU benckmarks')

parser.add_argument('alg', choices=algs.__all__, #required=True,
                    help='GLM algorithms to use')

parser.add_argument('--data-dir', type=str, default='./data/T_float', 
                    help='directory of place store data')
parser.add_argument('--optim', type=str, default='SCD', 
                    help='optimization algorithm used, SCD, SGD are supported')
parser.add_argument('--lr', type=float, default=0.01, 
					help='learning rate for opt')
parser.add_argument('--batch-size', type=int, default=1, 
                    help='batch size used in mini-bath SGD')
parser.add_argument('--enable-gpu', type=bool, default=False, 
					help='wether/or not enable GPU')
parser.add_argument('--load-in-memory', type=bool, default=False, 
					help='weather load the entire dataset into GPU memory')
parser.add_argument('--max-steps', type=int, default=1000, 
					help='max iterations we run for opt algs')
args = parser.parse_args()

if __name__ == "__main__":
    alg = importlib.import_module('algs.' + args.alg)

    load_data_start = time.time()
    #data = load_data('./data/T_float')
    data = load_data(args.data_dir)
    load_data_dur = time.time() - load_data_start
    print("Data Loading Time: {}".format(load_data_dur))

    dataset = Dataset(data)
    if args.optim == 'SCD':
        args = {'lr':args.lr, 
                'max_steps':args.max_steps,
                'enable_gpu':args.enable_gpu, 
                'load_memory':args.load_in_memory}
        trainer = alg.SCDOpimizer(**args)
        trainer.minimize(dataset)
    else:
        batch_size = args.batch_size
        args = {'lr':args.lr, 'max_steps':args.max_steps,'enable_gpu':args.enable_gpu, 'load_memory':args.load_in_memory}
        dataset = SynthesizedDataset(dataset)
        train_loader=torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        trainer = alg.SGDOpimizer(**args)
        trainer.minimize(dataset, train_loader)