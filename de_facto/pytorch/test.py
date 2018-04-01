from algs.logistic_regression import LogisticRegressionOptimizer
from data.synthesized import Dataset, load_data

if __name__ == "__main__":
	data = load_data('./data/T_float')
	dataset = Dataset(data)
	lr_trainer = LogisticRegressionOptimizer(lr=0.00001)
	lr_trainer.minimize(dataset)