import argparse
import matplotlib.pyplot as plt

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('files', metavar='N', type=str, nargs='+')
	return parser.parse_args()

def truncate_after_best(accuracies):
	best_idx = best_acc = 0
	for idx, acc in enumerate(accuracies):
		if acc > best_acc:
			best_acc = acc
			best_idx = idx
	return accuracies[:best_idx+1]

def main(args):

	line_begin = 'val accuracy is'
	accuracies = []
	for file_path in args.files:
		if len(accuracies) > 0:
			accuracies = truncate_after_best(accuracies)
		with open(file_path) as fd:
			for line in fd:
				if line[:len(line_begin)] != line_begin:
					continue
				acc = float(line.split()[-1])
				accuracies.append(acc)

	iterations = [ (i+1)*11000 for i in range(len(accuracies)) ]
	plt.figure()
	plt.plot(iterations, accuracies)
	plt.xlabel('Optimization steps')
	plt.ylabel('Validation accuracy')
	plt.show()

if __name__ == '__main__':
	main(get_args())
