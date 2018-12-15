"""
This script is used to select some word vectors from raw word vector files
because the raw word vectors are too many to fit into memory.
"""
import os
import argparse

import numpy as np

from utils import SPECIAL_TOKENS

def main(args):

	# Only select words that appear in training data
	token_freqs = {}
	with open(args.path_to_train, 'r') as train_file:
		for line in train_file.readlines():
			tokens = line.split()
			for t in tokens:
				ts = [t]
				for t in ts:
					t = t.lower()
					token_freqs[t] = token_freqs.get(t, 0) + 1

	freqs = list(token_freqs.values())

	print('# tokens:', len(freqs))
	print('Mean frequency:', np.mean(freqs))
	print('Max frequency:', np.amax(freqs))
	print('Min frequency:', np.amin(freqs))

	# Save the tokens from train
	with open(args.path_to_tokens, 'w') as token_file:

		# First save special tokens
		for t in SPECIAL_TOKENS:
			token_file.write(t + '\n')

		# Then for all tokens in train data
		for t in token_freqs:
			if token_freqs[t] >= args.freq_threshold:
				token_file.write(t + '\n')

	# Explore the original word vector file
	all_tokens = set()
	selected_tokens = set()
	with open(args.path_to_input, 'r') as in_file:
		with open(args.path_to_output, 'w') as out_file:
			for line in in_file.readlines():
				token = line.split()[0].strip().lower()
				all_tokens.add(token)
				if token_freqs.get(token, 0) >= args.freq_threshold:
					# Token is frequent enough so we want to use its word vector
					out_file.write(line.strip())
					out_file.write('\n')
					selected_tokens.add(token)

	# Print out some sample words that are in our train file
	# but not in fasttext word vectors
	ctr = 0
	print('======================')
	for t in token_freqs:
		if t not in selected_tokens:
			print(t)
			ctr += 1
			if ctr >= 20: break
	print('======================')

	print('{} / {} selected'.format(len(selected_tokens), len(all_tokens)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path_to_input', default=None)
	parser.add_argument('--path_to_output', default=None)
	parser.add_argument('--path_to_train', default=None)
	parser.add_argument('--path_to_tokens', default=None)
	parser.add_argument('--freq_threshold', type=int, default=1)
	args = parser.parse_args()
	main(args)