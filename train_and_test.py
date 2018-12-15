"""
This script contains the training and testing pipeline,
"""
import os
import time
import random
import argparse
from io import StringIO

import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable

from sacrebleu import corpus_bleu
from models import RNNEncoder, RNNDecoder, SelfAttentionEncoder
import models
from utils import SPECIAL_TOKENS

MAX_ITRS = int(1e6)

EMBEDDING_DIM = 300

LOG_PER_ITRS = 10
VAL_PER_ITRS = 200

np.random.seed(2018)

class TranslationGenerator:
    def __init__(self,
                 batch_size,
                 source_lang,
                 path_to_data,
                 mode,
                 source_token_to_idx,
                 target_token_to_idx,
                 max_len,
                 should_loop=True,
                 should_shuffle=True):
        """
        Load the tokens in the text file to batches of sequence data
        """

        self.batch_size = batch_size
        self.mode = mode
        self.max_len = max_len
        self.should_loop = should_loop
        self.should_shuffle = should_shuffle

        self.source_sentences = []
        self.target_sentences = []
        self.raw_source_sentences = []
        self.raw_target_sentences = []

        def load_a_sentence(line, sentences, token_to_idx, is_target=False):
            """
            Helper function to load a sentence to a sequence of integers
            """
            s = []
            for t in line.split():
                ts = [t]
                for t in ts:
                    t = t.lower()
                    # If encounter an unknown token,
                    # use the <unk> embedding vector
                    if t in token_to_idx:
                        i = token_to_idx[t]
                    else:
                        i = token_to_idx['<unk>']
                    s.append(i)
            if is_target:
                s = [token_to_idx['<start>']] + s + [token_to_idx['<eos>']]
            s = s[:max_len]
            if len(s) > 0:
                sentences.append(torch.LongTensor(s))

        # Load source sentences to sequences of integers
        ignore_line_idx = set()
        with open(os.path.join(path_to_data, mode + '.tok.' + source_lang), 'r') as f:
            ctr = 0
            for line in f.readlines():
                self.raw_source_sentences.append(line.strip().lower())
                before_ctr = len(self.source_sentences)
                load_a_sentence(line.strip(), self.source_sentences, source_token_to_idx, is_target=False)
                after_ctr = len(self.source_sentences)
                if after_ctr == before_ctr:
                    ignore_line_idx.add(ctr)
                ctr += 1

        # Load target sentences to sequences of integers
        with open(os.path.join(path_to_data, mode + '.tok.' + 'en'), 'r') as f:
            ctr = -1
            for line in f.readlines():
                ctr += 1
                if ctr in ignore_line_idx:
                    continue
                self.raw_target_sentences.append(line.strip().lower())
                load_a_sentence(line.strip(), self.target_sentences, target_token_to_idx, is_target=True)

        assert len(self.source_sentences) == len(self.target_sentences)

        print('Loaded {:3,} pairs of sentences for {}'.format(len(self.source_sentences), self.mode))

        self.indexes = list(range(len(self.source_sentences)))
        self.current_idx = 0
        self.data_size = len(self.indexes)

    def reset(self):
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of padded source sequences and target sequences,
        along with their seq lengths.
        """

        # Reset data pointer for a new epoch
        if self.current_idx + self.batch_size > len(self.source_sentences):
            if self.should_shuffle:
                random.shuffle(self.indexes)
            self.current_idx = 0

        X, X_seq_lens, y, y_seq_lens = [], [], [], []

        raw_X, raw_y = [], []

        for i in range(self.batch_size):
            source_seq = self.source_sentences[self.indexes[self.current_idx]]
            target_seq = self.target_sentences[self.indexes[self.current_idx]]
            raw_X.append(self.raw_source_sentences[self.indexes[self.current_idx]])
            raw_y.append(self.raw_target_sentences[self.indexes[self.current_idx]])
            X.append(source_seq)
            X_seq_lens.append(len(source_seq))
            y.append(target_seq)
            y_seq_lens.append(len(target_seq))
            self.current_idx += 1

        return raw_X, raw_y, pad_sequence(X, batch_first=True), X_seq_lens, pad_sequence(y, batch_first=True), y_seq_lens

def load_word_embeddings(path_to_embeddings, tokens, train_embed):
    """
    Helper function to load word embeddings
    """
    model = KeyedVectors.load_word2vec_format(path_to_embeddings)

    embeddings_val = []

    for i, token in enumerate(tokens):
        if token in model:
            embeddings_val.append(model[token])
        else:
            embeddings_val.append(1 - 2 * np.random.randn(EMBEDDING_DIM))

    embeddings = torch.FloatTensor(embeddings_val)

    return embeddings

def load_tokens(path_to_tokens):
    tokens = []

    with open(path_to_tokens, 'r') as f:
        for line in f.readlines():
            tokens.append(line.strip())

    return tokens

def compute_loss(criterion, output_log_softmax, y):
    """
    Helper function to compute loss based on the decoding results
    """

    vocab_size = list(output_log_softmax.size())[2]

    output_log_softmax = output_log_softmax.contiguous().view(-1, vocab_size)
    y = y.contiguous().view(-1)

    loss = criterion(output_log_softmax, y)

    return loss

def corpus_predict(data_generator, encoder, decoder, idx_to_token, beam=1, gpu=False, val_size=100):
    """
    Run inference on a corpus of data.

    Return two stream pointers, reference and prediction.
    """
    def decode_token(seqs):
        """
        Decode a batch of padded integer sequences to a batch of token sequences
        """
        token_seqs = []

        if gpu:
            seqs = seqs.cpu().numpy().tolist()
        else:
            seqs = seqs.numpy().tolist()

        for i in range(len(seqs)):
            s = []
            for j in range(len(seqs[i])):
                idx = seqs[i][j]
                s.append(idx_to_token[idx])
                if idx == SPECIAL_TOKENS.index('<eos>'):
                    break
            token_seqs.append(s)

        token_seqs = [' '.join(s) for s in token_seqs]

        return token_seqs

    # Make sure model is on eval mode
    # which is important for any batch normalization and dropout layers
    encoder.eval()
    decoder.eval()

    source_sentences = []
    ref_sentences = []
    pred_sentences = []

    val_size = min(val_size, data_generator.data_size)
    val_itrs = val_size // data_generator.batch_size

    # Reset data generator so we can always start from the beginning
    data_generator.reset()

    total_time = 0

    for _ in tqdm(range(val_itrs)):
        raw_X, raw_y, X, X_seq_lens, y, y_seq_lens = next(data_generator)
        if gpu:
            X = X.cuda()
            y = y.cuda()
        source_sentences.append(raw_X)
        ref_sentences.append(raw_y)

        start_time = time.time()

        # Forward pass - encoder
        encoder_output, h_n = encoder(X, input_lengths=X_seq_lens, gpu=gpu)
        # Forward pass - decoder
        output_log_softmax, preds = decoder(encoder_output, h_n, gpu=gpu, beam=beam)
        
        end_time = time.time()
        total_time += end_time - start_time

        # Convert predicted indexes to actual tokens
        pred_tokens = decode_token(preds)
        pred_sentences.append(pred_tokens)

    source_str = '\n'.join(['\n'.join(s) for s in source_sentences])
    ref_str = '\n'.join(['\n'.join(s) for s in ref_sentences])
    raw_pred_str = '\n'.join(['\n'.join(s) for s in pred_sentences])

    def filter_raw_pred(ss):
        ret = []
        for t in ss.split():
            if t not in SPECIAL_TOKENS:
                ret.append(t)
        return ' '.join(ret)
    pred_str = '\n'.join(['\n'.join([filter_raw_pred(ss) for ss in s]) for s in pred_sentences])

    for i in range(ctr):
        print('='*20)
        print(source_str.split('\n')[i])
        print(ref_str.split('\n')[i])
        print(pred_str.split('\n')[i])

    print('='*30)
    print('Source:', source_str.split('\n')[0])
    print('')
    print('Reference:', ref_str.split('\n')[0])
    print('')
    print('Predicted:', pred_str.split('\n')[0])
    print('')
    print('Average Inference Time: {} seconds / batch of sentence'.format(total_time / val_itrs))
    print('='*30)

    # Compute average BLEU per reference sequence length
    bleu_by_length = {}
    ref_str_sentences = ref_str.split('\n')
    pred_str_sentences = pred_str.split('\n')
    assert len(ref_str_sentences) == len(pred_str_sentences)
    total_bleu = 0

    for i in range(len(ref_str_sentences)):
        pred_s = pred_str_sentences[i]
        ref_s = ref_str_sentences[i]
        ref_len = len(ref_s.split())
        sentence_bleu = corpus_bleu(pred_s, ref_s, tokenize='none', lowercase=True)
        total_bleu += sentence_bleu.score
        if ref_len not in bleu_by_length:
            bleu_by_length[ref_len] = []
        bleu_by_length[ref_len].append(sentence_bleu.score)

    for l in bleu_by_length:
        bleu_by_length[l] = np.mean(bleu_by_length[l])
    avg_bleu = total_bleu / len(ref_str_sentences)
    print('Average BLEU score:', avg_bleu)

    return pred_str, ref_str

def save_model(encoder, decoder, path_to_log):
    """
    Save the parameters of encoder and decoder
    """
    if path_to_log is None:
        return

    if not os.path.isdir(path_to_log):
        os.mkdir(path_to_log)

    torch.save(encoder.state_dict(), os.path.join(path_to_log, 'encoder'))
    torch.save(decoder.state_dict(), os.path.join(path_to_log, 'decoder'))

def load_model(encoder, decoder, path_to_log, gpu):
    """
    Load the parameters of encoder and decoder
    """
    if path_to_log is None or not os.path.isdir(path_to_log):
        return

    map_location = 'cpu' if not gpu else 'cuda:0'

    encoder.load_state_dict(torch.load(os.path.join(path_to_log, 'encoder'), map_location=map_location))
    decoder.load_state_dict(torch.load(os.path.join(path_to_log, 'decoder'), map_location=map_location))

def main(args):

    # Load tokens
    source_tokens = load_tokens(os.path.join(args.path_to_embeddings, args.lang + '.tok'))
    target_tokens = load_tokens(os.path.join(args.path_to_embeddings, 'en.tok'))
    source_idx_to_token, source_token_to_idx = {}, {}
    target_idx_to_token, target_token_to_idx = {}, {}
    for i, t in enumerate(source_tokens):
        source_idx_to_token[i] = t
        source_token_to_idx[t] = i
    for i, t in enumerate(target_tokens):
        target_idx_to_token[i] = t
        target_token_to_idx[t] = i

    # Load word embeddings
    train_embed = True if str(args.train_embed).lower() == 'true' else False
    source_word_embeddings = load_word_embeddings(os.path.join(args.path_to_embeddings, 'wiki.'+args.lang+'.vec'), source_tokens, train_embed)
    target_word_embeddings = load_word_embeddings(os.path.join(args.path_to_embeddings, 'wiki.en.vec'), target_tokens, train_embed)
    print('Source word embeddings size:', source_word_embeddings.size())
    print('Target word embeddings size:', target_word_embeddings.size())

    # Build network
    if str(args.self_attention).lower() == 'true':
        encoder = SelfAttentionEncoder(source_word_embeddings,
                                       args.encode_max_len,
                                       hidden_size=args.hidden_size)
    else:
        encoder = RNNEncoder(source_word_embeddings,
                             bidirectional=True,
                             hidden_size=args.hidden_size,
                             num_hidden_layers=args.num_hidden_layers)

    decoder = RNNDecoder(target_word_embeddings,
                         args.decode_max_len,
                         True if str(args.attention).lower() == 'true' else False,
                         hidden_size=args.hidden_size,
                         num_hidden_layers=args.num_hidden_layers)

    gpu = True if str(args.gpu).lower() == 'true' else False

    if gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        if str(args.self_attention).lower() == 'true':
            for i in range(encoder.num_blocks):
                encoder.encoder_blocks[i] = encoder.encoder_blocks[i].cuda()
                for j in range(encoder.encoder_blocks[i].num_attention_heads):
                    encoder.encoder_blocks[i].Qs[j] = encoder.encoder_blocks[i].Qs[j].cuda()
                    encoder.encoder_blocks[i].Ks[j] = encoder.encoder_blocks[i].Ks[j].cuda()
                    encoder.encoder_blocks[i].Vs[j] = encoder.encoder_blocks[i].Vs[j].cuda()

    if args.path_to_log is not None:
        load_model(encoder, decoder, args.path_to_log, gpu)
    print('Encoder and decoder built.')

    should_save_model = True if str(args.save_model).lower() == 'true' else False

    # Define loss function
    # Because the output from our decoder is log softmax
    # here we use negative log likelihood function
    # so that the end results are just cross entropy loss
    criterion = nn.NLLLoss(ignore_index=SPECIAL_TOKENS.index('<pad>'))

    if args.mode == 'train':

        # Prepare data
        train_data_generator = TranslationGenerator(args.batch_size,
                                                    args.lang,
                                                    args.path_to_data,
                                                    'train',
                                                    source_token_to_idx,
                                                    target_token_to_idx,
                                                    args.encode_max_len)
        val_data_generator = TranslationGenerator(args.batch_size,
                                                  args.lang,
                                                  args.path_to_data,
                                                  'dev',
                                                  source_token_to_idx,
                                                  target_token_to_idx,
                                                  args.decode_max_len,
                                                  should_shuffle=False)

        # Summarize model parameters for training
        params = [source_word_embeddings, target_word_embeddings]
        params += list(encoder.parameters()) + list(decoder.parameters())

        # Define an Adam optimizer
        optimizer = optim.Adam(params, lr=args.lr)

        best_val_bleu = 0

        val_size = val_data_generator.data_size if args.val_size <= 0 else args.val_size

        losses, val_bleus = [], []

        for itr in range(MAX_ITRS):

            encoder.train()
            decoder.train()

            # Get data
            raw_X, raw_y, X, X_seq_lens, y, y_seq_lens = next(train_data_generator)

            if gpu:
                X = X.cuda()
                y = y.cuda()

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass - encoder
            encoder_output, h_n = encoder(X, input_lengths=X_seq_lens, gpu=gpu)

            # Forward pass - decoder
            output_log_softmax, preds = decoder(encoder_output, h_n, gpu=gpu, y=y, y_seq_lens=y_seq_lens)

            # Compute loss
            loss = compute_loss(criterion, output_log_softmax, y[:, 1:])

            losses.append(loss.item())

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Do some logging
            if itr % LOG_PER_ITRS == 0:
                print('Itr {}, Loss: {}'.format(itr, loss.item()))

            # Validation
            if itr % VAL_PER_ITRS == 0:
                pred_stream, ref_stream = corpus_predict(val_data_generator,
                                                         encoder,
                                                         decoder,
                                                         target_idx_to_token,
                                                         gpu=gpu,
                                                         val_size=val_size,
                                                         beam=args.beam)
                val_bleu = corpus_bleu(pred_stream, ref_stream, tokenize='none', lowercase=True)
                print('{}, Validation BLEU: {}'.format(time.strftime("%Y-%m-%d %H:%M"), val_bleu))
                val_bleus.append(val_bleu)

                # Save losses
                if args.path_to_log is not None:

                    if not os.path.isdir(args.path_to_log):
                        os.mkdir(args.path_to_log)

                    # Record losses
                    with open(os.path.join(args.path_to_log, 'losses'), 'a') as f:
                        for l in losses:
                            f.write(str(l))
                            f.write('\n')
                    with open(os.path.join(args.path_to_log, 'val_bleus'), 'a') as f:
                        for b in val_bleus:
                            f.write(str(b.score))
                            f.write('\n')

                    # Reset losses
                    losses, val_bleus = [], []

                # Save model
                if itr > 0 and args.path_to_log is not None and should_save_model and val_bleu.score > best_val_bleu:
                    best_val_bleu = val_bleu.score
                    save_model(encoder, decoder, args.path_to_log)
                    print('Saved model to {}'.format(args.path_to_log))

    elif args.mode == 'test':
        test_data_generator = TranslationGenerator(1,
                                                   args.lang,
                                                   args.path_to_data,
                                                   'test',
                                                   source_token_to_idx,
                                                   target_token_to_idx,
                                                   args.decode_max_len,
                                                   should_shuffle=False)
        val_size = test_data_generator.data_size if args.val_size <= 0 else args.val_size
        pred_stream, ref_stream = corpus_predict(test_data_generator,
                                                 encoder,
                                                 decoder,
                                                 target_idx_to_token,
                                                 gpu=gpu,
                                                 val_size=val_size,
                                                 beam=args.beam)
        test_bleu = corpus_bleu(pred_stream, ref_stream, tokenize='none', lowercase=True)
        print('{}, Testing BLEU: {}'.format(time.strftime("%Y-%m-%d %H:%M"), test_bleu))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--lang', choices=['zh', 'vi'], default='zh', help='language')
    parser.add_argument('--path_to_data', default=None)
    parser.add_argument('--path_to_log', default=None)
    parser.add_argument('--path_to_embeddings', default=None)
    parser.add_argument('--lr', type=float, default='1e-3', help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--attention', default=False, help='Whether to use attention in decoding')
    parser.add_argument('--decode_max_len', type=int, default=100)
    parser.add_argument('--encode_max_len', type=int, default=100)
    parser.add_argument('--train_embed', default=True)
    parser.add_argument('--logs_dir', default=None)
    parser.add_argument('--gpu', default=False)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--val_size', type=int, default=0)
    parser.add_argument('--save_model', default=True)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--self_attention', default=False)

    main(parser.parse_args())
