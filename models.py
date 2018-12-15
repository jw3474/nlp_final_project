import math

import torch
import torch.nn as nn
import numpy as np

DEFAULT_HIDDEN_SIZE = 128

DEFAULT_NUM_HIDDEN_LAYERS = 1

PAD_IDX = 0
EOS_IDX = 1
START_IDX = 2
UNK_IDX = 3

GPU = False

class BaseRNN(nn.Module):
    def __init__(self,
                 embedding,
                 bidirectional,
                 hidden_size=DEFAULT_HIDDEN_SIZE,
                 num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS):

        super().__init__()

        # Set attributes
        self.vocab_size, self.embedding_dim = embedding.size()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_hidden_layers = num_hidden_layers

        # Embedding lookup module
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight = nn.Parameter(embedding)
        self.embedding.requires_grad = True

        # RNN cell
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_size, self.num_hidden_layers,
                          batch_first=True, bidirectional=self.bidirectional)

    def embedding_lookup(self, x, input_lengths=None):
        """
        Helper function to extract embeddings for selected indexes
        Inputs:
            x, padded (with PAD_IDX) sequences
            input_lengths, original lengths of the padded sequences
        """
        selected_embeddings = self.embedding(x)

        return selected_embeddings

    def forward(self, x, input_lengths=None):
        pass

class SelfAttentionEncoderBlock(nn.Module):
    def __init__(self,
                 max_len,
                 num_attention_heads=8,
                 hidden_size=512,
                 q_size=64,
                 k_size=64,
                 v_size=64):

        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.q_size = q_size
        self.k_size = k_size
        self.v_size = v_size
        self.max_len = max_len

        # Define linear layers to help transform input vectors
        # to query, key, and value vectors
        self.Qs, self.Ks, self.Vs = [], [], []
        for h in range(self.num_attention_heads):
            self.Qs.append(nn.Linear(self.hidden_size, self.q_size))
            self.Ks.append(nn.Linear(self.hidden_size, self.k_size))
            self.Vs.append(nn.Linear(self.hidden_size, self.v_size))

        self.shrink_head = nn.Linear(self.v_size * self.num_attention_heads, self.hidden_size)

        # Feed forward layer
        self.first_feed_forward = nn.Linear(self.hidden_size, 4*self.hidden_size)
        self.second_feed_forward= nn.Linear(4*self.hidden_size, self.hidden_size)

        # Layer Normal layer
        self.first_layer_norm = nn.LayerNorm(self.hidden_size)
        self.second_layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x, input_lengths=None, gpu=False):
        """
        Inputs:
            x: [batch_size, max_len, hidden_size]

        Outputs:
            z: [batch_size, max_len, hidden_size]
        """

        batch_size, seq_len = list(x.size())[0], list(x.size())[1]

        # Multi Heads
        self.Zs = []
        for h in range(self.num_attention_heads):
            Q = self.Qs[h](x)
            K = self.Ks[h](x)
            V = self.Vs[h](x)
            Kt = torch.transpose(K, 1, 2)
            raw_scores = torch.matmul(Q, Kt) / np.sqrt(self.k_size)
            # Mask raw scores by input lengths
            # so that padded tokens won't contaminate the following softmax scores
            # raw_scores has shape [batch_size, max_len, max_len]
            mask = []
            for i in range(batch_size):
                mask.append([input_lengths[i] * [1] + (seq_len - input_lengths[i]) * [0]])
            mask = torch.FloatTensor(mask)
            if gpu:
                mask = mask.cuda()
            mask = mask.expand(batch_size, seq_len, seq_len)
            masked_scores = raw_scores * mask
            scores = nn.functional.softmax(masked_scores)
            z = torch.matmul(scores, V)
            self.Zs.append(z.unsqueeze(dim=2))
        # Convert Zs to shape [batch_size, max_len, num_heads, v_size]
        self.Zs = torch.cat(self.Zs, dim=2)
        batch_size, max_len, _, _ = list(self.Zs.size())
        self.Zs = self.Zs.view(batch_size, max_len, self.num_attention_heads * self.v_size)

        # Shrink heads to [batch_size, max_len, hidden_size]
        z0 = self.shrink_head(self.Zs)

        # Add & Normalize
        z1 = self.first_layer_norm(z0 + x)

        # Feed forward
        z2 = self.first_feed_forward(z1)
        z2 = nn.functional.relu(z2)
        z2 = self.second_feed_forward(z2)

        # Add & Normalize
        z3 = self.second_layer_norm(z1 + z2)

        return z3

class SelfAttentionEncoder(nn.Module):
    def __init__(self,
                 embedding,
                 max_len,
                 num_blocks=6,
                 hidden_size=512):
        super().__init__()

        # Set attributes
        self.max_len = max_len
        self.num_blocks = num_blocks
        self.vocab_size, self.embedding_dim = embedding.size()
        self.hidden_size = hidden_size

        assert self.embedding_dim == self.hidden_size

        # Embedding lookup module
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight = nn.Parameter(embedding)
        self.embedding.requires_grad = True

        self.encoder_blocks = []
        for i in range(self.num_blocks):
            self.encoder_blocks.append(SelfAttentionEncoderBlock(self.max_len, hidden_size=self.hidden_size))

        # Generate positional encodings
        pe = torch.zeros(max_len, self.hidden_size)
        for pos in range(max_len):
            for i in range(0, self.hidden_size, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/self.hidden_size)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/self.hidden_size)))

        self.positional_encodings = nn.Parameter(pe)
        self.positional_encodings.requires_grad = False

        self.zero_tensor = None

    def embedding_lookup(self, x, input_lengths=None):
        """
        Helper function to extract embeddings for selected indexes
        Inputs:
            x, padded (with PAD_IDX) sequences
            input_lengths, original lengths of the padded sequences
        """
        selected_embeddings = self.embedding(x)

        return selected_embeddings

    def forward(self, x, input_lengths=None, gpu=False):
        """
        Implement self-attention architecture

        Inputs:
            x: [batch_size, max_len]
        """

        # First pad x to max_len
        # batch_size, cur_len = list(x.size())
        # if self.zero_tensor is None:
        #     # Create the zero tensor
        #     self.zero_tensor = torch.LongTensor(np.zeros((batch_size, 1), np.int32))
        #     if gpu:
        #         self.zero_tensor = self.zero_tensor.cuda()
        # if cur_len < self.max_len:
        #     x = torch.cat([x, self.zero_tensor.expand(batch_size, self.max_len - cur_len)], dim=-1)

        x_embeddings = self.embedding_lookup(x, input_lengths)

        # Convert embeddings to have the same shape as hidden embeddings
        # z shape [batch_size, max_len, hidden_size]
        # z = self.post_process_embedding(x_embeddings)

        # Positional encoding
        pe = self.positional_encodings[:list(x.size())[1], :]

        # Add positional information to the input
        z = pe + x_embeddings

        # Run through self-attention-based encoder blocks
        for i in range(self.num_blocks):
            z = self.encoder_blocks[i](z, input_lengths=input_lengths, gpu=gpu)

        # While z has shape [batch_size, seq_len, hidden_size]
        # h_n should have shape [num_layers*num_directions, batch_size, hidden_size]
        #h_n = torch.stack([zz[input_lengths[i]-1, :] for i, zz in enumerate(z)], dim=0)
        h_n = z[:, 0, :]
        h_n = h_n.unsqueeze(0)
        h_n = h_n.expand(2, list(h_n.size())[1], list(h_n.size())[2])

        # output z should have shape [batch_size, max_len, 2*self.hidden_size]
        batch_size, max_len, _ = list(z.size())
        z = z.unsqueeze(dim=2)
        z = z.expand(batch_size, max_len, 2, self.hidden_size)
        z = z.contiguous().view(batch_size, max_len, 2*self.hidden_size)

        return z, h_n


class RNNEncoder(BaseRNN):
    def __init__(self,
                 embedding,
                 bidirectional=True,
                 hidden_size=DEFAULT_HIDDEN_SIZE,
                 num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS):
        super().__init__(embedding,
                         bidirectional,
                         hidden_size,
                         num_hidden_layers)

    def forward(self, x, input_lengths=None, gpu=False):
        batch_size = list(x.size())[0]

        # Embedding lookup
        x = self.embedding_lookup(x, input_lengths)

        # Sort batch by input_lengths
        ascending_indexes = np.argsort(input_lengths)
        descending_indexes = list(reversed(ascending_indexes.tolist()))
        old_to_new = {old: new for new, old in enumerate(descending_indexes)}
        restore_indexes = [old_to_new[i] for i in range(batch_size)]

        descending_indexes = torch.LongTensor(descending_indexes)
        restore_indexes = torch.LongTensor(restore_indexes)
        input_lengths = torch.LongTensor(list(reversed(np.sort(input_lengths).tolist())))

        if gpu:
            descending_indexes = descending_indexes.cuda()
            restore_indexes = restore_indexes.cuda()
            input_lengths = input_lengths.cuda()

        x = torch.index_select(x, 0, descending_indexes)

        # Pack embeddings
        packed_x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        # RNN encoding
        output, h_n = self.rnn(packed_x)

        # Unpack output
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Restore the original batch order
        output = torch.index_select(output, 0, restore_indexes)
        h_n = torch.transpose(h_n, 0, 1)
        h_n = torch.index_select(h_n, 0, restore_indexes)
        h_n = torch.transpose(h_n, 0, 1)

        return output, h_n

class RNNDecoder(BaseRNN):
    def __init__(self,
                 embedding,
                 max_len,
                 attention=False,
                 hidden_size=DEFAULT_HIDDEN_SIZE,
                 num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS):

        super().__init__(embedding,
                         bidirectional=False,
                         hidden_size=hidden_size,
                         num_hidden_layers=num_hidden_layers)

        self.max_len = max_len
        self.attention = attention

        self.logit = nn.Linear(2 * self.hidden_size if self.attention else self.hidden_size, self.vocab_size)
        self.log_softmax = nn.LogSoftmax()

    def forward(self,
                encoder_output,
                initial_hidden,
                gpu=False,
                y=None,
                y_seq_lens=None,
                beam=1):
        """
        Decoding implementation.
        Manully unroll RNN cell to decode.

        Inputs:
            encoder_output, shape [batch_size, max_encoder_input_len, hidden_size]
            initial_hidden, initial hidden state for decoder rnn,
                            shape [batch_size, num_hidden_layers, hidden_size]
            input_lengths, variable input lengths for the encoding stage
        """

        def forward_one_step(encoder_output, k_last_pred, k_hidden_state, k_log_prob_sum):
            """
            Helper function to unroll for one step

            Inputs:
                encoder_output, all output from the encoder
                k_last_pred, [k, batch_size, 1]
                k_hidden_state, [k, num_layers*num_directions, batch_size, hidden_size]
                k_log_prob_sum, [k, batch_size, 1]

            Outputs:
                next_k_log_softmax: [k, batch_size, vocab_size],
                                    log softmax output for k best last predited tokens
                next_k_hidden_state: [k, num_layers*num_directions,
                                     batch_size, hidden_size], hidden state for k best last predicted tokens
                next_k_log_prob_sum: [k, batch_size, vocab_size],
                                     log prob of the sequence so far for k best last predicted tokens

            """

            next_k_log_softmax = []
            next_k_hidden_state = []
            next_k_log_prob_sum = []

            for i in range(beam):

                # Get last_pred for current candidate
                last_pred = k_last_pred[i, :, :]
                hidden_state = k_hidden_state[i, :, :, :]
                hidden_state = hidden_state.contiguous()

                selected_embeddings = self.embedding_lookup(last_pred)
                output, next_hidden_state = self.rnn(selected_embeddings, hidden_state)

                if self.attention:
                    # If attention model, then output needs to adjusted
                    assert list(output.size())[1] == 1

                    batch_size, num_steps, _ = list(encoder_output.size())

                    # Extract the encoder output of forward direction
                    encoder_output_forward = encoder_output.view(batch_size, num_steps, 2, self.hidden_size)[:, :, 0, :]

                    # Compute attention scores
                    expand_num_steps = list(encoder_output_forward.size())[1]
                    tiled_output = output.expand(batch_size, expand_num_steps, self.hidden_size)
                    attention_scores = torch.sum(encoder_output_forward * tiled_output, dim=2)

                    assert len(list(attention_scores.size())) == 2

                    # Compute attention distribution
                    attention_probs = nn.functional.softmax(attention_scores - torch.max(attention_scores, dim=1)[0].view(batch_size, 1))

                    # Compute weighted sum of encoder output
                    tiled_attention_probs = attention_probs.view(batch_size, expand_num_steps, 1).expand(batch_size, expand_num_steps, self.hidden_size)
                    attention_out = torch.sum(encoder_output_forward * tiled_attention_probs, dim=1)

                    # Concatenate the weighted sum and the latest hidden state
                    output = torch.cat([attention_out, output.squeeze(dim=1)], dim=1)

                    # Expand the time step dimension to 1
                    # because the following code relies on this assumption
                    output = output.unsqueeze(dim=1)

                    assert list(output.size())[2] == 2 * self.hidden_size

                logit = self.logit(output[:, -1, :])
                log_softmax = self.log_softmax(logit)

                next_k_log_softmax.append(log_softmax)
                next_k_hidden_state.append(next_hidden_state)

                log_prob_sum = log_softmax + k_log_prob_sum[i, :, :]

                next_k_log_prob_sum.append(log_prob_sum)

            next_k_log_softmax = torch.stack(next_k_log_softmax)
            next_k_hidden_state = torch.stack(next_k_hidden_state)
            next_k_log_prob_sum = torch.stack(next_k_log_prob_sum)

            return next_k_log_softmax, next_k_hidden_state, next_k_log_prob_sum

        def select_top_k(k_log_softmax_full, k_hidden_state_full, k_log_prob_sum_full):
            """
            Select top k candidates based on log prob sum

            Inputs:
                k_log_softmax_full: [k, batch_size, vocab_size]
                k_hidden_state_full: [k, num_layers*num_directions, batch_size, hidden_size],
                                     new hidden state for best k last predicted tokens.
                                     hidden state will remain the same shape but need to re-sorted.
                k_log_prob_sum_full: [k, batch_size, vocab_size],
                                     log prob of sequence so far for best k last predicted tokens.
            
            Outputs:
                top_k_indexes_k: [batch_size, k], sorted indexes of k importance order
                k_best_pred, [k, batch_size, 1]
                k_log_softmax, [k, batch_size, vocab_size]
                k_hidden_state: [k, num_layers*num_directions, batch_size, hidden_size]
                k_log_prob_sum: [k, batch_size, 1]

            """
            # print('k_log_softmax_full:', k_log_softmax_full.size())
            # print('k_hidden_state_full:', k_hidden_state_full.size())
            # print('k_log_prob_sum_full:', k_log_prob_sum_full.size())

            k, batch_size, vocab_size = list(k_log_prob_sum_full.size())

            # Reshape k_log_softmax_full to prepare for top k computation
            k_log_prob_sum_full_by_batch = torch.transpose(k_log_prob_sum_full, 0, 1)
            k_log_prob_sum_full_flat = k_log_prob_sum_full_by_batch.contiguous().view(batch_size, k * vocab_size)

            # Compute top k
            top_k_log_prob_sum, top_k_indexes = torch.topk(k_log_prob_sum_full_flat, k=k)
            # print('top_k_log_prob_sum:', top_k_log_prob_sum.size())

            # Convert the flattened indexes to k index and vocab index
            top_k_indexes_k = top_k_indexes // vocab_size
            top_k_indexes_vocab = top_k_indexes % vocab_size

            # Re-arrange k_hidden_state according to sorted indexes
            k_hidden_state_full = torch.transpose(k_hidden_state_full, 1, 2)
            k_hidden_state_full_transposed = torch.transpose(k_hidden_state_full, 0, 1)
            # Now we know k_hidden_state_full_transposed is in shape [batch_size, k, num_layers*num_directions, hidden_size]
            k_hidden_state = torch.stack([torch.index_select(h, 0, i) for h, i in zip(k_hidden_state_full_transposed, top_k_indexes_k)])
            k_hidden_state = torch.transpose(k_hidden_state, 0, 1)
            k_hidden_state = torch.transpose(k_hidden_state, 1, 2)

            # Re-arrange k_log_softmax
            k_log_softmax = torch.transpose(k_log_softmax_full, 0, 1)
            # print('k_log_softmax after first transpose:', k_log_softmax.size())
            # print(top_k_indexes_k)
            k_log_softmax = torch.stack([torch.index_select(h, 0, i) for h, i in zip(k_log_softmax, top_k_indexes_k)])
            k_log_softmax = torch.transpose(k_log_softmax, 0, 1)
            # print('k_log_softmax after second transpose:', k_log_softmax.size())

            # Update k_log_prob_sum
            k_log_prob_sum = torch.transpose(top_k_log_prob_sum, 0, 1)
            k_log_prob_sum = k_log_prob_sum.unsqueeze(dim=2)

            # Get k best predictions
            k_best_pred = torch.transpose(top_k_indexes_vocab, 0, 1)
            k_best_pred = k_best_pred.unsqueeze(dim=2)

            # print('top_k_indexes_k shape:', top_k_indexes_k.size())
            # print('k_best_pred shape:', k_best_pred.size())
            # print('k_log_softmax:', k_log_softmax.size())
            # print('k_hidden_state:', k_hidden_state.size())
            # print('k_log_prob_sum:', k_log_prob_sum.size())

            return top_k_indexes_k, k_best_pred, k_log_softmax, k_hidden_state, k_log_prob_sum

        if y is not None:
            assert beam == 1

        output_log_softmax_so_far = []
        preds_so_far = []
        parents = []

        # Initialize input and hidden state
        batch_size = encoder_output.size()[0]
        hidden_state = initial_hidden
        hidden_state = hidden_state.view(self.num_hidden_layers, 2, batch_size, self.hidden_size)[:, 0, :, :]
        hidden_state = hidden_state.unsqueeze(dim=0)
        k_hidden_state = hidden_state.repeat(beam, 1, 1, 1)

        # print('initial_hidden:', initial_hidden.size())
        # print('k_hidden_state:', k_hidden_state.size())

        k_log_prob_sum = torch.zeros(beam, batch_size, 1)

        if gpu:
            k_log_prob_sum = k_log_prob_sum.cuda()

        for i in range(self.max_len):
            # If target sequence is fed,
            # then use target sequence instead of predicted tokens
            if y is not None:
                if i >= max(y_seq_lens) - 1:
                    break
                else:
                    k_last_pred = y[:, i].unsqueeze(dim=1).unsqueeze(dim=0)
            elif i == 0:
                k_last_pred = START_IDX * torch.LongTensor(np.ones((beam, batch_size, 1), dtype=np.int32))
                if gpu:
                    k_last_pred = k_last_pred.cuda()

            k_log_softmax_full, k_hidden_state_full, k_log_prob_sum_full = forward_one_step(encoder_output, k_last_pred, k_hidden_state, k_log_prob_sum)

            top_k_indexes_k, k_last_pred, k_log_softmax, k_hidden_state, k_log_prob_sum = select_top_k(k_log_softmax_full, k_hidden_state_full, k_log_prob_sum_full)

            output_log_softmax_so_far.append(k_log_softmax)
            preds_so_far.append(k_last_pred)
            parents.append(top_k_indexes_k)

        # Find the best path
        output_log_softmax, preds = [], []
        parent_idx = None
        for i in reversed(range(len(parents))):
            if i == len(parents) - 1:
                output_log_softmax.append(output_log_softmax_so_far[i][0])
                preds.append(preds_so_far[i][0])
                parent_idx = [0] * batch_size
            else:
                log_softmax = torch.transpose(output_log_softmax_so_far[i], 0, 1)
                log_softmax = torch.cat([torch.index_select(h, 0, i) for h, i in zip(log_softmax, parents[i+1][list(range(batch_size)), parent_idx])])
                output_log_softmax.append(log_softmax)

                p = torch.transpose(preds_so_far[i], 0, 1)
                p = torch.cat([torch.index_select(h, 0, i) for h, i in zip(p, parents[i+1][list(range(batch_size)), parent_idx])])
                preds.append(p)

                parent_idx = parents[i+1][list(range(batch_size)), parent_idx]

        # print('output_log_softmax:', len(output_log_softmax))
        # print('output_log_softmax[0]:', output_log_softmax[0].size())
        # print('output_log_softmax[1]:', output_log_softmax[1].size())
        # print('output_log_softmax[2]:', output_log_softmax[2].size())

        output_log_softmax.reverse()
        preds.reverse()

        output_log_softmax = torch.stack(output_log_softmax)
        output_log_softmax = torch.transpose(output_log_softmax, 0, 1)

        preds = torch.stack(preds)
        preds = torch.transpose(preds, 0, 1)
        preds = preds.squeeze(dim=2)

        return output_log_softmax, preds
