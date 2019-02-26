# Sequence Models and Long-Short Term Memory Networks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden \
            = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, 1)

        return tag_scores






if __name__ == "__main__":
    torch.manual_seed(1)

    ###########
    # confirmation
    ###########
    # lstm = nn.LSTM(3, 3)
    # inputs = [torch.randn(1, 3) for _ in range(5)]  # (1, 3) x 5
    # print(inputs)
    #
    # # initialize the hidden state.
    # hidden = (torch.randn(1, 1, 3),
    #           torch.randn(1, 1, 3))
    #
    # for i in inputs:
    # # Step through the sequence one element at a time.
    # # after each step, hidden contains the hidden state.
    #     out, hidden = lstm(i.view(1, 1, -1), hidden)
    #
    # # alternatively, we can do the entire sequence all at once.
    # # the first value returned by LSTM is all of the hidden states throughout
    # # the sequence. the second is just the most recent hidden state
    # # (compare the last slice of "out" with "hidden" below, they are the same)
    # # The reason for this is that:
    # # "out" will give you access to all hidden states in the sequence
    # # "hidden" will allow you to continue the sequence and backpropagate,
    # # by passing it as an argument  to the lstm at a later time
    # # Add the extra 2nd dimension
    # inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    # hidden = (torch.randn(1, 1, 3),
    #           torch.randn(1, 1, 3))
    # out, hidden = lstm(inputs, hidden)
    # print(out)
    # print(hidden)

    training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody, read that book".split(), ["NN", "V", "DET", "NN"])
        ]
    word_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    print(word_to_ix)
    tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    EMBEDDING_DIM = 6
    HIDDEN = 6

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print()
    print("gt")
    print(training_data[0][0])
    print(prepare_sequence(training_data[0][1], tag_to_ix))

    print()
    print("predict before training")
    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        print(tag_scores)
        print(tag_scores.max(1)[1]) # values, indices

    print()
    # train
    num_epochs = 300
    for epoch in range(num_epochs):
        # print("%d / %d" %(epoch, num_epochs))
        for sentence, tags in training_data:
            model.zero_grad()
            model.hidden = model.init_hidden()

            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            tag_scores = model(sentence_in)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    # test
    print("predict after training")
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        print(tag_scores)
        print(tag_scores.max(1)[1]) # values, indices
