# -*- coding: utf-8 -*-
# Word Embeddings: Encoding Lexical Semantics
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)

        return log_probs


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2 * context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)

        return log_probs


if __name__ == "__main__":
    torch.manual_seed(1)

    # demonstration
    word_to_ix = {"hello": 0, "world": 1}
    embeds = nn.Embedding(2, 5) # 2 words in vocab, 5 dimensional embeddings
    lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
    hello_embed = embeds(lookup_tensor)
    print(hello_embed)

    CONTEXT_SIZE = 2
    EMBEDDING_DIM = 100

    # We will use Shakespeare Sonnet 2
    test_sentence = """When forty winters shall besiege thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery so gazed on now,
    Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.""".split()

    # we should tokenize the input, but we will ignore that for now
    # build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
    trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2]) for i in range(len(test_sentence) - 2)]

    # print the first 3, just so you can see what they look like
    print(trigrams[:3])

    vocab = set(test_sentence)
    word_to_ix = {word: i for (i, word) in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}
    losses = []
    criterion = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    num_epochs = 500
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        for context, target in trigrams:
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words into integer indices and wrap them in tensors)
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

            model.zero_grad()
            log_probs = model(context_idxs)
            loss = criterion(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("epoch %d / %d : %f" %(epoch, num_epochs, total_loss / len(trigrams)))
        losses.append(total_loss / len(trigrams))

    # print(losses)
    plt.figure()
    plt.plot(losses)
    plt.title("losses")
    plt.savefig("loss_word_embed.png")
    plt.close()

    ####################
    # test NGramLanguageModeler
    ####################
    with torch.no_grad():
        context = ["own", "deep"]
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        log_probs = model(context_idxs)#.data.numpy()
        topv, topi = log_probs.topk(1)
        topi = topi[0][0]
        print('Raw text: {}\n'.format(' '.join(test_sentence)))
        print('Context: {}\n'.format(context))
        print('Prediction: {}'.format(ix_to_word[topi.item()]))



    print()
    ##################################################################
    # EXERCISE: COMPUTING WORD EMBEDDINGS: CONTINUOUS BAG-OF-WORDS
    ##################################################################
    CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
    EMBEDDING_DIM = 1000
    raw_text = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".split()

    # By deriving a set from `raw_text`, we deduplicate the array
    vocab = set(raw_text)
    vocab_size = len(vocab)

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}
    data = []
    for i in range(2, len(raw_text) - 2):
        context = [raw_text[i - 2], raw_text[i - 1],
                   raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))
    print(data[:5])


    losses = []
    criterion = nn.NLLLoss()
    model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)


    num_epochs = 500
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        for context, target in data:
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words into integer indices and wrap them in tensors)
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

            model.zero_grad()
            log_probs = model(context_idxs)
            loss = criterion(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("epoch %d / %d : %f" %(epoch, num_epochs, total_loss / len(data)))
        losses.append(total_loss / len(data))

    # print(losses)
    plt.figure()
    plt.plot(losses)
    plt.title("losses")
    plt.savefig("loss_CBOW.png")
    plt.close()

    ####################
    # test CBOW
    ####################
    with torch.no_grad():
        context = ["People", "create", "to", "direct"]
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        log_probs = model(context_idxs)#.data.numpy()
        topv, topi = log_probs.topk(1)
        topi = topi[0][0]
        print('Raw text: {}\n'.format(' '.join(raw_text)))
        print('Context: {}\n'.format(context))
        print('Prediction: {}'.format(ix_to_word[topi.item()]))
