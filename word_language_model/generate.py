###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

import data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pytorch Wikitext-2 Language Model")
    # model parameters
    parser.add_argument("--data", type=str, default="./data/wikitext-2",
                        help="location of the data corpus [./data/wikitext-2] .")
    parser.add_argument("--checkpoint", type=str, default="./model.pt",
                        help="model checkpoint to use [./model.pt] .")
    parser.add_argument("--outf", type=str, default="generated.txt",
                        help="output file for generated text [generated.txt] .")
    parser.add_argument("--words", type=int, default=1000,
                        help="number of words to generate [1000] .")
    parser.add_argument("--seed", type=int, default=1111,
                        help="random seed [1111] .")
    parser.add_argument("--cuda", action="store_true",
                        help="use CUDA [False] .")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature - higher will increase diversity [1.0] .")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="reporting interval [100] .")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3")

    with open(args.checkpoint, "rb") as f:
        model = torch.load(f).to(device)
    model.eval()

    corpus = data.Corpus(args.data)
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    with open(args.outf, "w") as outf:
        with torch.no_grad():   # no tracking history
            for i in range(args.words):
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)
                word = corpus.dictionary.idx2word[word_idx]

                outf.write(word + ("\n" if i % 20 == 19 else " "))

                if i % args.log_interval == 0:
                    print('| Generated {}/{} words'.format(i, args.words))
