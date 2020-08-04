#!/usr/bin/python
#-*- coding: utf-8 -*-

import argparse
import multiprocessing as mp
from time import time
import re
import sys

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


parser = argparse.ArgumentParser()
parser.add_argument('--corpus', help='the path to the corpus file',
                    type=str, metavar='<file path>', default=None)
parser.add_argument('--model', help='the path to the file of the model to '
                                    'continue training; if specified, all '
                                    'next args until ``epochs`` has no effect',
                    type=str, metavar='<file path>', default=None)
parser.add_argument('--sg', help='use Skip-gram instead of CBOW; default off',
                    action='store_const', const=True, default=False)
parser.add_argument('--cbow_sum', help='use the sum of context vectors instead '
                                       'of mean (only applies for CBOW); '
                                       'default off',
                    action='store_const', const=True, default=False)
parser.add_argument('--vector_len', help='the length of result vectors; '
                                         'default 300',
                    type=int, metavar='<int>', default=300)
parser.add_argument('--window', help='the max distance from the center word '
                                     'upto the farest one; default 7',
                    type=int, metavar='<int>', default=7)
parser.add_argument('--vocab_min_count', help='itnore all words with total '
                                              'frequency lower than <int>; '
                                              'default 5',
                    type=int, metavar='<int>', default=5)
parser.add_argument('--seed', help='default None',
                    type=int, metavar='<int>', default=None)
parser.add_argument('--epochs', help='number iterations during the current '
                                     'train; default 20',
                    type=int, metavar='<int>', default=20)
parser.add_argument('--max_epochs', help='number of iterations during all '
                                         'trains; default None (no limits)',
                    type=int, metavar='<int>', default=None)
parser.add_argument('--checkpoint_every', help='save model every <int> '
                                               'iterations; default None (off)',
                    type=int, metavar='<int>', default=None)
parser.add_argument('--workers', help='number of working threads; '
                                      '0 means cpu_count; '
                                      '-<int> means cpu_count - <int>; '
                                      '+<int> means cpu_count + <int>; '
                                      'default 1',
                    type=str, metavar='<int>', default='1')
parser.add_argument('--save_model', help='save model that can be load later '
                                         'with ``model`` to continue training',
                    action='store_const', const=True, default=False)
args = parser.parse_args()
workers = 1
try:
    workers = int(args.workers)
except ValueError:
    parser.print_usage(file=sys.stderr)
    print("{}: error: argument --workers: invalid int value: '{}'"
              .format(sys.argv[0], args.workers), file=sys.stderr)
    exit(1)
if workers == 0 or args.workers[0] in ['-', '+']:
    workers = max(mp.cpu_count() + workers, 1)

def model_save (with_epoch=True):
    fn = 'w2v_{}{}_{}'.format('sg' if args.sg else 'cbow',
                               '_sum' if args.cbow_sum else '_mean',
                               args.vector_len)
    if with_epoch:
        fn += '.{:03}'.format(model.callbacks[0].epoch)
    if args.save_model:
        model.save(fn + '.bin')
    model.wv.save_word2vec_format(fn + '.txt', binary=False)


class MaxEpochReachedException(Exception):
    pass


class Callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch'''

    def __init__ (self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end (self, model):
        self.epoch += 1
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        if args.checkpoint_every and self.epoch % args.checkpoint_every == 0:
            print('    checkpoint for epoch {}...'.format(self.epoch), end='')
            sys.stdout.flush()
            model_save()
            print(' done.')
        if self.epoch >= args.max_epochs:
            print('Max epoch has been reached. Process stopped')
            raise MaxEpochReachedException()


model = Word2Vec.load(args.model) if args.model else \
        Word2Vec(min_count=10,
                 window=7,
                 size=args.vector_len,
                 sample=1e-5,
                 alpha=0.03,
                 min_alpha=0.0007,
                 negative=20,
                 workers=workers,
                 sg=args.sg,
                 cbow_mean=not args.cbow_sum,
                 compute_loss=True,
                 callbacks=[Callback()],
                 seed=args.seed)

if args.model:
    print('Continue training model. Current epoch:', model.callbacks[0].epoch)
else:
    print('Building vocab started...')
    t = time()
    model.build_vocab(corpus_file=args.corpus, progress_per=10000)
    print('sents: {}, tokens: {}, words: {}'
              .format(model.corpus_count, model.corpus_total_words,
                      len(model.wv.vocab)))
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

t = time()
try:
    model.train(corpus_file=args.corpus,
                total_examples=model.corpus_count,
                total_words=model.corpus_total_words,
                epochs=args.epochs,
                report_delay=1.)
except MaxEpochReachedException:
    pass

model_save(False)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
