# Copyright (C) 2016-2017  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import embeddings

import argparse
import collections
import numpy as np
import re
import sys
import time


# Maximum dimensions for the similarity matrix computation in memory
# A MAX_DIM_X * MAX_DIM_Z dimensional matrix will be used
MAX_DIM_X = 10000
MAX_DIM_Z = 10000


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map the source embeddings into the target embedding space')
    parser.add_argument('src_input', help='the input source embeddings')
    parser.add_argument('trg_input', help='the input target embeddings')
    parser.add_argument('src_output', help='the output source embeddings')
    parser.add_argument('trg_output', help='the output target embeddings')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    mapping_group = parser.add_argument_group('mapping arguments', 'Basic embedding mapping arguments (EMNLP 2016)')
    mapping_group.add_argument('-d', '--dictionary', default=sys.stdin.fileno(), help='the training dictionary file (defaults to stdin)')
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb'], nargs='*', default=[], help='the normalization actions to perform in order')
    mapping_group.add_argument('-c', '--orthogonal', dest='orthogonal', action='store_true', help='use orthogonal constrained mapping (default)')
    mapping_group.add_argument('-u', '--unconstrained', dest='orthogonal', action='store_false', help='use unconstrained mapping')
    parser.set_defaults(orthogonal=True)
    self_learning_group = parser.add_argument_group('self-learning arguments', 'Optional arguments for self-learning (ACL 2017)')
    self_learning_group.add_argument('--self_learning', action='store_true', help='enable self-learning')
    self_learning_group.add_argument('--direction', choices=['forward', 'backward', 'union'], default='forward', help='the direction for dictionary induction (defaults to forward)')
    self_learning_group.add_argument('--numerals', action='store_true', help='use latin numerals (i.e. words matching [0-9]+) as the seed dictionary')
    self_learning_group.add_argument('--threshold', default=0.000001, type=float, help='the convergence threshold (defaults to 0.000001)')
    self_learning_group.add_argument('--validation', default=None, help='a dictionary file for validation at each iteration')
    self_learning_group.add_argument('--log', help='write to a log file in tsv format at each iteration')
    self_learning_group.add_argument('-v', '--verbose', action='store_true', help='write log information to stderr at each iteration')
    args = parser.parse_args()

    # Read input embeddings
    srcfile = open(args.src_input, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_input, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile)
    trg_words, z = embeddings.read(trgfile)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # Build training dictionary
    src_indices = []
    trg_indices = []
    if args.numerals:
        if args.dictionary != sys.stdin.fileno():
            print('WARNING: Using numerals instead of the training dictionary', file=sys.stderr)
        numeral_regex = re.compile('^[0-9]+$')
        src_numerals = {word for word in src_words if numeral_regex.match(word) is not None}
        trg_numerals = {word for word in trg_words if numeral_regex.match(word) is not None}
        numerals = src_numerals.intersection(trg_numerals)
        for word in numerals:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
    else:
        f = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
        for line in f:
            src, trg = line.split()
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                src_indices.append(src_ind)
                trg_indices.append(trg_ind)
            except KeyError:
                print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)

    # Read validation dictionary
    if args.validation is not None:
        f = open(args.validation, encoding=args.encoding, errors='surrogateescape')
        validation = collections.defaultdict(set)
        oov = set()
        vocab = set()
        for line in f:
            src, trg = line.split()
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                validation[src_ind].add(trg_ind)
                vocab.add(src)
            except KeyError:
                pass
                oov.add(src)
        oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
        validation_coverage = len(validation) / (len(validation) + len(oov))

    # Create log file
    if args.log:
        log = open(args.log, mode='w', encoding=args.encoding, errors='surrogateescape')

    # Normalize embeddings
    for action in args.normalize:
        if action == 'unit':
            x = embeddings.length_normalize(x)
            z = embeddings.length_normalize(z)
        elif action == 'center':
            x = embeddings.mean_center(x)
            z = embeddings.mean_center(z)
        elif action == 'unitdim':
            x = embeddings.length_normalize_dimensionwise(x)
            z = embeddings.length_normalize_dimensionwise(z)
        elif action == 'centeremb':
            x = embeddings.mean_center_embeddingwise(x)
            z = embeddings.mean_center_embeddingwise(z)

    # Training loop
    prev_objective = objective = -100.
    it = 1
    t = time.time()
    while it == 1 or objective - prev_objective >= args.threshold:

        # Update the embedding mapping
        if args.orthogonal:  # orthogonal mapping
            u, s, vt = np.linalg.svd(np.dot(z[trg_indices].T, x[src_indices]))
            w = np.dot(vt.T, u.T)
        else:  # unconstrained mapping
            x_pseudoinv = np.dot(np.linalg.inv(np.dot(x[src_indices].T, x[src_indices])), x[src_indices].T)
            w = np.dot(x_pseudoinv, z[trg_indices])
        xw = x.dot(w)

        # Self-learning
        if args.self_learning:

            # Update the training dictionary
            best_sim_forward = np.full(x.shape[0], -100.)
            src_indices_forward = range(x.shape[0])
            trg_indices_forward = np.zeros(x.shape[0], dtype=int)
            best_sim_backward = np.full(z.shape[0], -100.)
            src_indices_backward = np.zeros(z.shape[0], dtype=int)
            trg_indices_backward = range(z.shape[0])
            for i in range(0, x.shape[0], MAX_DIM_X):
                for j in range(0, z.shape[0], MAX_DIM_Z):
                    sim = xw[i:i+MAX_DIM_X].dot(z[j:j+MAX_DIM_Z].T)
                    for k in range(sim.shape[0]):
                        l = sim[k].argmax()
                        if sim[k, l] > best_sim_forward[i+k]:
                            best_sim_forward[i+k] = sim[k, l]
                            trg_indices_forward[i+k] = j + l
                    if args.direction in ('backward', 'union'):  # Slow, only do if necessary
                        for l in range(sim.shape[1]):
                            k = sim[:, l].argmax()
                            if sim[k, l] > best_sim_backward[j+l]:
                                best_sim_backward[j+l] = sim[k, l]
                                src_indices_backward[j+l] = i + k
                    sim = None
            if args.direction == 'forward':
                src_indices = src_indices_forward
                trg_indices = trg_indices_forward
            elif args.direction == 'backward':
                src_indices = src_indices_backward
                trg_indices = trg_indices_backward
            elif args.direction == 'union':
                src_indices = np.concatenate((src_indices_forward, src_indices_backward))
                trg_indices = np.concatenate((trg_indices_forward, trg_indices_backward))

            # Objective function evaluation
            prev_objective = objective
            if args.direction == 'forward':
                objective = np.mean(best_sim_forward)
            elif args.direction == 'backward':
                objective = np.mean(best_sim_backward)
            elif args.direction == 'union':
                objective = (np.mean(best_sim_forward) + np.mean(best_sim_backward)) / 2

            # Accuracy and similarity evaluation in validation
            if args.validation is not None:
                accuracy = np.mean([1 if trg_indices_forward[src] in trg else 0 for src, trg in validation.items()])
                similarity = np.mean([np.max(z[list(trg)].dot(xw[src])) for src, trg in validation.items()])

            # Logging
            duration = time.time() - t
            if args.verbose:
                print(file=sys.stderr)
                print('ITERATION {0} ({1:.2f}s)'.format(it, duration), file=sys.stderr)
                print('\t- Objective:        {0:9.4f}%'.format(100 * objective), file=sys.stderr)
                if args.validation is not None:
                    print('\t- Val. similarity:  {0:9.4f}%'.format(100 * similarity), file=sys.stderr)
                    print('\t- Val. accuracy:    {0:9.4f}%'.format(100 * accuracy), file=sys.stderr)
                    print('\t- Val. coverage:    {0:9.4f}%'.format(100 * validation_coverage), file=sys.stderr)
                sys.stderr.flush()
            if args.log is not None:
                val = '{0:.6f}\t{1:.6f}\t{2:.6f}'.format(
                    100 * similarity, 100 * accuracy, 100 * validation_coverage) if args.validation is not None else ''
                print('{0}\t{1:.6f}\t{2}\t{3:.6f}'.format(it, 100 * objective, val, duration), file=log)
                log.flush()

        t = time.time()
        it += 1

    # Write mapped embeddings
    srcfile = open(args.src_output, mode='w', encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_output, mode='w', encoding=args.encoding, errors='surrogateescape')
    embeddings.write(src_words, xw, srcfile)
    embeddings.write(trg_words, z, trgfile)
    srcfile.close()
    trgfile.close()


if __name__ == '__main__':
    main()
