# -*- coding: utf-8 -*-
"""
@author: Harsh

Computing the Greedy Matching Score for the ground truth and the predicted response.

Example run:
    python GreedyMatching.py path_to_ground_truth.txt path_to_predictions.txt path_to_embeddings.bin
"""

import os
import sys
import numpy as np
import gensim
from sklearn.metrics.pairwise import cosine_similarity


class Embedding(object):
    def __init__(self):
        emb = sys.argv[3]
        self.embedding = gensim.models.KeyedVectors.load_word2vec_format(emb, binary=True)
        self.unk = self.embedding.vectors.mean(axis=0)

    def vec(self, key):
        try:
            return self.embedding.vectors[self.embedding.vocab[key].index]
        except KeyError:
            return self.unk


def get_data(ref, cand):
   #Reading the reference and the candidate file
    reference = []
    if '.txt' in ref:
        reference = [line.rstrip('\n') for line in open(ref, 'r', encoding='utf-8')]

    else:
        for root, dirs, files in os.walk(ref):
            for f in files:
                reference = [line.rstrip('\n') for line in open(ref, 'r', encoding='utf-8')]

    with open(cand, 'r', encoding='utf-8') as h:
        candidate = h.readlines()

    h.close()

    return reference, candidate


def sentence_tokenize(reference):
    #Splitting the sentence into words
    new_reference = []
    new_reference = reference[0].split()

    return new_reference


def greedy_matching(hypothesis, reference, emb=None):
#Calculating the greedy matching score
    if emb is None:
        emb = Embedding()
    emb_hyps = []
    embs = [emb.vec(word) for word in sentence_tokenize(hypothesis)]
    emb_hyps.append(embs)

    emb_refs = []
    references = [reference]
    for ref in references:
        emb_refsource = []
        embs = [emb.vec(word) for word in sentence_tokenize(ref)]
        emb_refsource.append(embs)
        emb_refs.append(emb_refsource)

    scores = []
    for emb_ref in emb_refs:
        score = []
        for ref, hyp in zip(emb_ref, emb_hyps):
            similarity_matrix = cosine_similarity(ref, hyp)
            score1 = similarity_matrix.max(axis=0).mean()
            score2 = similarity_matrix.max(axis=1).mean()
            score.append((score1 + score2) / 2)
        scores.append(score)
    scores = np.max(scores, axis=0).mean()

    return scores


if __name__ == '__main__':
    reference, candidate = get_data(sys.argv[1], sys.argv[2])

    score = greedy_matching(candidate, reference)
    print("Greedy Matching Score is: %0.4f" % score)
