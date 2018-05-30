# -*- coding: utf-8 -*-
"""
@author: Harsh
Computing the Vector Extrema and Embedding Average Score for the ground truth and the predicted response.

Example run:
    python Extrema-Average.py path_to_ground_truth.txt path_to_predictions.txt path_to_embeddings.bin

"""

import gensim
import numpy as np
import sys
import os

def get_data(ref, cand):
    #Reading the reference and the candidate file
    reference = []
    if '.txt' in ref:
        reference = [line.rstrip('\n') for line in open(ref, 'r', encoding='utf-8')]
    
    else:
        for root, dirs, files in os.walk(ref):
            for f in files:
                reference = [line.rstrip('\n') for line in open(ref, 'r', encoding='utf-8')]
                
    with open(cand,'r', encoding='utf-8') as h:
        candidate = h.readlines()
        
    
    h.close()
        
    return reference,candidate

def sentence_to_words(reference,candidate):
    #Splitting the sentence into words
    new_reference = []
    new_candidate = []
    new_reference = reference[0].split()
    new_candidate = candidate[0].split()
    
    
    return new_reference,new_candidate


def vector_extrema(ref, cand, w2v):
    #Calculating vector extrema score
    scores = []

    for i in range(len(ref)):
        tokens1, tokens2 = sentence_to_words(ref,cand)
        
        X= []
        for tok in tokens1:
            if tok in w2v:
                X.append(w2v[tok])
        Y = []
        for tok in tokens2:
            if tok in w2v:
                Y.append(w2v[tok])

        # if none of the words have embeddings in ground truth, skip
        if np.linalg.norm(X) < 0.00000000001:
            continue

        # if none of the words have embeddings in response, count result as zero
        if np.linalg.norm(Y) < 0.00000000001:
            scores.append(0)
            continue

        xmax = np.max(X, 0)  
        xmin = np.min(X,0)  
        xtrema = []
        for i in range(len(xmax)):
            if np.abs(xmin[i]) > xmax[i]:
                xtrema.append(xmin[i])
            else:
                xtrema.append(xmax[i])
        X = np.array(xtrema)   

        ymax = np.max(Y, 0)
        ymin = np.min(Y,0)
        ytrema = []
        for i in range(len(ymax)):
            if np.abs(ymin[i]) > ymax[i]:
                ytrema.append(ymin[i])
            else:
                ytrema.append(ymax[i])
        Y = np.array(ytrema)

        o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)
        
        scores.append(o)
        
    scores = np.asarray(scores)
    
    return np.mean(scores)

def embedding_average(ref, cand, w2v):
    #Calculating embedding average score
    dim = w2v.vector_size # dimension of embeddings

    scores = []

    for i in range(len(ref)):
        tokens1,tokens2 = sentence_to_words(ref,cand)
        X= np.zeros((dim,))
        for tok in tokens1:
            if tok in w2v:
                X+=w2v[tok]
        Y = np.zeros((dim,))
        for tok in tokens2:
            if tok in w2v:
                Y += w2v[tok]

        # if none of the words in ground truth have embeddings, skip
        if np.linalg.norm(X) < 0.00000000001:
            continue

        # if none of the words have embeddings in response, count result as zero
        if np.linalg.norm(Y) < 0.00000000001:
            scores.append(0)
            continue

        X = np.array(X)/np.linalg.norm(X)
        Y = np.array(Y)/np.linalg.norm(Y)
        o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)

        scores.append(o)
    
    scores = np.asarray(scores)
    
    return np.mean(scores)


if __name__ == "__main__":
    reference, candidate = get_data(sys.argv[1], sys.argv[2])
    
    emb = sys.argv[3]
    embedding = gensim.models.KeyedVectors.load_word2vec_format(emb, binary=True)
    
    extrema_score = vector_extrema(reference, candidate,embedding)
    print("Vector Extrema Score is:%0.4f" % extrema_score)
    
    average_score = embedding_average(reference, candidate,embedding)
    print("Embedding Average Score is:%0.4f" % average_score)