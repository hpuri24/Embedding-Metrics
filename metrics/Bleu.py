# -*- coding: utf-8 -*-
"""

@author: Harsh

Computing the Bleu Score for the ground truth and the predicted response.

Example run:
    python Bleu.py path_to_ground_truth.txt path_to_predictions.txt path_to_embeddings.bin
"""
import sys
import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

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
    

def get_bleu_score(reference, candidate, weights):
    #Calculating Bleu Score
    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference,candidate,weights,smoothing_function=smoothie)
    return score

def sentence_to_words(references,candidate):
    #Splitting the sentence into words
    new_references = []
    new_candidate = []
    for i in range(len(references)):
        reference = references[i].split()
        new_references.append(reference)
    new_candidate = candidate[0].split()
    
    
    return new_references,new_candidate
    
if __name__ == "__main__":
    references,candidate = get_data(sys.argv[1], sys.argv[2])
    new_ref,new_cand = sentence_to_words(references,candidate)
    weights = (1,0,0,0)
    if len(new_cand) == 1:
        bleu = get_bleu_score(new_ref, new_cand,weights)
    else:
        bleu = get_bleu_score(new_ref, new_cand,weights)
    
    print("The bleu score is :", bleu)
    