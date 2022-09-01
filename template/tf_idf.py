 # tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020
# Modified by Kiran Ramnath 02/13/2021
# Modified by Mohit Goyal (mohit@illinois.edu) on 01/16/2022
"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter, defaultdict
import time
import operator

import reader

def compute_tf_idf(train_set, train_labels, dev_set):
       
        num_train_sets = len(train_labels)
        train_document_words = {}       # keys: word, values: number of document it occurs in
        
        for document in train_set:
                words_in_document = {} # keys: word, values: num of times in document
                for wrd in document:
                        words_in_document[wrd] = document.count(wrd)
                for word_unique in words_in_document.keys():
                        val = train_document_words.get(word_unique) or 0
                        train_document_words[word_unique] = val + 1
       

        all_tfidfs = []
        for docA in dev_set:   
                num_words = len(docA)
                tf_idfs = []
                for word in docA:
                        prob = ( docA.count(word) / num_words ) * np.log(  num_train_sets / (1 + (train_document_words.get(word) or 0)) )
                        tf_idfs.append(prob)
                max_idf_word = docA[np.argmax(tf_idfs)]
                all_tfidfs.append(max_idf_word)
           
    # return list of words (should return a list, not numpy array or similar)
        print(all_tfidfs)
        return all_tfidfs
