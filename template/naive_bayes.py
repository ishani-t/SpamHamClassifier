# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader


"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""




"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    # print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}
    ##TODO: 

    for i, email in enumerate(X):
        pos_or_neg = y[i]
        for word in email: 
            if pos_or_neg == 1: # if this email is pos
                val = pos_vocab.get(word) or 0
                pos_vocab[word] = val + 1
            if pos_or_neg == 0: # if this email is neg
                val = neg_vocab.get(word) or 0
                neg_vocab[word] = val + 1
                
    # raise RuntimeError("Replace this line with your code!")
    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word pair appears 
    """
    #print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}
    ##TODO:

    for i, email in enumerate(X):
        pos_or_neg = y[i]
        for j in range(len(email)-1):
            word = email[j] + " " + email[j+1]
            if pos_or_neg == 1: # if this email is pos
                val = pos_vocab.get(word) or 0
                pos_vocab[word] = val + 1
            if pos_or_neg == 0: # if this email is neg
                val = neg_vocab.get(word) or 0
                neg_vocab[word] = val + 1
    
    # merging in the uni dictionaries
    pos_vocab_uni, neg_vocab_uni = create_word_maps_uni(X, y, None)
    pos_vocab = {**pos_vocab_uni, **pos_vocab}
    neg_vocab = {**neg_vocab_uni, **neg_vocab}

    # raise RuntimeError("Replace this line with your code!")
    return dict(pos_vocab), dict(neg_vocab)



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)

    #raise RuntimeError("Replace this line with your code!")
    dev_labels = []

    pos_vocab, neg_vocab = create_word_maps_uni(train_set, train_labels)

    total_spam_words = sum(pos_vocab.values())
    distinct_spam_words = len(pos_vocab.keys())
    spam_denominator = total_spam_words + laplace*(1 + distinct_spam_words)

    total_ham_words = sum(neg_vocab.values())
    distinct_ham_words = len(neg_vocab.keys())
    ham_denominator = total_ham_words + laplace*(1 + distinct_ham_words)

    for email in dev_set:
        prob_spam = np.log(pos_prior)
        prob_ham = np.log(1 - pos_prior)

        for word in email:
            prob_spam += np.log(((pos_vocab.get(word) or 0) + laplace) / spam_denominator)
            prob_ham += np.log(((neg_vocab.get(word) or 0) + laplace) / ham_denominator)


        if(prob_spam > prob_ham):
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    max_vocab_size = None

    # raise RuntimeError("Replace this line with your code!")

    dev_labels = []

    # UNIGRAM CALCULATIONS
    pos_vocab_uni, neg_vocab_uni = create_word_maps_uni(train_set, train_labels)

    total_spam_words_uni = sum(pos_vocab_uni.values())
    distinct_spam_words_uni = len(pos_vocab_uni.keys())
    spam_denominator_uni = total_spam_words_uni + unigram_laplace*(1 + distinct_spam_words_uni)

    total_ham_words_uni = sum(neg_vocab_uni.values())
    distinct_ham_words_uni = len(neg_vocab_uni.keys())
    ham_denominator_uni = total_ham_words_uni + unigram_laplace*(1 + distinct_ham_words_uni)

    ### BIGRAM CALCULATIONS
    pos_vocab_bi, neg_vocab_bi = create_word_maps_bi(train_set, train_labels)

    total_spam_words_bi = sum(pos_vocab_bi.values())
    distinct_spam_words_bi = len(pos_vocab_bi.keys())
    spam_denominator_bi = total_spam_words_bi + bigram_laplace*(1 + distinct_spam_words_bi)

    total_ham_words_bi = sum(neg_vocab_bi.values())
    distinct_ham_words_bi = len(neg_vocab_bi.keys())
    ham_denominator_bi = total_ham_words_bi + bigram_laplace*(1 + distinct_ham_words_bi)


    for email in dev_set:
        uni_prob_spam = np.log(pos_prior)
        uni_prob_ham = np.log(1 - pos_prior)

        bi_prob_spam = np.log(pos_prior)
        bi_prob_ham = np.log(1 - pos_prior)

        for i, word in enumerate(email):
            uni_prob_spam += np.log(((pos_vocab_uni.get(word) or 0) + unigram_laplace) / spam_denominator_uni)
            uni_prob_ham += np.log(((neg_vocab_uni.get(word) or 0) + unigram_laplace) / ham_denominator_uni)

            if(i == (len(email) - 1)):
                break
            
            bi_prob_spam += np.log(((pos_vocab_bi.get(word + " " + email[i+1]) or 0) + bigram_laplace) / spam_denominator_bi)
            bi_prob_ham += np.log(((neg_vocab_bi.get(word + " " + email[i+1]) or 0) + bigram_laplace) / ham_denominator_bi)

        prob_spam = (1 - bigram_lambda) * uni_prob_spam + (bigram_lambda * bi_prob_spam)
        prob_ham = (1 - bigram_lambda) * uni_prob_ham + (bigram_lambda * bi_prob_ham)


        if(prob_spam > prob_ham):
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels
