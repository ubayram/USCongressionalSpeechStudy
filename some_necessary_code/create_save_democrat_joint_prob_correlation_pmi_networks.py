# author: Ulya Bayram
# ulyabayram@gmail.com

from glob import glob
import numpy as np
import math

# This code saves two numpy matrices, square matrices of size |Vocabulary|
# indexing order is the same order in vocabulary lists in merged sets
# For each vocabulary file, counts the number of times words wi and wj both exists in the same interview
# Saves 2 numpy matrices per group: 
#  and the normalized version

def getVocab(filename):

    fo = open(filename, 'r')
    vocab_ = []
    for line in fo:
        word_ = line.split()[0]
        vocab_.append(word_)
    return vocab_

def splitByClassLabel(list_of_files):

    d_list = []
    r_list = []

    del list_of_files[list_of_files.index('')]

    for filename in list_of_files:

        if 'd' in filename[0]:
            d_list.append(filename)
        elif 'r' in filename[0]:
            r_list.append(filename)
        else:
            print('Skipping last - empty - line!')

    return d_list, r_list
##########################################################################################

list_of_stuff = ['97', '98','99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114']

# there's only one training-test split
for i in list_of_stuff:

    f_colnames = open('features'+i+'/unigram_matrix_columnames.txt')
    d_vocab_ = f_colnames.read().split('\n')[:-1]
    f_colnames.close()
    print(d_vocab_)

    train_location = 'House'+i+'_sentences/'
    #d_vocab_ = getVocab('features/democrats_training_vocabulary_all.txt')
    # remove the stopwords from the vocabulary - networks don't utilize the stopwords
    #d_vocab_ = removeStopWords(d_vocab_)

    f_files = open('train'+i+'.txt')
    list_files = f_files.read().split('\n')
    d_files, r_files = splitByClassLabel(list_files)

    # -------------------------- extract and save the democrat class features ----------------------
    d_freq_matrix = np.zeros((len(d_vocab_), len(d_vocab_))) # words versus words matrix
    num_sentences = 0

    # also we'll need the word level marginal probabilities
    word_counts_d = [0.0]*len(d_vocab_)

    for file_ in d_files: # loop over the training files

        fo = open(train_location + file_, 'r')
        sentence_list = fo.read().split(' * ')

        for sentence in sentence_list:

            sent_words_list = sentence.split(' ')
            sent_words_list = list(set(sent_words_list)) # unique words in the same sentence
            for s_index1 in xrange(len(sent_words_list)-1):
                word_1 = sent_words_list[s_index1]

                # increase the marginal probabilities of the words in the sentence - last word is missing due to the triangle loop
                if word_1 in d_vocab_:
                    index_1 = d_vocab_.index(word_1)
                    word_counts_d[index_1] += 1.0
                # get the second word, repeats aren't okay so we reduce the computational time
                for s_index2 in  xrange(s_index1+1, len(sent_words_list)):
                    word_2 = sent_words_list[s_index2]

                    #  check if both words are among the frequent ones, in the vocabulary, if yes then update the matrix
                    if word_1 in d_vocab_ and word_2 in d_vocab_:
                        # get their indices in the vocab so we'll have the same vocab order in the probability matrix
                        index_1 = d_vocab_.index(word_1)
                        index_2 = d_vocab_.index(word_2)

                        # add 1 to co-occurrence of these two words in the same sentence in both orders
                        d_freq_matrix[index_1, index_2] += 1
                        d_freq_matrix[index_2, index_1] += 1
            # now, add the probability of the last word ignored by the first loop due to its ending
            if len(sent_words_list) > 1: # only if there are more than one word in the sentence
                num_sentences += 1
                if word_2 in d_vocab_:
                    index_2 = d_vocab_.index(word_2)
                    word_counts_d[index_2] += 1.0

    # normalize the matrix, make it a probability matrix
    normalized_d_freq_matrix = d_freq_matrix / float(num_sentences)

    # convert freqs to marginal probs at the sentence level - prob. of observing the word x in a sentence in d training
    word_probs_d = [0.0]*len(d_vocab_)
    for ii in xrange(len(d_vocab_)):
        word_probs_d[ii] = word_counts_d[ii] / float(num_sentences)

    # now save the joint probability association network matrix
    np.savetxt('features'+i+'/d_training_joint_prob_word_sentence_network_matrix_house.txt', normalized_d_freq_matrix)

    del word_counts_d, d_freq_matrix

    d_correlation_matrix = np.zeros((len(d_vocab_), len(d_vocab_)))
    d_pmi_matrix = np.zeros((len(d_vocab_), len(d_vocab_)))
    # need another pass over the vocabulary words ti fill out this second matrix-network
    for index1 in xrange(len(d_vocab_)-1): #i
        word1 = d_vocab_[index1]
        p_i = word_probs_d[index1]

        for index2 in xrange(index1+1, len(d_vocab_)): #j
            word2 = d_vocab_[index2]
            p_j = word_probs_d[index2]
            # compute the correlation
            p_ij = normalized_d_freq_matrix[index1, index2] # take the joint probability, remember, this matrix was symmetrical

            if p_ij > p_i:
                print('Error, pi is lower than joint')
            if p_ij > p_j:
                print('Error, pj is lower than joint')

            # take care of the correlation coefficient computations and fill up the matrix
            denom_ = p_i * (1-p_i) * p_j * (1-p_j)
            corr_coeff_ij = (p_ij - (p_i*p_j))/math.sqrt(denom_)

            if corr_coeff_ij < -1 or corr_coeff_ij > 1:
                print(corr_coeff_ij)

            if corr_coeff_ij <= 0:
                d_correlation_matrix[index1, index2] = 0.0
                d_correlation_matrix[index2, index1] = 0.0
            else:
                d_correlation_matrix[index1, index2] = corr_coeff_ij
                d_correlation_matrix[index2, index1] = corr_coeff_ij

            # now do the same for the pointwise mutual information computation
            denom_pmi = p_i * p_j
            if denom_ > 0 and p_ij > 0:
                pmi_ij = np.log(p_ij/denom_pmi)/-np.log(p_ij)
            else:
                pmi_ij = 0
            
            if pmi_ij < -1 or pmi_ij > 1:
                print(pmi_ij)

            if pmi_ij <= 0:
                d_pmi_matrix[index1, index2] = 0.0
                d_pmi_matrix[index2, index1] = 0.0
            else:
                d_pmi_matrix[index1, index2] = pmi_ij
                d_pmi_matrix[index2, index1] = pmi_ij

    print('Democrat final matrix size ' + str(np.shape(d_correlation_matrix)))
    print('Democrat final matrix size ' + str(np.shape(d_pmi_matrix)))
    np.savetxt('features'+i+'/d_training_correlation_coeff_word_sentence_network_matrix_house.txt', d_correlation_matrix)
    np.savetxt('features'+i+'/d_training_pmi_word_sentence_network_matrix_house.txt', d_pmi_matrix)
