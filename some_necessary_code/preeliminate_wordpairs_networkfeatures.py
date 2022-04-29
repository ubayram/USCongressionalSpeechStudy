# author: Ulya Bayram
# ulyabayram@gmail.com
# Code to pre-eliminate/or elect acceptable word pairs from the networks
# This will reduce the space consumption of plenty of edges
#
import numpy as np
import time
from itertools import permutations

def eliminateNonVocabWords(test_words, vocab):
    return list(set(test_words).intersection(vocab))

def mergeVocabsAndMatrices(weight_type, i_num):

    # read the common vocabulary corresponding to the cols and rows in d and r matrices
    f_colnames = open('features'+i_num+'/unigram_matrix_columnames.txt', 'r')
    vocab = f_colnames.read().split('\n')[:-1]
    f_colnames.close()

    #d_training_joint_prob_word_sentence_network_matrix_house
    s_matrix_merged = np.loadtxt('features'+i_num+'/d_training_' + weight_type +'_word_sentence_network_matrix_house.txt')
    c_matrix_merged = np.loadtxt('features'+i_num+'/r_training_' + weight_type +'_word_sentence_network_matrix_house.txt')

    # read the nltk stopwords - stopwordleri silmesek ne olacak? onu da dene
    fstop = open('../StopwordFiles/nltk_stopwords.txt', 'r')
    stop_list = fstop.read().split('\n')[:-1] # skip the last line (it's empty)
    fstop.close()

    # now, remove those columns and rows that correspond to the stopwords in the unigram matrix, and update the vocab
    indices_to_remove = []
    merged_vocab = []
    for word_ in vocab:
        if word_ in stop_list:
            indices_to_remove.append(vocab.index(word_))
        else:
            merged_vocab.append(word_) # the correct order is preserved

    print('Num of stopwords to be removed ' + str(len(indices_to_remove)) )
    print('New num vocab words ' + str(len(merged_vocab)))
    
    # delete the corresponding columns and rows : get rid of the stopwords
    s_matrix_merged = np.delete(s_matrix_merged, indices_to_remove, axis=1)
    c_matrix_merged = np.delete(c_matrix_merged, indices_to_remove, axis=1)
    s_matrix_merged = np.delete(s_matrix_merged, indices_to_remove, axis=0)
    c_matrix_merged = np.delete(c_matrix_merged, indices_to_remove, axis=0)

    print('New matrix sizes ' + str(np.shape(s_matrix_merged)) + '\t' + str(np.shape(c_matrix_merged)))

    # threshold the edges with small - ignorable - weights (features) to speed up the processing time and reduce feature space
    # remove weak connections - by rounding their weights to zero
    s_matrix_merged = np.where(np.abs(s_matrix_merged)<0.01, 0, s_matrix_merged)
    c_matrix_merged = np.where(np.abs(c_matrix_merged)<0.01, 0, c_matrix_merged)

    difference_matrix = np.abs(s_matrix_merged - c_matrix_merged)

    s_matrix_merged = np.where(difference_matrix<0.01, 0, s_matrix_merged)
    c_matrix_merged = np.where(difference_matrix<0.01, 0, c_matrix_merged)

    new_difference_matrix = s_matrix_merged + c_matrix_merged

    # now, compute the acceptable word pairs list (lansa feature set)
    wordpairs_list = []
    for index_1 in xrange(len(merged_vocab)-1):
        for index_2 in xrange(index_1+1, len(merged_vocab)):
            if new_difference_matrix[index_1, index_2] > 0:
                wordpairs_list.append((merged_vocab[index_1], merged_vocab[index_2]))

    print('Num of wordpairs (lansa features) collected ' + str(len(wordpairs_list)))

    return s_matrix_merged, c_matrix_merged, merged_vocab, wordpairs_list

# ----------------------------- main -------------------------------

thresh = 50 # the min number of documents the word pair should exist - not necessarily concurrently (we are finding number of edges)
weight_type = 'correlation_coeff'

for i_split in ['97','98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114']:
    
    # get the processed matrices where only the most important edges are kept and stopwords removed
    d_matrix_, r_matrix_, merged_vocab, lansa_list = mergeVocabsAndMatrices(weight_type, i_split) # threshold weak connections here

    # create an empty counter for each wordpair in the lansa_list
    document_counter = [0]*len(lansa_list)

    # now, for each file, start counting
    fo_uni = open('features'+i_split+'/feature_matrix_row_filenames.txt', 'r')
    total_filenames_constant = fo_uni.read().split('\n')[:-1]
    fo_uni.close()

    file_counter = 0
    for filename in total_filenames_constant:

        # read the words in the current file (training or test, doesn't matter)
        f_text = open('House'+ str(i_split) + '_unigrams/' + filename, 'r')
        file_vocab_words = list(set( f_text.read().split(' ') ))
        #file_vocab_words = eliminateNonVocabWords(text_file_words, merged_vocab)
        #del text_file_words
        f_text.close()

        possible_wordpairs_set = permutations(file_vocab_words, 2)
        actual_wordpairs_set = list(set(possible_wordpairs_set).intersection(lansa_list))

        print('Num found from current file ' + str(len(actual_wordpairs_set)))
        # find each possible word pair - if they exist in current document
        for pair_ in actual_wordpairs_set: # each pair is a tuple
            pair_index = lansa_list.index(pair_)
            document_counter[pair_index] += 1

        if (total_filenames_constant.index(filename) % 50) == 0:
            print('Reached 500 modulus ' + str(filename))

    print(max(document_counter))

    fo_lansa = open('features'+ str(i_split) + '/list_of_acceptable_lansa_features.txt', 'w')
    for lansa_pair in lansa_list:

        index_ = lansa_list.index(lansa_pair)
        if document_counter[index_] >= thresh: # this lansa is selected, add to the file
            fo_lansa.write(lansa_pair[0] + '_' + lansa_pair[1] + '\n')

fo_lansa.close()
