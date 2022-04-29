# author: Ulya Bayram
# ulyabayram@gmail.com
# This code extracts unigrams from all data files and saves them in a numpy matrix format
# for easy recalling and operations
import numpy as np
import glob
from collections import Counter
from spreadActivation_v_sentence_weighted import *
import nltk
from itertools import combinations
import operator
from scipy import sparse

def separateGroupFiles(files_list):

    d_list = []
    r_list = []

    # remember, we have two of those last empty lines, so remove both
    if '' in files_list:
        del files_list[files_list.index('')]
    if '' in files_list:
        del files_list[files_list.index('')]

    for filename in files_list:
        #print(filename)
        #print(len(filename))
        if 'd' in filename[0]:
            d_list.append(filename)
        elif 'r' in filename[0]:
            r_list.append(filename)
        else:
            print('Skipping last - empty - line!')

    return d_list, r_list

def getVocab(filename):

    fo = open(filename, 'r')
    vocab_ = []
    for line in fo:
        word_ = line.split()[0]
        vocab_.append(word_)
    return vocab_

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

    s_matrix_merged = np.where(difference_matrix < 0.01, 0, s_matrix_merged)
    c_matrix_merged = np.where(difference_matrix < 0.01, 0, c_matrix_merged)

    new_difference_matrix = s_matrix_merged + c_matrix_merged

    # now, compute the acceptable word pairs list (lansa feature set)
    wordpairs_list = []
    f_list = open('features'+i_num+'/list_of_acceptable_lansa_features.txt', 'r')
    pairs_list = f_list.read().split('\n')
    for pair in pairs_list:
        if len(pair) > 2:
            word1 = pair.split('_')[0]
            word2 = pair.split('_')[1]
            index_1 = merged_vocab.index(word1)
            index_2 = merged_vocab.index(word2)
            wordpairs_list.append((index_1, index_2))
        #for index_2 in xrange(index_1+1, len(merged_vocab)):
        #    if new_difference_matrix[index_1, index_2] > 0:
        #        wordpairs_list.append((index_1, index_2))

    print('Num of wordpairs (lansa features) collected ' + str(len(wordpairs_list)))

    return s_matrix_merged, c_matrix_merged, merged_vocab, wordpairs_list

# takes the list of test data words,
# returns a binary vector, where test words which match the vocab has value 1
def generateInitialActivationVector(vocab, test_words, a):

    # initialize the vector
    vec_ = np.zeros((1, len(vocab)))[0]

    for w in test_words:
        i = vocab.index(w) # every test word must be in vocab, since we eliminated those who aren't in
        vec_[i] = a

    return vec_

def getLANSAFeatureVectorFromActiveNodes(A_curr, d_matrix, r_matrix, wordpairs_list, merged_vocab):

    # initialize an empty feature vector of constant size
    feature_vector = np.zeros( (1, len(wordpairs_list)) )

    # find the list of active nodes (words in the current file)
    found_indices = np.where(A_curr > 0)[0]

    diff_matrix = d_matrix - r_matrix

    possible_wordpairs_set = combinations(found_indices, 2)
    actual_wordpairs_set = list(set(possible_wordpairs_set).intersection(wordpairs_list))

    # for each active node pair
    for pair_ in actual_wordpairs_set:
        i = pair_[0]
        j = pair_[1]

        # if only two active nodes have an edge between, collect the weight between them
        if diff_matrix[i, j] != 0:

            tuple1 = (i, j)
            tuple2 = (j, i)

            # find the tuple of the word pair in the list
            if tuple1 in wordpairs_list:
                tuple_index = wordpairs_list.index( tuple1 )
                feature_vector[0, tuple_index] = diff_matrix[i, j]
            elif tuple2 in wordpairs_list:
                tuple_index = wordpairs_list.index( tuple2 )
                feature_vector[0, tuple_index] = diff_matrix[j, i]

    return feature_vector

def applyFeatureSelection(training_matrix, feature_names):

    feature_names = np.array(feature_names)

    # binarize the matrix elements based on being zero or nonzero, higher than 0 items set to 1
    binary_training_matrix = np.where(training_matrix!=0, 1, 0)

    # remove those columns that sum up to 0 or 1 (e.g. no values in the column or only 1 value)
    summed_matrix_columnwise = np.sum(binary_training_matrix, axis=0)
    indices_to_be_removed = np.where(summed_matrix_columnwise < 100)[0]
    
    training_matrix = np.delete(training_matrix, indices_to_be_removed, axis=1)
    feature_names  = np.delete(feature_names, indices_to_be_removed, axis=0)

    return training_matrix, feature_names
# ----------------------------- main -------------------------------

weight_type = 'correlation_coeff'

for i_split in ['100']:
    # get the processed matrices where only the most important edges are kept and stopwords removed
    d_matrix_, r_matrix_, merged_vocab, lansa_list = mergeVocabsAndMatrices(weight_type, i_split) # threshold weak connections here

    # read the desired filename order for filling the feature matrix - so it'll match the rest of the statistical features
    fo_uni = open('features'+i_split+'/feature_matrix_row_filenames.txt', 'r')
    total_filenames_constant = fo_uni.read().split('\n')[:-1]
    fo_uni.close()

    # initialize the feature matrices for the three feature types
    feature_matrix_lansa = sparse.lil_matrix( np.zeros( (len(total_filenames_constant), len(lansa_list)) ) )

    file_counter = 0
    for filename in total_filenames_constant:

        # read the words in the current file (training or test, doesn't matter)
        f_text = open('House'+ str(i_split) + '_unigrams/' + filename, 'r')
        text_file_words = list(set( f_text.read().split(' ') ))
        file_vocab_words = eliminateNonVocabWords(text_file_words, merged_vocab)
        del text_file_words
        f_text.close()

        # -------- spreading activation operations here / cut the spread part because this is no spread case -------
        A_curr = generateInitialActivationVector(merged_vocab, file_vocab_words, 1)

        # get the feature vector for the current training file based on the spreading results
        feature_vector_lansa = getLANSAFeatureVectorFromActiveNodes(A_curr, d_matrix_, r_matrix_, lansa_list, merged_vocab)

        # save the feature vectors in the feature matrices
        feature_matrix_lansa[file_counter, :] = feature_vector_lansa

        print('Feature vector toplami ' + str(np.sum(np.abs(feature_vector_lansa))))
        file_counter += 1

        del feature_vector_lansa, A_curr, file_vocab_words

    # eliminate the uninformative features from the columns before saving the matrix - space reduction
    # read the training filenames to know by which files we'll eliminate infreq lansa's
    #fo_train = open('train'+ str(i_split) + '.txt', 'r')
    #train_filenames = fo_train.read().split('\n')
    #fo_train.close()
    
    #d_training_files, r_training_files = separateGroupFiles(train_filenames)

    feature_matrix_lansa = feature_matrix_lansa.toarray()
    feature_matrix_lansa, lansa_list = applyFeatureSelection(feature_matrix_lansa, lansa_list)

    print('After eliminating edge features not existing in >= 50 speeches, remaining lansa ' + str(np.shape(feature_matrix_lansa)))
    np.savetxt('features'+i_split+'/lansa_feature_matrix.txt', feature_matrix_lansa)

    # save the features corresponding to the feature matrix columns per motif feature type
    fo_feature_word_pairs = open('features'+i_split+'/lansa_matrix_columnames.txt', 'w')
    for i_pair in xrange(len(lansa_list)):
        word1 = merged_vocab[ lansa_list[i_pair][0] ]
        word2 = merged_vocab[ lansa_list[i_pair][1] ]

        fo_feature_word_pairs.write(word1 + '_' + word2 + '\n')
    fo_feature_word_pairs.close()

    del feature_matrix_lansa, merged_vocab, lansa_list, d_matrix_, r_matrix_
