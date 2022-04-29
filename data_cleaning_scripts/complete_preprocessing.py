# Automatically select 6000 data from the set, half d half r
# read the selected ones, extract the unigrams and save to its folder
# then extract the sentences and save to another folder
# while at it, randomly divide the set into training and test

import glob
import os
import re
import random

def cleanLocationInformation(filename_list):

    new_list = []
    for filename_ in filename_list:
        new_list.append(filename_.split('/')[1])

    return new_list

def saveTrainingTestSplit(files_a, files_c, i):
    fo_train = open('train'+ i +'_1.txt', 'w')
    fo_test = open('test'+ i +'_1.txt', 'w')

    num_test = 1000
    num_train = 2000

    selected_test_files_a = random.sample(files_a, num_test)
    selected_test_files_c = random.sample(files_c, num_test)

    remaining_a = list(set(files_a) - set(selected_test_files_a))
    remaining_c = list(set(files_c) - set(selected_test_files_c))

    selected_train_files_a = random.sample(remaining_a, num_train)
    selected_train_files_c = random.sample(remaining_c, num_train)
    # loop over all the filenames in the corpus, use if statements to find which are test
    c = 0
    # write the suicidal files first
    for file_ in files_a:

        if file_ in selected_test_files_a:
            fo_test.write(file_ + '\n')
        if file_ in selected_train_files_a:
            fo_train.write(file_ + '\n')

    # write the control files next
    for file_ in files_c:

        if file_ in selected_test_files_c:
            fo_test.write(file_ + '\n')
        if file_ in selected_train_files_c:
            fo_train.write(file_ + '\n')

    fo_train.close()
    fo_test.close()

    return selected_train_files_a+selected_test_files_a, selected_train_files_c+selected_test_files_c

def cleanSpeechData(txt_):
    # delete some unnecessary stuff
    txt_ = txt_.replace('Mr Speaker. ', '')
    txt_ = txt_.replace('Mr Speaker, ', '')
    txt_ = txt_.replace('Madam Speaker. ', '')
    txt_ = txt_.replace('Madam Speaker, ', '')
    txt_ = txt_.replace('Mr Chairman. ', '')
    txt_ = txt_.replace('Mr Chairman, ', '')
    txt_ = txt_.replace('Madam Chair. ', '')
    txt_ = txt_.replace('h.r.', '')
    txt_ = txt_.replace('Madam Chair, ', '')
    txt_ = txt_.replace('\n.', '')
    txt_ = txt_.replace('-', " ")
    #print(txt_)
    # delete some symbols if they exist within the text
    txt_ = re.sub(r'\[(.*?)\]', '', txt_)

    txt_ = re.sub(r'\,|\(|\)|\[|\]|\*|\&|\$|\@|\%|\-|\:|\"|\_|\+|\#|\;|\\|\/', ' ', txt_)

    txt_ = txt_.replace('..', '.') # replace multiple periods with a single one

    # replace periods with our sentence boundary signs
    txt_ = txt_.replace('.', ' * ')
    txt_ = txt_.replace('?', ' * ')
    txt_ = txt_.replace('!', ' * ')
    
    txt_ = re.sub(r' +', ' ', txt_)
    txt_ = txt_.lower()

    # clean the numericals from this text - convert them to 17 - my constant
    text_ = txt_.split(' ')

    return text_

def makeProperForSentences(words_list):

    complete_text = " ".join(words_list)
    del words_list
    complete_text = complete_text.replace('\n', '') # delete newline characters if there're any
    complete_text = complete_text.replace(" ' ", '') # delete if there're aposthropes hanging out alone
    complete_text = re.sub(r'[^\x00-\x7f]',r'', complete_text) # delete any non-ascii characters if there're any

    # now split the whole text into a list of words
    words_list = complete_text.split()
    words_list = removeSpaceFromWords(words_list)

    all_text = " ".join(words_list)

    return all_text

def makeProperForUnigrams(words_list):

    complete_text = " ".join(words_list)
    del words_list
    complete_text = complete_text.replace('\n', '') # delete newline characters if there're any
    complete_text = complete_text.replace(' *', '') # delete the end of sentence marks
    complete_text = complete_text.replace(" ' ", '') # delete if there're aposthropes hanging out alone
    complete_text = re.sub(r'[^\x00-\x7f]',r'', complete_text) # delete any non-ascii characters if there're any

    # now split the whole text into a list of words
    words_list = complete_text.split()
    words_list = removeSpaceFromWords(words_list)

    all_text = " ".join(words_list)

    return all_text

def removeSpaceFromWords(tlist):

    for wordi in xrange(len(tlist)):
        word = tlist[wordi]
        if ' ' in tlist[wordi]:
            word = word.replace(' ', '')

        tlist[wordi] = word
    return tlist
#--------------------------------------------------------------------------
i_list = ['113']#['098', '099', '101', '102', '104', '105', '107', '108', '110', '111']#['100', '103', '106', '109', '112']

for i in i_list:
    print('Current ' + i)
    files_a = cleanLocationInformation( glob.glob('House_'+i+'/d_*.txt') )
    files_c = cleanLocationInformation( glob.glob('House_'+i+'/r_*.txt') )

    # select 3000 speeches from each set at random
    random.shuffle(files_a)
    random.shuffle(files_c)

    d_files, r_files = saveTrainingTestSplit(files_a, files_c, i)

    del files_a, files_c

    # now read the selected files and pre-process them
    os.system('mkdir House' + i + '_sentences')
    os.system('mkdir House' + i + '_unigrams')

    for file_ in d_files + r_files:

        fo = open('House_'+i+'/' + file_, 'r')
        text_ = fo.read()
        fo.close()
        text_ = cleanSpeechData(text_) # returns a list of words

        uni_text = makeProperForUnigrams(text_)
        sent_text = makeProperForSentences(text_)

        fo = open('House'+i+'_sentences/'+file_, 'w')
        fo.write(sent_text)
        fo.close()

        fo = open('House'+i+'_unigrams/'+file_, 'w')
        fo.write(uni_text)
        fo.close()
        
