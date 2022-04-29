# Ulya Bayram
# this code is to apply another pass through the words in the file, some punctuations aren't properly removed during the first pass
# there are words like 'i, 'yes, 'then, 'you, etc. remaining, also there are *e, *f, for some reason
# these two cases will be considered and removed here.
import glob
import sys
import codecs
import re

def removeSpaceFromWords(tlist):

    for wordi in range(len(tlist)):
        word = tlist[wordi]
        if ' ' in tlist[wordi]:
            word = word.replace(' ', '')

        tlist[wordi] = word
    return tlist

def cleanStart(filelist):
    newlist = []

    for file_ in filelist:
        newlist.append(file_.split('/')[1])

    return newlist

###################################################################################
# save all chapters, all pre-processed but found plenty of errors, here pre-processing over the faults

#reload(sys)
#sys.setdefaultencoding('utf-8')

raw_filenames_list1 = cleanStart(glob.glob("House97_sentences/*.txt"))
#raw_filenames_list2 = cleanStart(glob.glob("House114_unigrams/*.txt"))

c = 0
for file_ in list(set(raw_filenames_list1)):
    # read the current file
    fo = open("House97_sentences/" + file_, 'r')
    complete_text = fo.read()
    fo.close()

    complete_text = complete_text.replace('\n', '') # delete newline characters if there're any
    complete_text = complete_text.replace(' *', '') # delete the end of sentence marks
    complete_text = complete_text.replace(" ' ", '') # delete if there're aposthropes hanging out alone
    complete_text = re.sub(r'[^\x00-\x7f]',r'', complete_text) # delete any non-ascii characters if there're any

    # now split the whole text into a list of words
    words_list = complete_text.split()
    words_list = removeSpaceFromWords(words_list)
    #print(words_list)

    fo = open('House97_unigrams/' + file_, 'w')
    fo.write(" ".join(words_list))
    fo.close()

    del fo, complete_text, words_list
