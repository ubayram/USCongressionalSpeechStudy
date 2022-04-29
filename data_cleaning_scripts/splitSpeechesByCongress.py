# author: Ulya Bayram
# purpose is to download all 114th Congress data
# later I'll eliminate the non-senate data
import os

def correctText(speech_text):

    # find the periods, if a word after period is lowercase, replace period with comma
    # remember though, abbreviations also have periods. Get rid of some
    speech_text = speech_text.replace('U.S.', 'UnitedStates')
    speech_text = speech_text.replace('Mr.', 'Mr')
    speech_text = speech_text.replace('Mrs.', 'Mrs')
    speech_text = speech_text.replace('Ms.', 'Ms')
    speech_text = speech_text.replace('a.m.', '')
    speech_text = speech_text.replace('p.m.', '')
    # then, remaining periods should be mostly the end of sentence info, or those replacing comma's
    period_separated = speech_text.split('.')

    if len(period_separated) > 2: # to make sure there are at least 2 sentences, or comma separated stuff
        new_text = period_separated[0]

        for i_piece in range(len(period_separated)-1):
            no_space_sentence = period_separated[i_piece+1].replace(' ', '') # delete the spaces to eliminate them, so we can check for uppercase

            # to check whether the first word of the line following . is starting with uppercase
            if len(no_space_sentence) > 0:
                if no_space_sentence[0].islower(): # if true, then replace with the correct sign (,)
                    new_text += ',' + period_separated[i_piece+1]
                else:
                    new_text += '.' + period_separated[i_piece+1]
        # end of the current speech text. does it end with a period? if so, it is ready to be returned
        if new_text[-1] == '.':
            return new_text
        else:
            new_text += '.' # add the final period
            return new_text
    else: # not enough sentences to begin with, return just an empty string
        return ''

# check the unsuable and irrelevant texts to eliminate them
def checkTextsForElimination(speech_text):

    # eliminate few-sentence long texts
    period_separated = speech_text.split('.')

    if len(period_separated) < 5 : # remove those speeches with less than 5 sentences long
        return 1 # returning 1 means, yes, eliminate
    elif 'take the opportunity to express my appreciation to' in speech_text and 'intern' in speech_text:
        return 1 # these speeches are intern appreciation, delete them
    else:
        return 0 # rest includes chat about ideologies, bills, complaints, honoring death ppl
    # initially I was going to delete the honorings to death or congratulating people, but I've read some and
    # realized these honorings might be party/ideology related. For example, republicans are honoring those deads that have been
    # church members, and they praise them by their religion. Democrats, on the other hand, congratulating research center openings etc.

# ------------------------------------------------------------------------------------------------------------
# read the speech data, indexed by the speech id's

selected_congresses = ['097', '098', '099', '100', '101', '102', '103', '104', '105', '106'] #'114', '113', '112', '111', '110', '109', '108', '107']

for selected_congress in selected_congresses:

    if not os.path.isdir('Senate_' + selected_congress):
        os.mkdir('Senate_' + selected_congress)
    if not os.path.isdir('House_' + selected_congress):
        os.mkdir('House_' + selected_congress)

    speech_fo = open('hein-daily/speeches_' + selected_congress + '.txt', 'r')
    date_fo = open('hein-daily/descr_' + selected_congress + '.txt', 'r')

    # collect the speeches of this congress here, associated with it's unique speech id
    # includes house, senate speeches, all.
    speech_dict = {}
    c = 0
    for line in speech_fo:
        if c > 0: # skip the header
            speech_id = line.split('|')[0]
            speech_text = line.split('|')[1]

            # here, preprocess (or rather fix) the text's problems, especially create the correct punctuations
            speech_text = correctText(speech_text)

            # add this text to the dictionary only if it is long and relevant enough
            eliminate_flag = checkTextsForElimination(speech_text)

            if eliminate_flag==0: # if the flag is 0 - meaning don't eliminate, then save the data, else, ignore the speech
                speech_dict[speech_id] = speech_text
        else: # skip the header
            c = 1
    speech_fo.close()
    #print(speech_dict)

    date_dict = {}
    c = 0
    for line in date_fo:
        if c > 0:
            speech_id = line.split('|')[0]
            date_ = line.split('|')[2]

            date_dict[speech_id] = date_
        else:
            c = 1
    date_fo.close()
    #print(date_dict)

    # read the metadata
    map_fo = open('hein-daily/' + selected_congress + '_SpeakerMap.txt', 'r')
    c = 0
    for line in map_fo:
        if c > 0:
            speech_id = line.split('|')[1]
            name_ = line.split('|')[3] + ' ' + line.split('|')[2]
            chamber_ = line.split('|')[4]
            party_ = line.split('|')[7]

            if party_ != 'I' and speech_id in speech_dict.keys():
                filenamew = party_.lower() + '_' + name_.replace(' ', '_').lower() + '_' + date_dict[speech_id] + '_' + speech_id + '.txt'

                # start writing up what you read here
                if chamber_ == 'H':
                    print('evet, house ' + filenamew)
                    fo_w = open('House_' + selected_congress + '/' + filenamew, 'w')
                    fo_w.write(speech_dict[speech_id])
                    fo_w.close()
                elif chamber_ == 'S':
                    print('evet, senate ' + filenamew)
                    fo_w = open('Senate_' + selected_congress + '/' + filenamew, 'w')
                    fo_w.write(speech_dict[speech_id])
                    fo_w.close()
        else:
            c = 1
    map_fo.close()
