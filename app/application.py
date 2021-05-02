#Highlight Extraction App
from flask import Flask, render_template, request


#Testing Below
#---------------
import json
import os
import re

import pandas as pd
import numpy as np
# from rouge import Rouge
import spacy
import os
import nltk.data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from scidt_repo.extract_highlights import HighlightExtractor
h = HighlightExtractor('models/scibert_scivocab_uncased', 'models/tagger', bidirectional=True)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#---------------

application = Flask(__name__)


@application.route('/')
def main():
    return render_template('app.html')


@application.route('/send', methods=['POST'])

def send(sum=sum):
    if request.method == 'POST':
        inputText = request.form['highlightArea']
        cleanText = inputText.replace('\n', ' ').replace('\r', '')
        encodeText = cleanText.encode("ascii", "ignore")
        decodeText = encodeText.decode()
        
        #CONVERT USER INPUT INTO SENTENCES
        allSentences = tokenizer.tokenize(decodeText)
        
        #SPLIT SENTENCES INTO LISTS | 40 SENTENCES PER LIST IS MAX (USING 30 TO ACCOMMODATE ABBREVIATIONS MID SENTENCE)
        l1 = allSentences[0:30]
        l2 = allSentences[30:60]
        l3 = allSentences[60:90]
        l4 = allSentences[90:120]
        l5 = allSentences[120:150]
        l6 = allSentences[150:180]
        l7 = allSentences[150:180]
        l8 = allSentences[180:210]
        l9 = allSentences[210:240]
        l10 = allSentences[240:270]
        l11 = allSentences[270:300]
        
        #SPLIT SENTENCES INTO LISTS | 40 SENTENCES PER LIST
        string1 = ' '.join([str(elem) for elem in l1])
        string2 = ' '.join([str(elem) for elem in l2])
        string3 = ' '.join([str(elem) for elem in l3])
        string4 = ' '.join([str(elem) for elem in l4])
        string5 = ' '.join([str(elem) for elem in l5])
        string6 = ' '.join([str(elem) for elem in l6])
        string7 = ' '.join([str(elem) for elem in l7])
        string8 = ' '.join([str(elem) for elem in l8])
        string9 = ' '.join([str(elem) for elem in l9])
        string10 = ' '.join([str(elem) for elem in l10])
        string11 = ' '.join([str(elem) for elem in l11])
        
        if len(string1) >0:
            tag1 = h.tag(string1, parsed=False)
        else:
            tag1 = pd.DataFrame()

        if len(string2) >0:
          tag2 = h.tag(string2, parsed=False)
        else:
          tag2 = pd.DataFrame()

        if len(string3) >0:
          tag3 = h.tag(string3, parsed=False)
        else:
          tag3 = pd.DataFrame()
    
        if len(string4) >0:
          tag4 = h.tag(string4, parsed=False)
        else:
          tag4 = pd.DataFrame()
        if len(string5) >0:
          tag5 = h.tag(string5, parsed=False)
        else:
          tag5 = pd.DataFrame()
        if len(string6) >0:
          tag6 = h.tag(string6, parsed=False)
        else:
          tag6 = pd.DataFrame()
        if len(string7) >0:
          tag7 = h.tag(string7, parsed=False)
        else:
          tag7 = pd.DataFrame()
        if len(string8) >0:
          tag8 = h.tag(string8, parsed=False)
        else:
          tag8 = pd.DataFrame()
        if len(string9) >0:
          tag9 = h.tag(string9, parsed=False)
        else:
          tag9 = pd.DataFrame()
        if len(string10) >0:
          tag10 = h.tag(string10, parsed=False)
        else:
          tag10 = pd.DataFrame()
        if len(string11) >0:
          tag11 = h.tag(string11, parsed=False)
        else:
          tag11 = pd.DataFrame()

        #a = h.tag(decodeText, parsed=False)

        #COMBINE TAGGED SENTENCES & SORT
        allTags = [tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8, tag9, tag10, tag11]
        finalTags = pd.concat(allTags)
        b = finalTags.sort_values('prob')
        #a = h.tag(decodeText, parsed=False)
        #b = a.sort_values('prob')
        sum = b.sentence.tail(5).tolist()

        return render_template('app.html', sum=sum)


if __name__ == "__main__":
    application.run()
