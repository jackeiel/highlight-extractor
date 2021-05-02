import os
import json
import re

# from .GRU_discourse_tagger_generator_bert import PassageTagger
from .discourse_tagger_generator_bert2 import PassageTagger
from .crf import CRF
from .attention import TensorAttention
from .custom_layers import HigherOrderTimeDistributedDense
from .util import from_BIO

from tensorflow.keras.models import model_from_json

import pandas as pd
import numpy as np

import spacy
nlp = spacy.load("en_core_web_sm")
config = {"punct_chars": None}
nlp.add_pipe("sentencizer", config=config)


class HighlightExtractor:
    # set all of the params needed for the PassageTagger
    def __init__(self, scibert_path, tagger_path, use_attention=False, att_context='LSTM_clause', lstm=False,
                 bidirectional=False, crf=False, batch_size=10, maxseqlen=None, maxclauselen=None):
        self.scibert_path = scibert_path  # need to set for PassageTagger class
        self.tagger_path = tagger_path
        self.use_attention = use_attention
        self.att_context = att_context
        self.lstm = lstm
        self.bidirectional = bidirectional
        self.crf = crf
        self.batch_size = batch_size
        self.maxseqlen = maxseqlen
        self.maxclauselen = maxclauselen
        # PassageTagger takes a dict of params
        self.params = {'repfile': self.scibert_path,
                       'tagger_path': self.tagger_path,
                       'use_attention': self.use_attention,
                       'att_context': self.att_context,
                       'lstm': self.lstm,
                       'bidirectional': self.bidirectional,
                       'crf': self.crf,
                       'batch_size': self.batch_size,
                       'maxseqlen': self.maxseqlen,
                       'maxclauselen': self.maxclauselen}

        # load tagging model
        model_ext = "att=%s_cont=%s_lstm=%s_bi=%s_crf=%s" % (
            str(self.use_attention), self.att_context, str(self.lstm), str(self.bidirectional),
            str(self.crf))
        model_config_file = open(os.path.join(self.tagger_path, "model_%s_config.json" % model_ext), "r")
        model_weights_file_name = os.path.join(self.tagger_path, "model_%s_weights" % model_ext)
        model_label_ind = os.path.join(self.tagger_path, "model_%s_label_ind.json" % model_ext)
        # build tagger
        self.nnt = PassageTagger(self.params)
        self.nnt.tagger = model_from_json(model_config_file.read(),
                                          custom_objects={"TensorAttention": TensorAttention,
                                                          "HigherOrderTimeDistributedDense": HigherOrderTimeDistributedDense,
                                                          "CRF": CRF})
        if not self.params["use_attention"]:
            self.params["maxseqlen"] = self.nnt.tagger.inputs[0].shape[1]
            self.params["maxclauselen"] = None
        self.nnt.tagger.load_weights(model_weights_file_name)
        label_ind_json = json.load(open(model_label_ind))
        self.label_ind = {k: int(label_ind_json[k]) for k in label_ind_json}

    def tag(self, text_or_path, from_file=False, parsed=True):
        if from_file:
            with open(text_or_path, 'r') as file:
                text = file.read()
            if parsed:
                doc = text.split('\n')
            else:
                doc = nlp(text)
                doc = [str(sent) for sent in doc.sents]

        else:
            if parsed:
                doc = text_or_path
            else:
                doc = nlp(text_or_path)
                doc = [str(sent) for sent in doc.sents]

        print('Tagging', len(doc), 'sentences...')

        tfile = open('temp_file.txt', mode='w')
        for sent in doc:
            tfile.write(str(sent).lower())
            tfile.write('\n')
        tfile.write('\n')
        test_file = tfile.name
        tfile.close()

        test_seq_lengths, test_generator = self.nnt.make_data(test_file,
                                                              label_ind=self.label_ind,
                                                              train=False)

        os.remove(test_file)

        pred_probs1, pred_label_seqs, _ = self.nnt.predict(test_generator, test_seq_lengths, tagger=self.nnt.tagger)

        pred_label_seqs = from_BIO(pred_label_seqs)[0] #  list of list, get first

        sentences = [str(sent) for sent in doc]
        s_len = len(sentences)
        pred_probs = np.max(pred_probs1[0], axis=-1)[-s_len:] # they pre-pad probs

        def get_tense(s):
            s = nlp(s)
            tense = 'UNK'
            for token in s:
                if token.dep_ == 'ROOT':
                    try:
                        tense = token.morph.get('Tense')[0]
                    except IndexError:
                        pass
            return tense

        tenses = [get_tense(s) for s in sentences]
        # return sentences, pred_probs, pred_label_seqs, pred_probs1
        if s_len > 40:
            a = {'sentence': sentences, 'tag': pred_label_seqs, 'prob': pred_probs, 'tense': tenses}
            df = pd.DataFrame.from_dict(a, orient='index')
            df = df.transpose()
#             df = pd.DataFrame({'sentence': sentences[-40:], 'tag': pred_label_seqs[-40:],
#                                'prob': pred_probs, 'tense': tenses})

            return df
        else:
            df = pd.DataFrame({'sentence': sentences, 'tag': pred_label_seqs,
                           'prob': pred_probs, 'tense': tenses})
            return df

    @staticmethod
    def get_highlights(df):
        regex = re.compile(r' \([^)]*\)')  # we need this to remove parenths

        df = df.sort_values('prob')
        highs = df.sentence.tail(5).tolist()
        highs = [re.sub(regex, '', h) for h in highs]

        return highs

    @staticmethod
    def print_highlights(highs):
        for h in highs:
            print(u'\u2022', h)