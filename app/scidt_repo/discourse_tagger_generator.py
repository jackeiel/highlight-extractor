import warnings
import sys
import codecs
import numpy as np
import argparse
import json
import pickle

from rep_reader import RepReader
from util import read_passages, evaluate, make_folds, clean_words, test_f1, to_BIO, from_BIO, from_BIO_ind, arg2param

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
import keras.backend as K
K.set_session(sess)
from keras.activations import softmax
from keras.regularizers import l2
from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed, Bidirectional
from keras.callbacks import EarlyStopping,LearningRateScheduler
from keras.optimizers import Adam, RMSprop, SGD
from crf import CRF
from attention import TensorAttention
from custom_layers import HigherOrderTimeDistributedDense
from generator import DiscourseGenerator

def reset_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)

class PassageTagger(object):
    def __init__(self, params, word_rep_file=None, pickled_rep_reader=None):
        self.params = params
        if pickled_rep_reader:
            self.rep_reader = pickled_rep_reader
        elif word_rep_file:
            self.rep_reader = RepReader(word_rep_file)
        self.input_size = self.rep_reader.rep_shape[0]
        self.tagger = None
    
    def make_data(self, trainfilename, maxseqlen=None, maxclauselen=None, label_ind=None, train=False):
        use_attention = self.params["use_attention"]
        maxseqlen = self.params["maxseqlen"]
        maxclauselen = self.params["maxclauselen"]
        batch_size = self.params["batch_size"]

        str_seqs, label_seqs = read_passages(trainfilename, is_labeled=train)
        print("Filtering data")
        str_seqs = clean_words(str_seqs)
        label_seqs = to_BIO(label_seqs)
        if not label_ind:
            self.label_ind = {"none": 0}
        else:
            self.label_ind = label_ind
        seq_lengths = [len(seq) for seq in str_seqs]
        if not maxseqlen:
            maxseqlen = max(seq_lengths)
        if not maxclauselen:
            if use_attention:
                clauselens = []
                for str_seq in str_seqs:
                    clauselens.extend([len(clause.split()) for clause in str_seq])
                    
                maxclauselen = np.round(np.mean(clauselens) + 3 * np.std(clauselens)).astype(int)
        X = []
        Y = []
        Y_inds = []
        init_word_rep_len = len(self.rep_reader.word_rep) # Vocab size
        if len(self.label_ind)<=1:
            for str_seq, label_seq in zip(str_seqs, label_seqs):
                for label in label_seq:
                    if label not in self.label_ind:
                        # Add new labels with values 0,1,2,....
                        self.label_ind[label] = len(self.label_ind)
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        discourse_generator = DiscourseGenerator(self.rep_reader, str_seqs, label_seqs, self.label_ind, batch_size, use_attention, maxseqlen, maxclauselen, train, self.input_size)
        self.maxseqlen = maxseqlen
        self.maxclauselen = maxclauselen
        return seq_lengths, discourse_generator # One-hot representation of labels

    def predict(self, discourse_generator, test_seq_lengths=None, tagger=None):
        if not tagger:
            tagger = self.tagger
        if test_seq_lengths is None:
            assert(False)
        else:
            x_lens = test_seq_lengths
        pred_probs = tagger.predict_generator(discourse_generator)
        pred_inds = np.argmax(pred_probs, axis=2)
        pred_label_seqs = []
        for pred_ind, x_len in zip(pred_inds, x_lens):
            pred_label_seq = [self.rev_label_ind[pred] for pred in pred_ind][-x_len:]
            # If the following number is positive, it means we ignored some clauses in the test passage to make it the same length as the ones we trained on.
            num_ignored_clauses = max(0, x_len - len(pred_label_seq))
            # Make labels for those if needed.
            if num_ignored_clauses > 0:
                warnings.warn("Test sequence too long. Ignoring %d clauses at the beginning and labeling them none." % num_ignored_clauses)
                ignored_clause_labels = ["none"] * num_ignored_clauses
                pred_label_seq = ignored_clause_labels + pred_label_seq
            pred_label_seqs.append(pred_label_seq)
        return pred_probs, pred_label_seqs, x_lens

    def fit_model(self, train_generator, validation_generator, reg=0):
        use_attention = self.params["use_attention"]
        att_context = self.params["att_context"]
        lstm = self.params["lstm"]
        bidirectional = self.params["bidirectional"]
        crf = self.params["crf"]
        embedding_dropout = self.params["embedding_dropout"]
        high_dense_dropout = self.params["high_dense_dropout"]
        attention_dropout = self.params["attention_dropout"]
        lstm_dropout = self.params["lstm_dropout"]
        word_proj_dim = self.params["word_proj_dim"]
        lr = self.params["lr"]
        epoch = self.params["epoch"]
        batch_size = self.params["batch_size"]
        hard_k = self.params["hard_k"]
        att_proj_dim = self.params["att_proj_dim"]
        rec_hid_dim = self.params["rec_hid_dim"]
        lstm_dim = self.params["lstm_dim"]
        validation_split = self.params["validation_split"]
        
        early_stopping = EarlyStopping(patience = 2)
        num_classes = len(self.label_ind)
        if use_attention:
            inputs = Input(shape=(self.maxseqlen, self.maxclauselen, self.input_size))
            x = Dropout(embedding_dropout)(inputs)
            x = HigherOrderTimeDistributedDense(input_dim=self.input_size, output_dim=word_proj_dim, reg=reg)(x)
            att_input_shape = (self.maxseqlen, self.maxclauselen, word_proj_dim)
            x = Dropout(high_dense_dropout)(x)
            x = TensorAttention(att_input_shape, context=att_context, hard_k=hard_k, proj_dim = att_proj_dim, rec_hid_dim = rec_hid_dim)(x)
            x = Dropout(attention_dropout)(x)
        else:
            inputs = Input(shape=(self.maxseqlen, self.input_size))
            x = Dropout(embedding_dropout)(inputs)
            x = Dense(input_dim=self.input_size, units=word_proj_dim)(x)
        
        if bidirectional:
            x = Bidirectional(LSTM(input_shape=(self.maxseqlen,word_proj_dim), units=lstm_dim, 
                                          return_sequences=True,kernel_regularizer=l2(reg),
                                          recurrent_regularizer=l2(reg), 
                                          bias_regularizer=l2(reg)))(x)
            x = Dropout(lstm_dropout)(x) 
        elif lstm:
            x = LSTM(input_shape=(self.maxseqlen,word_proj_dim), units=lstm_dim, return_sequences=True,
                            kernel_regularizer=l2(reg),
                            recurrent_regularizer=l2(reg), 
                            bias_regularizer=l2(reg))(x)
            x = Dropout(lstm_dropout)(x) 

        if crf:
            Crf = CRF(num_classes,learn_mode="join")
            discourse_prediction = Crf(x)
            tagger = Model(inputs=inputs, outputs=[discourse_prediction])        
        else:
            discourse_prediction = TimeDistributed(Dense(num_classes, activation='softmax'),name='discourse')(x)
            tagger = Model(inputs=inputs, outputs=[discourse_prediction])
                    
        def step_decay(current_epoch):
            initial_lrate = lr
            drop = 0.5
            epochs_drop = epoch/2
            lrate = initial_lrate * np.power(drop,  
                   np.floor((1+current_epoch)/epochs_drop))
            return lrate
        
        lr_fractions = [1]
        decay = 0
        for lr_fraction in lr_fractions:
            adam = Adam(lr=lr*lr_fraction, decay = decay)
            if crf:
                #rmsprop = RMSprop(lr=lr,decay = decay)
                tagger.compile(optimizer=adam, loss=Crf.loss_function, metrics=[Crf.accuracy])
            else:
                tagger.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

            tagger.fit_generator(train_generator, validation_data=validation_generator, epochs=epoch, callbacks=[early_stopping], verbose=2)

        tagger.summary()
        return tagger

    def train(self, train_generator, validation_generator):
        save = self.params["save"]
        
        f_mean, f_std, original_f_mean, original_f_std = 0,0,0,0
            
        self.tagger = self.fit_model(train_generator, validation_generator)
        if save:
            model_ext = "att=%s_cont=%s_lstm=%s_bi=%s_crf=%s"%(str(self.params["use_attention"]),self.params["att_context"], str(self.params["lstm"]), str(self.params["bidirectional"]), str(self.params["crf"]))
            model_config_file = open("model_%s_config.json"%model_ext, "w")
            model_weights_file_name = "model_%s_weights"%model_ext
            model_label_ind = "model_%s_label_ind.json"%model_ext
            model_rep_reader = "model_%s_rep_reader.pkl"%model_ext
            print(self.tagger.to_json(), file=model_config_file)
            self.tagger.save_weights(model_weights_file_name, overwrite=True)
            json.dump(self.label_ind, open(model_label_ind, "w"))
            pickle.dump(self.rep_reader, open(model_rep_reader, "wb"))
        return f_mean, f_std, original_f_mean, original_f_std

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run LSTM discourse tagger")
    argparser.add_argument('--repfile', type=str, help="Word embedding file")
    argparser.add_argument('--train_file', type=str, help="Training file. One clause<tab>label per line and passages separated by blank lines.")
    argparser.add_argument('--validation_file', type=str, help="Validation file. One clause<tab>label per line and passages separated by blank lines.")
    argparser.add_argument('--test_file', type=str, help="Test file name, one clause per line and passages separated by blank lines.")
    argparser.add_argument('--use_attention', help="Use attention over words? Or else will average their representations", action='store_true')
    argparser.add_argument('--att_context', type=str, help="Context to look at for determining attention (word/clause)")
    argparser.set_defaults(att_context='word')
    argparser.add_argument('--lstm', help="Sentence level LSTM", action='store_true')
    argparser.add_argument('--bidirectional', help="Bidirectional LSTM", action='store_true')
    argparser.add_argument('--crf', help="Conditional Random Field", action='store_true')
    argparser.add_argument('--hard_k', help="Hard attention's choose top k")
    argparser.set_defaults(hard_k=0)
    argparser.add_argument('--lr', help="Learning rate")
    argparser.set_defaults(lr=1e-3)
    argparser.add_argument('--embedding_dropout', help="embedding_dropout rate")
    argparser.add_argument('--high_dense_dropout', help="high_dense_dropout rate")
    argparser.add_argument('--attention_dropout', help="attention_dropout rate")
    argparser.add_argument('--lstm_dropout', help="lstm_dropout rate")
    argparser.set_defaults(embedding_dropout=0.4)
    argparser.set_defaults(high_dense_dropout=0.4)
    argparser.set_defaults(attention_dropout=0.6)
    argparser.set_defaults(lstm_dropout=0.5)
    argparser.add_argument('--word_proj_dim', help="word_projection_dimension")
    argparser.set_defaults(word_proj_dim=225)
    argparser.add_argument('--lstm_dim', help="Discourse level LSTM dimension")
    argparser.set_defaults(lstm_dim=200)
    argparser.add_argument('--att_proj_dim', help="Attention projection dimension")
    argparser.set_defaults(att_proj_dim=110)
    argparser.add_argument('--rec_hid_dim', help="Attention RNN hidden dimension")
    argparser.set_defaults(rec_hid_dim=60)
    argparser.add_argument('--epoch', help="Training epoch")
    argparser.set_defaults(epoch=100)
    argparser.add_argument('--validation_split', help="validation_split")
    argparser.set_defaults(validation_split=0.1)
    argparser.add_argument('--save', help="Whether save the model or not",action='store_true')
    argparser.add_argument('--maxseqlen', help="max number of clauses per paragraph")
    argparser.set_defaults(maxseqlen=40)
    argparser.add_argument('--maxclauselen', help="max number of words per clause")
    argparser.set_defaults(maxclauselen=60)
    argparser.add_argument('--outpath', help="path of output labels")
    argparser.set_defaults(outpath="./")
    argparser.add_argument('--batch_size', help="batch size")
    argparser.set_defaults(batch_size=10)
    
    args = argparser.parse_args()
    params = arg2param(args)
    reset_random_seed(12345) # Good for word attention
    if args.train_file:
        params["train"] = True
        #assert args.repfile is not None, "Word embedding file required for training."
    else:
        params["train"] = False
    if args.test_file:
        params["test"] = True
    else:
        params["test"] = False

    if not params["train"] and not params["test"]:
        raise(RuntimeError, "Please specify a train file or test files.")

    if params["maxseqlen"] <= 0:
        params["maxseqlen"] = None
    if params["maxclauselen"] <= 0:
        params["maxclauselen"] = None
    
    model_name = "att=%s_cont=%s_lstm=%s_bi=%s_crf=%s"%(str(params["use_attention"]), params["att_context"], str(params["lstm"]), str(params["bidirectional"]),str(params["crf"]))
    print(model_name)
    f_mean, f_std, original_f_mean, original_f_std = 0,0,0,0
    if params["train"]:
        # First returned value is sequence lengths (without padding)
        nnt = PassageTagger(params, word_rep_file=params["repfile"])
        if params["repfile"]:
            print("Using embedding weight to find embeddings.")
            _, train_generator = nnt.make_data(params["train_file"], train=True)
            _, validation_generator = nnt.make_data(params["validation_file"], label_ind=nnt.label_ind, train=True)
        else:
            assert(0)
        f_mean, f_std, original_f_mean, original_f_std = nnt.train(train_generator, validation_generator)
    if params["test"]:
        if params["train"]:
            label_ind = nnt.label_ind
        else:
            # Load the model from file
            model_ext = "att=%s_cont=%s_lstm=%s_bi=%s_crf=%s"%(str(params["use_attention"]), params["att_context"], str(params["lstm"]), str(params["bidirectional"]), str(params["crf"]))
            model_config_file = open("model_%s_config.json"%model_ext, "r")
            model_weights_file_name = "model_%s_weights"%model_ext
            model_label_ind = "model_%s_label_ind.json"%model_ext
            model_rep_reader = "model_%s_rep_reader.pkl"%model_ext
            rep_reader = pickle.load(open(model_rep_reader, "rb"))
            print("Loaded pickled rep reader")
            nnt = PassageTagger(params, pickled_rep_reader=rep_reader)
            nnt.tagger = model_from_json(model_config_file.read(), custom_objects={"TensorAttention":TensorAttention, "HigherOrderTimeDistributedDense":HigherOrderTimeDistributedDense,"CRF":CRF})
            print("Loaded model:")
            print(nnt.tagger.summary())
            nnt.tagger.load_weights(model_weights_file_name)
            print("Loaded weights")
            label_ind_json = json.load(open(model_label_ind))
            label_ind = {k: int(label_ind_json[k]) for k in label_ind_json}
            print("Loaded label index:", label_ind)
        if not params["use_attention"]:
            params["maxseqlen"] = nnt.tagger.inputs[0].shape[1]
            params["maxclauselen"] = None
        else:
            for l in nnt.tagger.layers:
                if ("TensorAttention" in l.name) or ("tensor_attention" in l.name):
                    params["maxseqlen"], params["maxclauselen"] = l.td1, l.td2
                    break

        print("Predicting on file %s"%(params["test_file"]))
        test_out_file_name = "predictions/"+params["test_file"].split("/")[-1].replace(".txt", "")+model_name+".out"
        outfile = open(test_out_file_name, "w")
        print("maxseqlen", params["maxseqlen"])
        
        test_seq_lengths, test_generator = nnt.make_data(params["test_file"], label_ind=label_ind, train=False)
        pred_probs, pred_label_seqs, _ = nnt.predict(test_generator, test_seq_lengths)

        pred_label_seqs = from_BIO(pred_label_seqs)
        for pred_label_seq in pred_label_seqs:
            for pred_label in pred_label_seq:
                print(pred_label,file = outfile)
            print("",file = outfile)
