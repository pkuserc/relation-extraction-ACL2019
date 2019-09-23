
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import gc
import keras
import os
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Lambda, Embedding,Dropout, Activation,GRU,Bidirectional,Subtract, Permute, TimeDistributed, Reshape
from keras.layers import Conv1D,Conv2D,MaxPooling2D,GlobalAveragePooling1D,GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.layers import CuDNNGRU, CuDNNLSTM, SpatialDropout1D,Layer
from keras.layers.merge import concatenate, Concatenate, Average, Dot, Maximum, Multiply, Subtract
from keras.models import Model
from keras.optimizers import RMSprop,Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras import backend as K
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
import tensorflow as tf
from keras.activations import softmax

from keras.utils import plot_model
from keras.layers import *
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.special import erfinv

import gensim
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import to_categorical
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import re
from keras.engine.topology import Layer


# ##### read data

data = pd.read_csv('data_all_english.csv')


# ##### data preprocessing

data['between_docs'] = data['between_docs'].map(eval)
data['docs_'] = data['docs_'].map(eval)
tokenizer_api = Tokenizer()
tokenizer_api.fit_on_texts(data['docs_'])

data_char_seq = tokenizer_api.texts_to_sequences(data['between_docs'])
data['char_seq'] = pad_sequences(data_char_seq, maxlen = 16, padding='post').tolist()

LB = LabelEncoder()
LB.fit(list(data['entity_type_1']) + list(data['entity_type_2']))
print(len(LB.classes_))

data['entity_type_1_LB'] = LB.transform(data['entity_type_1'])
data['entity_type_2_LB'] = LB.transform(data['entity_type_2'])

data['between_tag'] = data['between_tag'].apply(eval)

data['tag_all_seq'] = pad_sequences(data['between_tag'], maxlen = 16, padding='post').tolist()


data['relation'] = data['relation'].fillna('Other')
label = list(data['relation'].unique())
label.remove('Other')


d2 = data[(data['relation'] != 'Other')|((data['relation'] == 'Other')&(data['entity_start_1'] < data['entity_start_2']))]
data = d2.copy()

data.index = range(data.shape[0])


y = data['relation'].values

LB_y = LabelEncoder()
LB_y.fit(y)
y = LB_y.transform(y)

y = to_categorical(y, np.max(y) + 1)

entity_start_1_in = []
entity_start_2_in = []
entity_end_1_in = []
entity_end_2_in = []


for i in range(data.shape[0]):
    
    temp = data.iloc[i, :]
    
    s1 = temp['entity_start_1']
    e1 = temp['entity_end_1']
    
    s2 = temp['entity_start_2']
    e2 = temp['entity_end_2']
    
    if s1<=s2:
        entity_start_1_in.extend([0])
        entity_end_1_in.extend([e1 - s1])
        
        entity_start_2_in.extend([s2 - s1])
        entity_end_2_in.extend([e2 - s1])
        
    else:
        entity_start_2_in.extend([0])
        entity_end_2_in.extend([e2 - s2])
        
        entity_start_1_in.extend([s1 - s2])
        entity_end_1_in.extend([e1 - s2])


data['entity_start_1_in'] = entity_start_1_in
data['entity_start_2_in'] = entity_start_2_in
data['entity_end_1_in'] = entity_end_1_in
data['entity_end_2_in'] = entity_end_2_in


max_len = 16

entity_location_vector_1 = []
entity_location_vector_2 = []
for i in range(data.shape[0]):
    
    if i%10000 == 0:
        print('*************%s**************'%i)
    temp = data.iloc[i, :]
    
    s1 = temp['entity_start_1_in']
    e1 = temp['entity_end_1_in']
    
    left = list(range(min(int(max_len/2), s1))) + [int(max_len/2)]*(s1 - int(max_len/2))
    left = [-1*x for x in left]
    left.reverse()
    location_vector_1 = left + [0]*(e1 - s1) +     list(range(min(int(max_len/2), int(max_len) - e1))) + [int(max_len/2)]*(int(max_len/2) - e1)
    
    entity_location_vector_1.append(np.array(location_vector_1))
    
    s2 = temp['entity_start_2_in']
    e2 = temp['entity_end_2_in']
    
    left = list(range(min(int(max_len/2), s2))) + [int(max_len/2)]*(s2 - int(max_len/2))
    left = [-1*x for x in left]
    left.reverse()
    location_vector_2 = left + [0]*(e2 - s2) +     list(range(min(int(max_len/2), int(max_len) - e2))) + [int(max_len/2)]*(int(max_len/2) - e2)
    
    entity_location_vector_2.append(np.array(location_vector_2))
    
data['entity_location_vector_1'] = entity_location_vector_1
data['entity_location_vector_2'] = entity_location_vector_2

def func(x):
    return x[:16]

data['entity_location_vector_1'] = data['entity_location_vector_1'].map(func)
data['entity_location_vector_2'] = data['entity_location_vector_2'].map(func)
data['tag_all_seq'] = data['tag_all_seq'].map(func)

y_mask = [[1, 1] if x != 'Other' else [0,1] for x in data['relation'] ]
y_mask = np.array(y_mask,dtype=np.float32)


#define input function
def get_input(X):
    
    input_ = [np.array(X.char_seq.values.tolist()),              np.array(X.entity_location_vector_1.values.tolist()),              np.array(X.entity_location_vector_2.values.tolist()),             np.array(X.tag_all_seq.values.tolist())]
    
    return input_


# ##### define model

def ranking_loss(y_true, y_pred):
    y_true_one_hot = y_true[:, :6]  # [batch_size * 10], one-hot, all zero for other

    y_label = y_true[:, 6:7]
    y_label = tf.cast(y_label, tf.int32)
    y_label = tf.reshape(y_label, shape=[-1])  # [ batch_size ], not one-hot label, a big number for other

    y_mask = y_true[:, 7:]  # [batch_size * 2], [1, 1] for positive and [0, 1] for other

    m_pos = 2.5
    m_neg = 0.5
    gamma = 2

    pos_score = tf.multiply(y_true_one_hot, y_pred)
    pos_score = tf.reduce_sum(pos_score, axis=1)

    pos_loss = tf.exp(gamma * (m_pos - pos_score))
    pos_loss = tf.log(1 + pos_loss)

    pred_top_2_val = tf.nn.top_k(y_pred, k=2).values
    pred_max_pos = tf.argmax(y_pred, axis=1)
    pred_max_pos = tf.cast(pred_max_pos, tf.int32)
    pred_is_max = tf.equal(pred_max_pos, y_label)
    pred_is_max = tf.cast(pred_is_max, tf.float32)
    pred_is_max = tf.expand_dims(pred_is_max, axis=-1)
    inv_pred_is_max = 1 - pred_is_max
    neg_score_mask = tf.concat([inv_pred_is_max, pred_is_max], axis=-1)
    neg_score = tf.multiply(pred_top_2_val, neg_score_mask)
    neg_score = tf.reduce_sum(neg_score, axis=1)
    neg_loss = tf.exp(gamma * (m_neg + neg_score))
    neg_loss = tf.log(1 + neg_loss)

    pos_loss = tf.expand_dims(pos_loss, axis=-1)
    neg_loss = tf.expand_dims(neg_loss, axis=-1)
    ranking_loss = tf.concat([pos_loss, neg_loss], axis=-1)
    ranking_loss = tf.multiply(y_mask, ranking_loss)
    ranking_loss = tf.reduce_sum(ranking_loss, axis=1)
    #     loss = tf.reduce_mean(ranking_loss)
    #     return loss
    return ranking_loss

def MTL_model(cnn_filter_num = 64):
    
    #Input layer
    
    char_input = Input(shape=(16,), dtype='int32', name = 'input10')
    entity_1_loc_input = Input(shape=(16,), dtype='int32', name = 'input11')
    entity_2_loc_input = Input(shape=(16,), dtype='int32', name = 'input12')
    tag_input = Input(shape=(16,), dtype='int32', name = 'input15')
    
    #embedding layer
    char_embedding = Embedding(len(tokenizer_api.word_index)+1, 200,
        input_length=16,#weights=[char_embedding_matrix],
        trainable=True)
    location_embedding = Embedding(150+1, 50, input_length = 16, 
                trainable=True, name = 'location_embedding')
    tag_embedding = Embedding(133, 50, input_length = 16, 
                trainable=True, name = 'tag_embedding')
    
    
    emb_char = char_embedding(char_input)
    emb_entity_1_loc = location_embedding(entity_1_loc_input)
    emb_entity_2_loc = location_embedding(entity_2_loc_input)
    emb_tag = tag_embedding(tag_input)
    
    
    emb_char = SpatialDropout1D(0.2)(emb_char)
    emb_entity_1_loc = SpatialDropout1D(0.2)(emb_entity_1_loc)
    emb_entity_2_loc = SpatialDropout1D(0.2)(emb_entity_2_loc)
    emb_tag = SpatialDropout1D(0.2)(emb_tag)
    
    merge_embedding = concatenate([emb_char, emb_entity_1_loc,                                   emb_entity_2_loc, emb_tag])
    
    #multi size CNN for emb_char
    
    kernel_sizes = [1, 2, 3, ]
    pooled_char = []
    pooled_char_mean = []
    
    for kernel in kernel_sizes:

        conv_char = Conv1D(filters=cnn_filter_num,
                      kernel_size=kernel,
                      padding='same',
                      strides=1,
                      kernel_initializer='he_uniform',
                      activation='relu')(merge_embedding)
        
        pool_char = MaxPooling1D(pool_size = 16)(conv_char)
        pool_char_2 = AvgPool1D(pool_size = 16)(conv_char)
        
        pooled_char.append(pool_char)
        pooled_char_mean.append(pool_char_2)
        
    merged_pooled_char = Concatenate(axis=-1)(pooled_char)
    flatten_pooled_char = Flatten()(merged_pooled_char)
    
    merged_pooled_char2 = Concatenate(axis=-1)(pooled_char_mean)
    flatten_pooled_char2 = Flatten()(merged_pooled_char2)
    
    merge_all = concatenate([flatten_pooled_char, flatten_pooled_char2])#rnn_output
    
    merge_all = BatchNormalization()(merge_all)
    merge_all = Dropout(0.5)(merge_all)
    merge_all = Dense(128, activation='relu')(merge_all)
    
    merge_all = BatchNormalization()(merge_all)
    merge_all = Dropout(0.2)(merge_all)
    
    pred1 = Dense(6, name = 'loss_1')(merge_all)
    
    pred2 = Dense(2, activation = 'softmax', name = 'loss_2')(merge_all)
    
    model = Model(inputs=[char_input, entity_1_loc_input, entity_2_loc_input,
    tag_input],outputs=[pred1, pred2])
    
    return model


y_binary = ['relation' if x != 'Other' else x for x in data['relation']]

LB_y_binary = LabelEncoder()
LB_y_binary.fit(y_binary)
y_binary = LB_y_binary.transform(y_binary)

y_binary = to_categorical(y_binary, 2)


# ##### split dataset

docs_id_list = list(data['docs_id'].unique())

docs_id_list = list(data['docs_id'].unique())

folds = [docs_id_list[:107], docs_id_list[107:214], docs_id_list[214:321], docs_id_list[321:428], docs_id_list[428:]]


# ##### train 3 times to reduce the randomness

acc_all =[]
for n in range(3):
    for i in range(5):
        gc.collect()
        K.clear_session()
        print('*******%s*******'%n)
        idx_train = np.array(data[~pd.DataFrame(data['docs_id'])['docs_id'].isin(folds[i])].index)
        idx_val = np.array(data[pd.DataFrame(data['docs_id'])['docs_id'].isin(folds[i])].index)
        
        X_train = get_input(data.loc[idx_train, :])
        y_train = y[idx_train]
        
        X_val = get_input(data.loc[idx_val, :])
        y_val = y_binary[idx_val]
        y_train_mask = y_mask[idx_train]
        y_val_mask = y_mask[idx_val]
        """
        Process y
        """
        y_train_6 =  np.delete(y[idx_train], 3, axis = 1)
        y_train_label_6 = []
        for x in y_train_6:
            if np.max(x) == 0:
                y_train_label_6 .append(8888)
            else:
                y_train_label_6 .append(np.argmax(x))
        y_train_label_6 = np.array(y_train_label_6)
        y_train_label_6 = np.expand_dims(y_train_label_6, axis=-1)
        y_train_ranking = np.concatenate([y_train_6, y_train_label_6, y_train_mask], axis=1)

        y_val_6 =  np.delete(y[idx_val], 3, axis = 1)
        y_val_label_6 = []
        for x in y_val_6:
            if np.max(x) == 0:
                y_val_label_6 .append(8888)
            else:
                y_val_label_6 .append(np.argmax(x))
        y_val_label_6 = np.array(y_val_label_6)
        y_val_label_6 = np.expand_dims(y_val_label_6, axis=-1)
        y_val_ranking = np.concatenate([y_val_6, y_val_label_6, y_val_mask], axis=1)

        bst_model_path =  str(i) + '_bestmodel.hdf5'

        gc.collect()
        K.clear_session()

        model = MTL_model()
        model.compile(loss={'loss_1':ranking_loss, 'loss_2':'categorical_crossentropy'},
            optimizer='RMSprop',
            loss_weights={'loss_1':1, 'loss_2':1})
            #metrics=[ranking_loss])

        early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode = 'min')
        model_checkpoint = ModelCheckpoint(bst_model_path, monitor = 'val_loss', 
             save_best_only=True, save_weights_only=True, verbose=1,  mode = 'min')
        callbacks = [
                    early_stopping,
                    model_checkpoint
                ]

        hist = model.fit(X_train, {'loss_1':y_train_ranking, 'loss_2':y_train},verbose=True,            validation_data=(X_val, {'loss_1':y_val_ranking, 'loss_2':y_val}),             epochs=50, batch_size=256, shuffle=True, callbacks=callbacks)#callbacks=callbacks, 
    #             del model

        #  test for f1
    #     p1 = model.predict(X_val)
    #     p1 = pd.DataFrame(p1)

    #     result = pd.DataFrame(LB_y.inverse_transform(np.argmax(np.array(p1),axis = 1)))
    #     true = pd.DataFrame(LB_y.inverse_transform(np.argmax(np.array(y_val),axis = 1)))

        label = list(data['relation'].unique())
        label.remove('Other')

    #         model = MTL_model()
    #         model.load_weights(bst_model_path)

        model.load_weights(bst_model_path)

        p1 = model.predict(X_val)[0]

        true_p1 = np.insert(np.array(p1), 3, values=0.5, axis=1)
        true_result = []
        for x in true_p1:
            max_score = np.max(x)
            max_arg = np.argmax(x)
            true_result.append(max_arg)
        true_result =np.array(true_result)

        true_true =np.array(np.argmax(y[idx_val], axis = 1))


        result = pd.DataFrame(LB_y.inverse_transform(true_result))
        true = pd.DataFrame(LB_y.inverse_transform(true_true))

        f1_result = classification_report(list(true[0]),list(result[0]), labels = label, digits = 4)
        print(f1_result)

        gc.collect()




