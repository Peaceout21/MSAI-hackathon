import os,re
os.chdir('/home/ai/arjun/msai/')
import numpy as np
import h5py
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers import multiply

def load_embedding():
    global embeddings_index
    embeddings_index = dict()
    f = open('glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
load_embedding()


f = open('data.tsv',"r",encoding="utf-8",errors="ignore")
query_list_train,query_id_train,passage_list_train,labels_train,passage_id_train=[],[],[],[],[]
for i,line in enumerate(f):
#     if i<21:
#         print(line)
# #         next
        tokens = line.strip().lower().split("\t")
        qid,query,passage,label,passageid = tokens[0],tokens[1],tokens[2],tokens[3],tokens[4]
        words = re.split('\W+', query)
        words = ' '.join([re.sub(r"[^a-zA-Z.!?]+", r" ",x) for x in words if x])# to remove empty words
#         print(query,words)
        query_list_train.append(words)
        query_id_train.append(qid)
        passage_id_train.append(passageid)
        words = re.split('\W+', passage)
        words = ' '.join([re.sub(r"[^a-zA-Z.!?]+", r" ",x) for x in words if x])# to remove empty words
        passage_list_train.append(words)
        labels_train.append(label)
        
#         if label=='0':
#             labels.append(1)
#         else:
#             labels.append(0)

f=open('eval1_unlabelled.tsv',"r",encoding="utf-8",errors="ignore")
query_list_eval,query_id_eval,passage_list_eval,passage_id_eval=[],[],[],[]
for i,line in enumerate(f):
#     if i<21:
#         print(line)
#         next
        tokens = line.strip().lower().split("\t")
        qid,query,passage,passageid = tokens[0],tokens[1],tokens[2],tokens[3]
        words = re.split('\W+', query)
        words = ' '.join([re.sub(r"[^a-zA-Z.!?]+", r" ",x) for x in words if x])# to remove empty words
#         print(query,words)
        query_list_eval.append(words)
        query_id_eval.append(qid)
        words = re.split('\W+', passage)
        words = ' '.join([re.sub(r"[^a-zA-Z.!?]+", r" ",x) for x in words if x])# to remove empty words
        passage_list_eval.append(passage)
        passage_id_eval.append(passageid)
        
        
def prepare_queryPassage_vectors(query_list,passage_list,prepare_query_embedding,prepare_passage_embedding):
    
    global embedding_matrix_query,embedding_matrix_passage,padded_query_docs,padded_passage_docs,vocab_query,vocab_passage
    max_query_words = 12
    max_passage_words = 100
    emb_dim = 100
    '''
    prepares padded doc and glove embedding matrix for  query
    '''
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(query_list)
    vocab_query = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(query_list)
#     print(encoded_docs)
    # pad documents to a max length of 4 words
    padded_query_docs = pad_sequences(encoded_docs, maxlen=max_query_words, padding='post')
    del t
    print('prepared padded_query_docs')
    # create a weight matrix for words in training docs
    if prepare_query_embedding:
        all_query_list=query_list_train+query_list_eval
        t = Tokenizer()
        t.fit_on_texts(all_query_list)
        vocab_query = len(t.word_index) + 1
        # integer encode the documents
        encoded_docs = t.texts_to_sequences(all_query_list)
        embedding_matrix_query = np.zeros((vocab_query, emb_dim))
        for word, i in t.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix_query[i] = embedding_vector
         
    '''
    prepares padded doc and glove embedding matrix for  query
    '''
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(passage_list)
    vocab_passage = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(passage_list)
#     print(encoded_docs)
#     pad documents to a max length of 50 words
    padded_passage_docs = pad_sequences(encoded_docs, maxlen=100, padding='post')
    print('prepared padded_passage_docs')
    # create a weight matrix for words in training docs
    if prepare_passage_embedding:
        all_passage_list=passage_list_train+passage_list_eval
        embedding_matrix_passage = np.zeros((vocab_passage, emb_dim))
        for word, i in t.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix_passage[i] = embedding_vector  
        print('loaded passage vectors')


prepare_queryPassage_vectors(query_list=query_list_train,passage_list=passage_list_train,prepare_query_embedding=True,prepare_passage_embedding=True)


def write_op_padded():
    with h5py.File('passage_padded_train.h5', 'w') as hf:
        hf.create_dataset("passage_padded_train",data=padded_passage_docs,compression="gzip")
    with h5py.File('query_padded_train.h5', 'w') as hf:
        hf.create_dataset("query_padded_train",data=padded_query_docs,compression="gzip")

def write_op_embedding():
    with h5py.File('passage_embeddings.h5', 'w') as hf:
        hf.create_dataset("passage_embeddings",  data=embedding_matrix_passage,compression='gzip')
        
    with h5py.File('query_embeddings.h5', 'w') as hf:
        hf.create_dataset("query_embeddings",  data=embedding_matrix_query,compression='gzip')
        
        
def read_op():
    global embedding_matrix_passage,embedding_matrix_query
    with h5py.File('passage_embeddings.h5', 'r') as hf:
        embedding_matrix_passage = hf.get('passage_embeddings').value
    with h5py.File('query_embeddings.h5', 'r') as hf:
        embedding_matrix_query = hf.get('query_embeddings').value    
    print(" loaded embeddings")
    
def read_pad():
    global padded_query_docs,padded_passage_docs
    with h5py.File('passage_padded.h5', 'r') as hf:
        padded_passage_docs = hf.get('passage_padded').value
    with h5py.File('query_padded.h5', 'r') as hf:
        padded_query_docs = hf.get('query_padded').value    
    print(" loaded padded docs")    
read_pad()

def create_pad_eval():
    global paddeddocs_query_eval,paddeddocs_passage_eval
    t = Tokenizer()
    t.fit_on_texts(query_list_eval)
    vocab_query = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(query_list_eval)
#     print(encoded_docs)
    # pad documents to a max length of 4 words
    paddeddocs_query_eval = pad_sequences(encoded_docs, maxlen=12, padding='post')
    
    t = Tokenizer()
    t.fit_on_texts(passage_list_eval)
    vocab_query = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(passage_list_eval)
#     print(encoded_docs)
    # pad documents to a max length of 4 words
    paddeddocs_passage_eval = pad_sequences(encoded_docs, maxlen=100, padding='post')
    
create_pad_eval()


embedding_layerA= Embedding(input_dim=embedding_matrix_query.shape[0],
                            output_dim=embedding_matrix_query.shape[1], 
                            input_length=12,
                            weights=[embedding_matrix_query], 
                            trainable=False, 
                            name='embedding_layer_qry')

input_query = Input(shape=(12,), dtype='int32', name='input_query')
x = embedding_layerA(input_query)
lstm_op_query=LSTM(100,name='lstm_op_query')(x)
dense_from_query=Dense(2, activation='sigmoid',name='dense_from_query')(lstm_op_query)


embedding_layerB= Embedding(input_dim=embedding_matrix_passage.shape[0],
                            output_dim=embedding_matrix_passage.shape[1], 
                            input_length=100,
                            weights=[embedding_matrix_passage], 
                            trainable=False, 
                            name='embedding_layer_passg')

input_passage = Input(shape=(100,), dtype='int32', name='input_passage')
y = embedding_layerB(input_passage)
lstm_op_passage=LSTM(100,name='lstm_op_passage')(y)
dense_from_passage=Dense(2, activation='sigmoid',name='dense_from_passage')(lstm_op_passage)

dense_mul=multiply([dense_from_query,dense_from_passage])
combined_dense=Dense(1,activation='sigmoid',name='combined_dense')(dense_mul)

# model=Model(inputs=[input_query,input_passage],outputs=combined_dense)
model=Model(inputs=[input_query,input_passage],outputs=combined_dense)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# type(labels)
labels=np.asarray(labels_train)
# And trained it via:
model.fit({'input_query': padded_query_docs, 'input_passage': padded_passage_docs},
          {'dense_from_query': labels, 'dense_from_passage': labels,'combined_dense':labels},
          epochs=2, batch_size=256)

paddeddocs_query_eval.shape, paddeddocs_query_eval.shape

model.predict(x=[paddeddocs_query_eval,paddeddocs_passage_eval])
