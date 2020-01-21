#! -*- coding: utf-8 -*-
import keras
from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs
from keras_bert import AdamWarmup, calc_train_steps
from keras.utils import multi_gpu_model
import glob
import random
import re
import os
import numpy as np
batch_size = 32
path_vocab = '/search/odin/guobk/streaming/vpa/bert4keras/pretraining/vocab.txt'
path_data = '/search/odin/guobk/streaming/vpa/data/data_inputs/*/*-seg/sents.txt'
path_save = 'model_kerasBert/model.h5'
head_num=8
transformer_num=12
embed_dim=128
feed_forward_dim=100
seq_len=20
pos_num=20
dropout_rate=0.05
gpus = 4
steps_per_epoch = 10000
epochs = 10000
validation_steps = 100
def getVocab(path_vocab):
    # Build token dictionary
    token_dict = get_base_dict()  # A dict that contains some special tokens
    with open(path_vocab,'r',encoding='utf-8') as f:
        words = f.read().strip().split('\n')
    for token in words:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
    token_list = list(token_dict.keys())  # Used for selecting a random word
    return token_dict,token_list
def build_model(token_dict):
    # Build & train the model
    with tf.device('/cpu:0'):
        model = get_model(
            token_num=len(token_dict),
            head_num=head_num,
            transformer_num=transformer_num,
            embed_dim=embed_dim,
            feed_forward_dim=feed_forward_dim,
            seq_len=seq_len,
            pos_num=pos_num,
            dropout_rate=0.05,
        )
        compile_model(model)
        model.summary()
    return model
#optimizer = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)
def get_sentence_pairs(files,batch_size,sym_sents = ',.!?;，。！？；',min_seqlen=4):
    pattern = u'.*?[\n' + sym_sents + ']+'
    S = []
    while True:
        for file in files:
            f = open(file,'r')
            for line in f:
                s = line.strip()
                sents = re.findall(pattern, s)
                if len(sents)<1:
                    continue
                if len(sents)==1:
                    A = sents[0].split(' ')
                    if len(A)<min_seqlen:
                        continue
                    if np.random.uniform(0,1)>0.05:
                        continue
                    B = [a for a in A]
                else:
                    idx = [i for i in range(len(sents)-1)]
                    i = random.sample(idx,1)[0]
                    A = sents[i].split(' ')
                    B = sents[i+1].split(' ')
                    if len(A)<min_seqlen or len(B)<min_seqlen:
                        continue
                if np.random.uniform(0,1)>0.3:
                    continue
                S.append([A,B])
                if len(S)>=batch_size:
                    f.close()
                    return S
            f.close()
def _generator(batch_size,token_dict,token_list):
    while True:
        filenames = glob.glob(path_data)
        random.shuffle(filenames)
        sentence_pairs = get_sentence_pairs(filenames,batch_size)
        yield gen_batch_inputs(
            sentence_pairs,
            token_dict,
            token_list,
            seq_len=20,
            mask_rate=0.3,
            swap_sentence_rate=1.0,
        )
def _generator_val(batch_size,token_dict,token_list):
    while True:
        filenames = glob.glob(path_data)
        random.shuffle(filenames)
        sentence_pairs = get_sentence_pairs(filenames,batch_size)
        yield gen_batch_inputs(
            sentence_pairs,
            token_dict,
            token_list,
            seq_len=20,
            mask_rate=0.3,
            swap_sentence_rate=1.0,
        )
def get_multiGPU_model(model):
    model_multi = multi_gpu_model(model, gpus=gpus)
    compile_model(model_multi)
    if os.path.exists(path_save):
        model_multi.load_weights(path_save)
        print('model initialized from pretrained model')
    else:
        print('model initialized randomly')
    return model_multi
def main():
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=path_save, monitor='val_loss',
                                     verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    token_dict,token_list = getVocab(path_vocab)
    model = build_model(token_dict)
    model_multi = get_multiGPU_model(model)
    model_multi.fit_generator(
        generator=_generator(batch_size,token_dict,token_list),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=_generator_val(batch_size,token_dict,token_list),
        validation_steps=validation_steps,
        callbacks=[checkpoint, earlyStopping],
    )
    iter = _generator(batch_size, token_dict, token_list)
    model_multi.fit()
def test():
    import tensorflow as tf
    from keras.applications import Xception
    from keras.utils import multi_gpu_model
    from keras.layers import Input, Dense
    from keras.models import Model
    import numpy as np
    num_samples = 1000
    height = 80
    width = 80
    num_classes = 1000
    # 实例化基础模型（或者「模版」模型）。
    # 我们推荐在 CPU 设备范围内做此操作，
    # 这样模型的权重就会存储在 CPU 内存中。
    # 否则它们会存储在 GPU 上，而完全被共享。
    with tf.device('/cpu:0'):
        inputs = Input((1024,))
        x = Dense(num_classes, activation='relu')(inputs)
        x = Dense(num_classes, activation='relu')(x)
        x = Dense(num_classes, activation='relu')(x)
        x = Dense(num_classes, activation='relu')(x)
        outputs =Dense(num_classes, activation='relu')(x)
        model = Model(inputs=inputs,outputs=outputs)
    # 复制模型到 8 个 GPU 上。
    # 这假设你的机器有 8 个可用 GPU。
    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop')
    from keras_bert import get_base_dict, get_model, compile_model
    compile_model(parallel_model)
    # 生成虚拟数据
    x = np.random.random((num_samples*1000, 1024))
    y = np.random.random((num_samples*1000, num_classes))
    # 这个 `fit` 调用将分布在 8 个 GPU 上。
    # 由于 batch size 是 256, 每个 GPU 将处理 32 个样本。
    parallel_model.fit(x, y, epochs=20, batch_size=256)
if __name__=='__main__':
    main()
