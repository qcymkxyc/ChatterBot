# -*- coding:utf-8 _*-
"""
 @author:qcymkxyc
 @email:qcymkxyc@163.com
 @software: PyCharm
 @file: seq2seq_model.py
 @time: 2019/2/9 16:22

    seq2seq模型
"""
from tensorflow import keras


def build_seq2seq():
    # 编码器
    encoder_input = keras.layers.Input(shape=[100, 10])
    encoder_output, state_h, state_c = keras.layers.LSTM(
        128, return_state=True)(encoder_input)
    encoder_state = (state_h, state_c)
    # encoder_output, encoder_state = keras.layers.LSTM(128)(encoder_input)

    # 解码器
    decoder_input = keras.layers.Input(shape=[100, 10], batch_size=10)
    decoder_net,_,_ = keras.layers.LSTM(
        128, stateful=True, return_sequences=True, return_state=True)(
        decoder_input, initial_state=encoder_state)
    decoder_net = keras.layers.Dense(10, activation="softmax")(decoder_net)

    model = keras.Model([encoder_input, decoder_input], decoder_net)
    return model
