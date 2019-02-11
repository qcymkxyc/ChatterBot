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


def build_seq2seq(input_dim, encoder_units, decoder_units,
                  hidden_units, batch_size):
    """创建Seq2Seq模型

    :param input_dim: int
        输入维度
    :param encoder_units: List[int]
        编码器
    :param decoder_units: List[int]
        解码器
    :param hidden_units: List[int]
        解码器的隐藏层
    :return: Model
        Seq2Seq模型
    """
    assert encoder_units == decoder_units, "编码器和解码器必须一致"

    # 编码器
    encoder_input = keras.layers.Input(
        shape=[None, input_dim], name="Encoder_Input")
    encoder_last_output = encoder_input

    encoder_state_list = list()
    for i, encoder_unit in enumerate(encoder_units):
        encoder_net = keras.layers.CuDNNLSTM(
            units=encoder_unit,
            return_sequences=True,
            return_state=True,
            name="Encoder_LSTM_{}".format(i + 1)
        )
        encoder_net_output, state_h, state_c = encoder_net(encoder_last_output)
        encoder_last_output = encoder_net_output
        encoder_state_list.append((state_h, state_c))

    # 解码器
    decoder_input = keras.layers.Input(
        shape=[None, input_dim],
        batch_size=batch_size,
        name="Decoder_Input")
    decoder_last_output = decoder_input
    for i, decoder_unit in enumerate(decoder_units):
        decoder_net = keras.layers.CuDNNLSTM(
            units=decoder_unit,
            stateful=True,
            return_sequences=True,
            return_state=True,
            name="Decoder_LSTM_{}".format(i + 1)
        )
        init_state = encoder_state_list.pop(0)
        decoder_output, state_h, state_c = decoder_net(
            decoder_last_output, initial_state=init_state)
        decoder_last_output = decoder_output

    # 解码器的隐藏层
    hidden_net = decoder_output
    for i, hidden_unit in enumerate(hidden_units):
        hidden_net = keras.layers.Dense(
            units=hidden_unit,
            activation="relu",
            name="Decoder_hidden_{}".format(i + 1)
        )(hidden_net)

    # 输出
    output = keras.layers.Dense(input_dim, activation="softmax")(hidden_net)

    model = keras.Model([encoder_input, decoder_input], output)
    return model
