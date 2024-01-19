import numpy as np
import tensorflow as tf
from tensorflow import keras


class PositionalEncoding(keras.layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def build(self, input_shape):
        seq_len = input_shape[1]
        d_model = input_shape[2]
        self.positional_encoding = self.calculate_positional_encoding(seq_len, d_model)

    def calculate_positional_encoding(self, seq_len, d_model):
        positional_encoding = np.zeros((seq_len, d_model))
        angles = np.arange(seq_len)[:, np.newaxis] / np.power(10000, np.arange(d_model)[np.newaxis, :] / d_model)
        positional_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        positional_encoding[:, 1::2] = np.cos(angles[:, 1::2])

        return tf.cast(positional_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]


def create_padding_mask(seq):
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(seq):
    seq_len = tf.shape(seq)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    return look_ahead_mask


def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # calculate the dot product of q and k
    dk = tf.cast(tf.shape(k)[-1], tf.float32)  # get the dimension of k
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # add the mask to the scaled tensor
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1  # calculate the attention weights
    )
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)
        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # split x into multiple headers
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, k, v, q, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights


class PointwiseFeedForward(keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(PointwiseFeedForward, self).__init__()
        self.dense1 = keras.layers.Dense(dff, activation="relu")
        self.dense2 = keras.layers.Dense(d_model)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)

        return x


class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointwiseFeedForward(d_model, dff)
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training, mask=None):
        attn_output, _ = self.multihead_attention(inputs, inputs, inputs, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model  # dimension of a word vector
        self.num_layers = num_layers  # Number of encoder layers
        self.maximum_position_encoding = maximum_position_encoding  # Maximum position code
        self.embedding = keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(
            seq_len, self.maximum_position_encoding,
            "seq_len should be less than or equal to self.maximum_position_encoding"
        )
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x


class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.multihead_attention1 = MultiHeadAttention(d_model, num_heads)
        self.multihead_attention2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointwiseFeedForward(d_model, dff)
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

    def call(self, inputs, enc_output, training,
             look_ahead_mask=None, padding_mask=None):
        # Decoder Self-Attention Layer
        attn1, attn_weights_block1 = self.multihead_attention1(inputs, inputs, inputs, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)  # dropout
        out1 = self.layernorm1(attn1 + inputs)
        # Encoder-Decoder Attention Layer
        attn2, attn_weights_block2 = self.multihead_attention2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)  # dropout
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)  # dropout
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.maximum_position_encoding = maximum_position_encoding
        self.embedding = keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(
            seq_len, self.maximum_position_encoding,
            "seq_len should be less than or equal to self.maximum_position_encoding"
        )
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        return x, attention_weights


class Transformer(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                 pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads,
                               dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads,
                               dff, target_vocab_size, pe_target, rate)
        self.final_layer = keras.layers.Dense(target_vocab_size)

    def call(self, en_inputs, de_inputs, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(en_inputs, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(
            de_inputs, enc_output, training, look_ahead_mask, dec_padding_mask
        )
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights


