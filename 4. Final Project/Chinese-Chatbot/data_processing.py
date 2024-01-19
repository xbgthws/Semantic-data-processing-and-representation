import re
import os
import pickle
import jieba
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_data(file_path, num_samples=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            if num_samples is None:
                samples = file.readlines()
            else:
                samples = file.readlines()[:num_samples]

            word_pairs = [[clean_text(text) for text in line.split("\t")] for line in samples]

            return zip(*word_pairs)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return []


def clean_text(text):
    text = text.strip()
    text = " ".join(text)
    text = re.sub(r'\s+', ' ', text)
    symbols = {
        r'…{1,100}': '…',
        r'\.{3,100}': '…',
        r'···{2,100}': '…',
        r',{1,100}': '，',
        r'\.{1,100}': '。',
        r'。{1,100}': '。',
        r'\?{1,100}': '？',
        r'？{1,100}': '？',
        r'!{1,100}': '！',
        r'！{1,100}': '！',
        r'~{1,100}': '～',
        r'～{1,100}': '～',
        r'[“”]{1,100}': '"',
        r'[^\s\w\u4e00-\u9fff"。，？！～·]+': '',
        r'[ˇˊˋˍεπのゞェーω]': ''
    }
    for pattern, repl in symbols.items():
        text = re.sub(pattern, repl, text)

    return text


def build_vocab(text_lst, vocab_file="./data/vocab.pkl"):
    if os.path.exists(vocab_file):
        with open(vocab_file, 'rb') as file:
            vocab_data = pickle.load(file)
            vocab_size = vocab_data['vocab_size']
            word2idx = vocab_data['word2idx']
            idx2word = vocab_data['idx2word']
    else:
        vocab = set()
        for seq in text_lst:
            vocab.update(seq.split())
        vocab = sorted(vocab)
        # Add special markers
        vocab.insert(0, '<pad>')
        vocab.insert(1, '<unk>')
        vocab.insert(2, '<start>')
        vocab.insert(3, '<end>')
        vocab_size = len(vocab) + 1
        idx2word = dict(enumerate(vocab))
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        # Save the dictionary
        vocab_data = {
            'vocab_size': vocab_size,
            'word2idx': word2idx,
            'idx2word': idx2word
        }
        with open(vocab_file, 'wb') as file:
            pickle.dump(vocab_data, file)

    return vocab_size, word2idx, idx2word


def tokenize(inp_text, tar_text):
    inp_res = []
    tar_res = []
    for inp, tar in zip(inp_text, tar_text):
        inp_res.append(" ".join(jieba.cut(inp)))
        tar_res.append(" ".join(jieba.cut(tar)))

    vocab_size, word2idx, _ = build_vocab(inp_res + tar_res)

    inp_seq = [[word2idx.get(word, word2idx['<unk>']) for word in seq.split()] for seq in inp_res]
    tar_seq = [[word2idx.get(word, word2idx['<unk>']) for word in seq.split()] for seq in tar_res]

    max_inp_len = max(len(seq) for seq in inp_seq)
    max_tar_len = max(len(seq) for seq in tar_seq)
    # padding the sequences
    inp_seq = keras.preprocessing.sequence.pad_sequences(inp_seq, maxlen=max_inp_len, padding='post')
    tar_seq = keras.preprocessing.sequence.pad_sequences(tar_seq, maxlen=max_tar_len, padding='post')
    # Add a beginning and an end token to the target sequence
    inp_seq = np.hstack((np.ones((inp_seq.shape[0], 1)) * word2idx['<start>'], inp_seq))
    tar_seq = np.hstack((np.ones((tar_seq.shape[0], 1)) * word2idx['<start>'], tar_seq))
    inp_seq = np.hstack((inp_seq, np.ones((inp_seq.shape[0], 1)) * word2idx['<end>']))
    tar_seq = np.hstack((tar_seq, np.ones((tar_seq.shape[0], 1)) * word2idx['<end>']))
    max_inp_len += 2
    max_tar_len += 2

    return inp_seq, tar_seq, vocab_size, max_inp_len, max_tar_len


class DataGenerator(keras.utils.Sequence):

    def __init__(self, tokenizer_data, batch_size, shuffle=True):
        self.indices = None
        self.inp_seq, self.tar_seq = tokenizer_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.inp_seq) // self.batch_size

    def __getitem__(self, index):
        # Generate indexes of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_inp_seq = [self.inp_seq[i] for i in batch_indices]
        batch_tar_seq = [self.tar_seq[i] for i in batch_indices]
        # transform to tensor
        batch_inp_seq = tf.convert_to_tensor(batch_inp_seq)
        batch_tar_seq = tf.convert_to_tensor(batch_tar_seq)

        return batch_inp_seq, batch_tar_seq

    def on_epoch_end(self):
        self.indices = np.arange(len(self.inp_seq))
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == '__main__':
    inp_text, tar_text = load_data(file_path="./weibo.tsv", num_samples=10)
    inp_seq, tar_seq, vocab_size, _, _ = tokenize(inp_text, tar_text)
    print(f"test the clean_text func: {clean_text('怎么证明	你问～我爱～你有～多深～我爱～你有～几～分～～～')}")
    print("-" * 100)
    print(inp_text)
    print('*'*100)
    print(tar_text)
    # print("-" * 100)
    # print(inp_seq, tar_seq)
    print("-" * 100)
    dataset = DataGenerator(tokenizer_data=(inp_seq, tar_seq), batch_size=64)

    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        # for batch, (inp_seq, tar_seq) in enumerate(dataset):
            # print(f"Batch {batch + 1}/{len(dataset)}")
            # print(inp_seq.shape, tar_seq.shape)
            # print(inp_seq, tar_seq)
            # print("-" * 100)

    print(f"vocab_size: {vocab_size}")
