import markovify
import jieba
import re


def has_repeated_words(sentence):
    # 使用正则查找两个相同单词并且相邻的模式，比如”总得总得
    pattern = r'(\b\w+\b)\s+\1'
    match = re.search(pattern, sentence)
    # 使用正则查找三个连续字符的模式,比如“我我我”
    pattern_2  = r'(\b\w\b)\1\1'
    match_2 = re.search(pattern_2, sentence)
    # 如果找到匹配的话，返回True，否则返回False
    if match:
        return True
    if match_2:
        return True
    return False

# 检查句子中是否包含特殊字符以及连续重复的字
def check_the_sentence(sentence):
    # 检查句子中是否包含特殊字符
    for ch in sentence:
        if ch in 'ヽ○^㉨^ﾉ▽★*☆':
            return False
    # 检查句子长度是否合适
    if len(sentence) < 10:
        return False
    if has_repeated_words(sentence):
        return False
    return True


def generate(num_sentences, cut_strategy='precise'):
    # 逐行读取文本
    with open("randomtext.txt", encoding="utf-8") as f:
        text = f.read()
    # 使用jieba进行分词,采用不同的分词策略
    if cut_strategy == 'precise':
        text = " ".join(jieba.cut(text, cut_all=False))
    if cut_strategy == 'cut_all':
        text = " ".join(jieba.cut(text, cut_all=True))
    if cut_strategy == 'HMM':
        text = " ".join(jieba.cut(text, HMM=True))
    # 使用Markovify创建一个文本模型
    text_model = markovify.NewlineText(text, state_size=3)
    for i in range(num_sentences):
        sentence = text_model.make_short_sentence(100)
        if check_the_sentence(sentence):
            print("".join(sentence.split()))
            print(len(sentence))
        else:
            # 如果句子不合适，打印出来并且标记为bad sentence
            print("".join(sentence.split()) + '(bad sentence)')


if __name__ == '__main__':
    print('-------------use precise_cut mode-------------')
    generate(5)
    print('-------------use cut_all mode-------------')
    generate(5, 'cut_all')
    print('-------------use HMM mode-------------')
    generate(5, 'HMM')
