# -*- coding:utf-8 -*-
# author:xxc
# 本程序是使用最大熵模型进行命名实体的识别
import nltk
import re
import pickle
import time
from preprocess.utils import file_find

BASIC_PATH = "tweet_ner/en-basic"

class MaxEntropy():

    def __init__(self):
        self.train_set = []
        self.sent = []
        self.classifier = None

        self._en_wordlist = []
        with open(file_find(BASIC_PATH), "r") as fr:
            for word in fr.readlines():
                self._en_wordlist.append(word)

    def load_model(self, modelName):
        with open(modelName, 'rb') as f:
            self.classifier = pickle.load(f)

    def get_data(self, filename):
        """
        将文件处理成需要的输入格式
        :param filename: 文件名
        :return:sents,tags
        """
        tag = []
        tags = []
        sent = []
        sents = []
        f = open(filename, 'r')
        for s in f.readlines():
            tup = []
            if s == '\n':                   # 如果遇到换行就保存一句话
                sents.append(sent)          # 将句子添加到句子集合里
                tags.append(tag)            # 将标签添加到命名实体标签集合里
                tag = []
                sent = []
                continue
            tag.append(s.split('\t')[2].strip())
            tup.append(s.split('\t')[0])
            tup.append(s.split('\t')[1])
            sent.append(tuple(tup))         # 句子列表添加元组
        f.close()
        return sents, tags

    def get_train_feature(self, filename):
        """
        获取特征
        :param filename:训练语料
        :return:返回训练集
        """
        (sents, tags) = self.get_data(filename)
        for index, sent in enumerate(sents):  # 取句子集合里的句子
            history = []
            for i in range(len(sent)):
                self.featureset = self.generate_feature(tokens=sent, index=i, history=history)  # 生成句子中某个词特征集
                self.train_set.append((self.featureset, tags[index][i]))    # 将特征集与标签添加到训练集里
                history.append(tags[index][i])
        return self.train_set

    def file_classify(self, filename, modelName):
        """
        进行命名实体识别
        :param filename: 传入的要识别的文件名
        :return:history,tags预测标签集，labels全部标签名
        """
        print ('Classifying ...')
        start_time = time.time()
        (sents, tags) = self.get_data(filename)     # 处理输入数据
        over_history = []
        with open(modelName, 'rb') as f:     # 加载模型
            classifier = pickle.load(f)
        for index, sent in enumerate(sents):                # 获取句子集中的某个句子和对应序号
            history = []            # 历史命名实体标注集合
            for i in range(len(sent)):
                self.featureset = self.generate_feature(tokens=sent, index=i, history=history)  # 生成特征集
                pre_tag = classifier.classify(self.featureset)  # 进行命名实体标签预测
                history.append(pre_tag)
                over_history.append(pre_tag)
        labels = classifier.labels()    # 得到分类器的所有命名实体标签名
        end_time = time.time()
        print ('Classifying finished! cost %d seconds' % (end_time - start_time))

        return over_history, tags, labels


    def sent_classify(self, token):
        """
        进行输入句子的实体识别
        :param token 为词性标注好的词组[(token_1, pos_1)...(token_2, pos_2)]
        :return: [(token_1, pos_1, ner_1)...(token_n, pos_n, ner_n)]
        """
        if self.classifier == None:
            print "使用错误，请先加载分类模型！"
            return

        history = []                          # 历史命名实体标注
        sentence = []
        for i in range(len(token)):
                featureset = self.generate_feature(tokens=token, index=i, history=history)   # 获取句子特征
                pre_tag = self.classifier.classify(featureset)
                history.append(pre_tag)      # 添加标注到历史命名实体标注集里
                sentence.append(token[i] + tuple([pre_tag]))
        return sentence

    def generate_feature(self, tokens, index, history):
        """
        生成特征
        :param tokens: 传入的带词性识别的句子
        :param index: 词序号
        :param history: 前词的标注历史
        :return:
        """

        word = tokens[index][0]     # 获取当前词
        pos = self.simplify_pos(tokens[index][1])   # 简化当前词的词性标注
        if index == 0:                  # 进行词位置判断，是否是首词，第二个词，倒数最后一个词，倒数第二个词
            prevword = prevprevword = None
            prevpos = prevprevpos = None
            prevshape = prevtag = prevprevtag = None
        elif index == 1:
            prevword = tokens[index-1][0].lower()
            prevprevword = None
            prevpos = self.simplify_pos(tokens[index-1][1])
            prevprevpos = None
            prevtag = history[index-1][0]
            prevshape = prevprevtag = None
        else:
            prevword = tokens[index-1][0].lower()
            prevprevword = tokens[index-2][0].lower()
            prevpos = self.simplify_pos(tokens[index-1][1])
            prevprevpos = self.simplify_pos(tokens[index-2][1])
            prevtag = history[index-1]
            prevprevtag = history[index-2]
            prevshape = self.shape(prevword)
        if index == len(tokens)-1:
            nextword = nextnextword = None
            nextpos = nextnextpos = None
        elif index == len(tokens)-2:
            nextword = tokens[index+1][0].lower()
            nextpos = tokens[index+1][1].lower()
            nextnextword = None
            nextnextpos = None
        else:
            nextword = tokens[index+1][0].lower()
            nextpos = tokens[index+1][1].lower()
            nextnextword = tokens[index+2][0].lower()
            nextnextpos = tokens[index+2][1].lower()

        features = {            # 特征集合
            'bias': True,
            'shape': self.shape(word),
            'wordlen': len(word),
            'prefix3': word[:3].lower(),
            'suffix3': word[-3:].lower(),
            'pos': pos,
            'word': word,
            'en-wordlist': (word in self._en_wordlist),
            'prevtag': prevtag,
            'prevpos': prevpos,
            'nextpos': nextpos,
            'prevword': prevword,
            'nextword': nextword,
            'word+nextpos': '%s+%s' % (word.lower(), nextpos),
            'pos+prevtag': '%s+%s' % (pos, prevtag),
            'shape+prevtag': '%s+%s' % (prevshape, prevtag),
            }

        return features

    def train(self, input_filename, output_modelName):
        """
        训练模型
        :param filename:文件名
        :return: 训练后的模型
        """
        print('Training classifier...')
        start_time = time.time()
        train_token = self.get_train_feature(input_filename)        # 处理训练文本
        classifier = nltk.MaxentClassifier.train(train_token, algorithm='GIS')      # 调用最大熵分类器的训练接口，'GIS'是generalized iterative scaling通用迭代算法
        with open(output_modelName, 'wb') as f:         # 保存模型
            pickle.dump(classifier, f, -1)
        end_time = time.time()
        print ('Training finished!  cost %d seconds' % (end_time - start_time))
        return classifier


    def simplify_pos(self, s):
        """
        简化标注，如果标注是以V开头的，就返回V
        :param s:
        :return:
        """
        if s.startswith('V'): return "V"
        else: return s.split('-')[0]

    def shape(self, word):
        """
        获取词型
        :param word:
        :return:词的形状（是否包含数字/首字母大写/包含符号/大小写混合）
        """
        if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word, re.UNICODE):
            return 'number'
        elif re.match('\W+$', word, re.UNICODE):
            return 'punct'
        elif re.match('\w+$', word, re.UNICODE):
            if word.istitle():
                return 'upcase'
            elif word.islower():
                return 'downcase'
            else:
                return 'mixedcase'
        else:
            return 'other'


    def Accuracy(self, pre_tag, true_tag, labels):
        """
        对分类结果进行评估
        :param pre_tag:预测的标签
        :param true_tag: 真实的标签
        :param labels: 所有标签
        :return:
        """
        acc_count = {}
        pre_count = {}
        tru_count = {}
        over_cor = 0.0      # 总体评估值，over_cor总的正确标签数，over_pre总的预测标签数，over_tru总的真实标签数
        over_pre = 0.0
        over_tru = 0.0
        print ('   标签         正确率         召回率         F1值         正确数')
        print ('   --------------------------------------------------------------')
        for label in labels:            # 对各标签种类进行赋值，acc_count是正确的标签数，pre_count是预测的标签数，tru_count是真实的标签数
            acc_count[label] = 0.0
            pre_count[label] = 0.0
            tru_count[label] = 0.0
        for pre, true in zip(pre_tag, true_tag):    # 将预测标签pre_tag和真实标签true_tag整合在一起
            if pre == 'O':          # 跳过为O的标签
                continue
            if pre == true:
                acc_count[pre] += 1
                over_cor += 1
        for label in pre_tag:
            if label == 'O':
                continue
            pre_count[label] += 1
            over_pre += 1
        for label in true_tag:
            if label == 'O':
                continue
            tru_count[label] += 1
            over_tru += 1
        for label in labels:
            if pre_count[label] == 0:
                continue
            accuracy = acc_count[label]/pre_count[label]            # 计算正确率
            recall = acc_count[label]/tru_count[label]              # 计算召回率
            F1 = accuracy*recall*2/(accuracy + recall)              # 计算F1值
            print ('   %6.6s        %.3f          %.3f        %.3f        %d' % (label, accuracy, recall, F1, acc_count[label]))
        over_acc = over_cor/over_pre            # 计算总的正确率
        over_rec = over_cor/over_tru            # 计算总的召回率
        over_f1 = over_acc*over_rec*2/(over_acc + over_rec)         # 计算总的f1值
        print ('   %6.5s        %.3f          %.3f        %.3f        %d' % ('Total', over_acc, over_rec, over_f1, over_cor))

        return True


    def get_sample_data(self, sentences):
        tag = []
        tags = []
        sent = []
        sents = []
        for s in sentences:
            for word in s:
                tup = []
                tag.append(word.split('\t')[2].strip())
                tup.append(word.split('\t')[0])
                tup.append(word.split('\t')[1])
                sent.append(tuple(tup))
            sents.append(sent)
            tags.append(tag)
            sent = []
            tag = []
        return sents, tags



if __name__ == '__main__':

    ner = MaxEntropy()
    ner.load_model("model/Maxent_tweet_model.pickle")

    token = [('a', 'NN'), ('Peter', 'NNP'), ('Blackburn', 'NNP')]
    print ner.sent_classify(token)
