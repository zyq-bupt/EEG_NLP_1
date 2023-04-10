# 用几种方法抽取事实词,获得一个词的各种特征
import json
import spacy
import re
import os
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

if __name__ == '__main__':

    # # 提取语料库，为计算tfidf。
    # corpus_hash_file='./corpus/corpus_hash.json'
    # with open(corpus_hash_file, "r") as f:
    #     hashlist=json.load(f)
    # filepath='./corpus/human_annotations_sentence.json'
    # with open(filepath, 'r') as load_f:
    #     samples=json.load(load_f)
    # print(len(hashlist))
    # corpus=[]
    # pickedhash=[]
    # for sample_dict in samples:
    #     if sample_dict['hash'] in hashlist and sample_dict['hash'] not in pickedhash:
    #         corpus.append(sample_dict['article'])
    #         pickedhash.append(sample_dict['hash'])
    # print('corpus before.shape:', len(corpus))
    # corpus_file="./corpus/corpus.json"
    # with open(corpus_file, "w") as f:
    #     json.dump(corpus, f, indent=1)




    #START

    article_hash={'1': "36506952", '2': "32143053", '3': "36914884", '9': "37895159",
                  '10':"37471830",'11':"33652722",'12':"31566848",'13':"38592703",
                  '14':"31920236",'15':"26625099"}

    expNo=4
    subNo=4
    artNo_list=[9,10,11,12,13,14,15]  # 本次实验的article编号。

    hashlist=[]
    feature_filename_list=[]

    #本次实验的电脑播放结果文件。!!!注意：
    compplay_filepath='./data/computer_play.json'
    with open(compplay_filepath, 'r') as load_f:
        samples=json.load(load_f)
    #加载计算tfidf的语料库
    corpus_file="./corpus/corpus.json"
    with open(corpus_file, 'r') as load_f:
        corpus=json.load(load_f)
    print('corpus before.shape:',len(corpus))
    #将本次实验的文章加入tfidf语料库
    findhash=[]
    for artNo in artNo_list:

        feature_filename="EXP%s_SUB%s_ART%s" % (expNo,subNo,artNo)
        hash=article_hash['%s'%(artNo)]
        feature_filename_list.append(feature_filename)
        hashlist.append(hash)
        # print(feature_filename,hash)
        for sample_dict in samples:
            if sample_dict['hash']==hash:
                corpus.append(sample_dict['article'])
                findhash.append(sample_dict['hash'])
            else:
                pass

    ##!!!如果找不到一些文章，请更新/data/computer_play.json
    print('已找到的文章hash:',findhash)
    print('没有找到的文章hash:', list(set(hashlist)-set(findhash)))
    # print('corpus later:',len(corpus))

    vectorizer=CountVectorizer()  #将文本中的词语转换为词频矩阵
    X=vectorizer.fit_transform(corpus)
    word=vectorizer.get_feature_names()  #获取词袋中所有文本关键词
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(X)  #将词频矩阵X统计成TF-IDF值
    weight=tfidf.toarray()#TF-IDF矩阵，size（文章顺序，tfidf值）

    top_k_perc=0.3  # 或者按照文章长度取30%
    tfidf_kvalue=[]  # 每篇文章的tfidf阈值

    # 加载sapcymodel，提取依存关系主干、词性、实体
    nlp=spacy.load("en_core_web_sm")
    dep_list=['ROOT', 'nsubj', 'nsubjpass', 'compound', 'poss', 'pcomp',
                     'xcomp', 'ccomp', 'conj', 'relcl','dobj', 'pobj', 'iobj',
                     'appos', 'acl']
    pos_list=['PRON', 'VERB', 'PROPN', 'NOUN', 'NUM']
    ent_list=[]

    article_number=len(findhash)#本次实验文章数目
    start=len(weight)-len(findhash)#语料库中，文章排序位置

    for i in range(article_number):

        doc_dict={}#存储特征
        article=[]  #文章原文
        factlabel_dep=[]  #通过句法依存关系提取出的事实词
        factlabel_pos=[]  # 词性提取事实词
        factlabel_ent=[]  # 实体词
        factlabel_keyword=[]

        # print('weight.shape:', weight.shape)
        # print('start:', start)
        # print('i:', i)
        array=weight[start+i]#一篇文章的tfidf值
        doc=nlp(corpus[start+i])  # 一篇文章的词列表

        # print('array.shape:',array.shape)
        # print('doc.len:',len(doc))

        #获取tfidf阈值
        top_k=math.ceil(top_k_perc*len(doc))
        top_k_idx=array.argsort()[::-1][top_k]  #文章中第k大的位置。
        tfidf_kvalue.append(array[top_k_idx])

        #获取实体词
        for ent in doc.ents:
            ent_text=re.sub("\\(|\\)|\\{|\\}|\\[|\\]", "", ent.text)
            entity=ent_text.split(' ')
            if len(entity) > 1:
                ent_list.extend(entity)
            else:
                ent_list.append(ent.text)

        #开始判断文章中的词是否是实体词
        for token in doc:
            w=token.text
            article.append(token.text)
            # tfidf
            if w in word:
                if array[word.index(w)] > array[top_k_idx]:
                    factlabel_keyword.append(1)
                else:
                    factlabel_keyword.append(0)
            else:
                factlabel_keyword.append(0)

            # dependency
            if token.dep_ in dep_list:
                factlabel_dep.append(1)
            else:
                factlabel_dep.append(0)

            # pos
            if token.pos_ in pos_list:
                factlabel_pos.append(1)
            else:
                factlabel_pos.append(0)

            # entity
            if token.text in ent_list:
                factlabel_ent.append(1)
            else:
                factlabel_ent.append(0)

        doc_dict['hash']=hashlist[i]
        doc_dict['article']=article
        doc_dict['entity']=factlabel_ent
        doc_dict['dependency']=factlabel_dep
        doc_dict['pos']=factlabel_pos
        doc_dict['tfidf']=factlabel_keyword

        path='./data/EXP%s_feature'% (expNo)
        if os.path.exists(path):pass
        else:
            os.mkdir(path)
        factf_file="./data/EXP%s_feature/%s_feature.json" % (expNo, feature_filename_list[i])
        with open(factf_file, "w") as f:
            json.dump(doc_dict, f, indent=1)



        # print(doc_dict.keys())
        print(doc_dict['hash'])
        # print(len(doc_dict['article']))
        # print(len(doc_dict['entity']))
        # print(len(doc_dict['dependency']))
        print(len(doc_dict['pos']))
        # print(len(doc_dict['tfidf']))
        npz_alig_path="./data/EXP%s_feature/%s_EEG_alig.npz"%(expNo,feature_filename_list[i])
        EEG_data=np.load(npz_alig_path, allow_pickle=True)['data']
        print(EEG_data.shape)
