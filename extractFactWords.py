# 用几种方法抽取事实词,获得一个词的各种特征
import json
import spacy
import re
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

if __name__ == '__main__':


    filepath='best_sample_human_annotations_sentence.json'
    with open(filepath, 'r') as load_f:
        samples=json.load(load_f)



    final_list=[]
    hashlist=['38955255','37895159','37471830','32457391']
    # hashlist=['32457391']
    corpus=[]
    for hash in hashlist:
        for sample_dict in samples:
            if sample_dict['hash']==hash and sample_dict["model_name"] == "BERTS2S":
                corpus.append(sample_dict['article'])
            else:
                pass

    # sapcy提取依存关系主干
    dependency_list=['ROOT', 'nsubj', 'nsubjpass', 'compound', 'poss', 'pcomp', 'xcomp', 'ccomp', 'conj', 'relcl',
                     'dobj', 'pobj', 'iobj', 'appos', 'acl']
    pos_list=['PRON', 'VERB', 'PROPN', 'NOUN', 'NUM']
    ent_list=[]




    # # 语料
    # corpus=[
    #     "This is the first document.",
    #     'This is the second second document.',
    #     'And the third one.',
    #     'Is this the first document?',
    #     'apple\'s size is small, and apples are big.He was seized by alQaeda in the Islamic Maghreb (AQIM) along with two other men, one of whom was freed in a dawn raid in 2015'
    # ]

    nlp=spacy.load("en_core_web_sm")

    vectorizer=CountVectorizer()  # 将文本中的词语转换为词频矩阵
    X=vectorizer.fit_transform(corpus)
    word=vectorizer.get_feature_names()  # 获取词袋中所有文本关键词

    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(X)  # 将词频矩阵X统计成TF-IDF值
    weight=tfidf.toarray()

    top_k_perc=0.3  # 或者按照文章长度取30%？
    tfidf_kvalue=[]  # 每篇文章的tfidf阈值

    for i in range(len(weight)):
        # 存储特征
        doc_dict={}
        article=[]  # 文章原文
        factlabel_dependency=[]  # 通过句法依存关系提取出的事实词
        factlabel_pos=[]  # 词性提取事实词
        factlabel_ent=[]  # 实体词
        factlabel_keyword=[]

        EEG_path="/Users/zhuyingqi/PycharmProjects/pickERPsamples/data/%s_EEG_data.json" % (hashlist[i])
        with open(EEG_path, 'r') as load_f:
            EEG_data=json.load(load_f)

        array=weight[i]

        doc=nlp(corpus[i])  # 一篇文章
        top_k=math.ceil(top_k_perc*len(doc))
        # print(top_k)
        top_k_idx=array.argsort()[::-1][top_k]  # 文章中第k大的位置。
        tfidf_kvalue.append(array[top_k_idx])


        for ent in doc.ents:
            ent_text=re.sub("\\(|\\)|\\{|\\}|\\[|\\]", "", ent.text)
            entity=ent_text.split(' ')
            if len(entity) > 1:
                ent_list.extend(entity)
            else:
                ent_list.append(ent.text)

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
            if token.dep_ in dependency_list:
                factlabel_dependency.append(1)
            else:
                factlabel_dependency.append(0)

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
        doc_dict['dependency']=factlabel_dependency
        doc_dict['pos']=factlabel_pos
        doc_dict['tfidf']=factlabel_keyword
        doc_dict['eeg']=EEG_data

        # print(doc_dict.keys())
        # print(len(doc_dict['hash']))
        # print(len(doc_dict['article']))
        # print(len(doc_dict['entity']))
        # print(len(doc_dict['dependency']))
        # print(len(doc_dict['pos']))
        # print(len(doc_dict['tfidf']))
        # print(len(doc_dict['eeg']))

        factf_file="/Users/zhuyingqi/PycharmProjects/pickERPsamples/data/%s_%s_%s.json"%('EXP3','SUB3',hashlist[i])
        with open(factf_file, "w") as f:
            json.dump(doc_dict, f, indent=1)



'''
# sapcy提取依存关系主干
dependency_list=['ROOT', 'nsubj', 'nsubjpass', 'compound', 'poss', 'pcomp', 'xcomp', 'ccomp', 'conj', 'relcl',
                 'dobj', 'pobj', 'iobj', 'appos', 'acl']
pos_list=['PRON', 'VERB', 'PROPN', 'NOUN', 'NUM']
ent_list=[]

def abstract_dependency(text):
    nlp=spacy.load("en_core_web_sm")
    doc=nlp(text)

    article=[]#文章原文


    factlabel_dependency=[]  # 通过句法依存关系提取出的事实词
    factlabel_pos=[]#词性提取事实词
    factlabel_ent=[]#实体词
    for sentence in doc.sents:
        sent=nlp(sentence.text)

        for ent in sent.ents:
            ent_text=re.sub("\\(|\\)|\\{|\\}|\\[|\\]", "",ent.text )
            entity=ent_text.split(' ')
            if len(entity)>1:
                ent_list.extend(entity)
            else:
                ent_list.append(ent.text)

        for token in sent:
            if token.dep_=='appos':
                print(token.text,token.pos_)

            if token.dep_ in dependency_list:
                print(token.text,token.pos_,token.dep_)
                article.append(token.text)
                factlabel_dependency.append(1)
            else:
                article.append(token.text)
                factlabel_dependency.append(0)

            if token.pos_ in pos_list:
                factlabel_pos.append(1)
            else:
                factlabel_pos.append(0)

            if token.text in ent_list:
                factlabel_ent.append(1)
            else:
                # print(token.text)
                factlabel_ent.append(0)

    return article,factlabel_dependency,factlabel_pos,factlabel_ent



if __name__ == '__main__':
    #用一个dict存储
    filepath='human_annotations_sentence.json'
    with open(filepath, 'r') as load_f:
        samples=json.load(load_f)
    for sample_dict in samples:
        if sample_dict['hash']=='39419795':
            text=sample_dict['article']
    text='He was seized by alQaeda in the Islamic Maghreb (AQIM) along with two other men, one of whom was freed in a dawn raid in 2015'


    result=abstract_dependency(text)
    # print(result)


    filepath='human_annotations_sentence.json'
    with open(filepath, 'r') as load_f:
        samples=json.load(load_f)

    # out_file_path='NoE_human_annotations_sentence.json'
    # out_data=[]
    #
    # doc_hash_path='NoE_documents_model.json'
    # doc_hash_path_csv='NoE_documents_model.csv'
    # doc_hash_model=[]


    model_name=[]
    doc_hash=[]
    for sample_dict in samples:



        article=sample_dict['article']
        article_list=article.split(" ")
        article_len=len(article_list)
        if article_len<300:
            doc_hash.append(sample_dict['hash'])

        else:pass
    print(set(doc_hash))
'''
    # with open(out_file_path, "w") as f:
    #     json.dump(out_data, f,indent=1)
    #
    # with open(doc_hash_path, "w") as f:
    #     json.dump(doc_hash_model, f,indent=1)
    #
    # with open(doc_hash_path_csv, mode="w", encoding="utf-8-sig", newline="") as f:
    #     writer=csv.writer(f)
    #     writer.writerows(doc_hash_model)

