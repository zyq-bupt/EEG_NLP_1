#根据hash.txt和hash.npz,对应出这次实验的脑电信号，保存成
import numpy as np
import re
import spacy
import os



expNo=4
subNo=4
artNo_list=[9,10,11,12,13,14,15] # 本次实验的article编号。
for i in artNo_list:
    articleNo=i
    npzfilename="EXP%s_SUB%s_ART%s" % (expNo,subNo,articleNo)

    wordplay_path="./data/EXP%s_play_data/%s.txt"%(expNo,npzfilename)
    wpfile=open(wordplay_path)
    words=wpfile.readlines()

    npz_path="./data/EXP4_EEGdata/%s.npz"%(npzfilename)
    EEG_data=np.load(npz_path, allow_pickle=True)['data']

    nlp=spacy.load("en_core_web_sm")
    word_list=[]
    eeg_data=[]#(spacy word number, 28, 375)
    words=words[2:len(words)-3]#(play word number, 32, 375)
    #！！！注意！！！words[2:len(words)-3]，2和len(words)-3是去掉了开头的静息脑电和结尾的问题判断，
    # 具体看EXP4_SUB4_ART9.txt。截取出来的words是所有播放的词。
    # print(len(words))
    # print(EEG_data.shape)
    for i in range(len(words)):
        item=words[i]
        word=item.split(" ")[-1]
        word=re.sub("\n", "", word)
        tokens=nlp(word)

        for t in tokens:
            eeg_data.append(EEG_data[i].tolist())

    eeg_data=np.array(eeg_data)

    path='./data/EXP%s_feature' % (expNo)
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    #不应该存成字典！！因为一篇文章中可能出现相同词！！
    wordfile="./data/EXP%s_feature/%s_EEG_alig.npz"%(expNo,npzfilename)#alignment
    np.savez(wordfile, data=eeg_data)

    # npz_alig_path="./data/EXP%s_feature/%s_EEG_alig.npz"%(expNo,npzfilename)
    # EEG_data=np.load(npz_alig_path, allow_pickle=True)['data']
    # print(EEG_data.shape)



