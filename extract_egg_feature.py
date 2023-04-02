#根据hash.txt和hash.npz,对应出这次实验的脑电信号
import numpy as np
import re
import json
import spacy

hashlist=['38955255','37895159','37471830','32457391']
# hashlist=['38955255']
for hash in hashlist:
    wordplay_path="/Users/zhuyingqi/PycharmProjects/pickERPsamples/data/%s.txt"%(hash)

    wpfile=open(wordplay_path)
    words=wpfile.readlines()
    #(184, 32, 375)
    npz_path="/Users/zhuyingqi/PycharmProjects/pickERPsamples/data/%s.npz"%(hash)
    EEG_data=np.load(npz_path, allow_pickle=True)['data']

    nlp=spacy.load("en_core_web_sm")
    word_list=[]
    eeg_data=[]#32,104,375
    words=words[2:len(words)-3]
    for i in range(len(words)):#to 2 !!!
        item=words[i]
        word=item.split(" ")[-1]
        word=re.sub("\n", "", word)
        tokens=nlp(word)
        eeg=[]
        for j in range(32):
            eeg.append(EEG_data[j][i].tolist())
        for t in tokens:
            eeg_data.append(eeg)
    #不应该存成字典！！因为一篇文章中可能出现相同词！！
    wordfile="/Users/zhuyingqi/PycharmProjects/pickERPsamples/data/%s_EEG_data.json"%(hash)
    with open(wordfile, "w") as f:
        json.dump(eeg_data, f, indent=1)


