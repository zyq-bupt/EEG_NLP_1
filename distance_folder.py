import os
import glob
import json
import spacy
import numpy as np
import csv

# 加载预训练的模型
nlp = spacy.load("en_core_web_md")

def calculate_distances(class_N, class_E, all_words):

    # 计算每个词的向量表示（事实词、非事实词、全部）
    vectors_N = [nlp(word).vector for word in class_N]
    vectors_E = [nlp(word).vector for word in class_E]
    all_vectors = [nlp(word).vector for word in all_words]

    # 计算每个类别的中心向量
    center_N = np.mean(vectors_N, axis=0)
    center_E = np.mean(vectors_E, axis=0)
    center_all_words = np.mean(all_vectors, axis=0)

    # 计算类内距离
    distances_N = [np.linalg.norm(vector - center_N) for vector in vectors_N]
    distances_E = [np.linalg.norm(vector - center_E) for vector in vectors_E]
    distances_all_words = [np.linalg.norm(vector - center_all_words) for vector in all_vectors]

    intra_distance_N = np.mean(distances_N)
    intra_distance_E = np.mean(distances_E)
    intra_distance_all_words = np.mean(distances_all_words)

    # 计算类间距离
    inter_distance = np.linalg.norm(center_N - center_E)

    return intra_distance_N, intra_distance_E, inter_distance, intra_distance_all_words

# 准备CSV文件
csv_file = open("Experiment_1/Distance/distance.csv", "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

# 写入表头
header = ["filename"]
suffixes = ["entity", "dependency", "pos", "tfidf"]
for suffix in suffixes:
    header.extend([f"Class N ({suffix}) intra-distance",    # 非事实词的类内距离
                   f"Class E ({suffix}) intra-distance",    # 事实词的类内距离
                   f"All Words ({suffix}) Intra-distance",  # 所有词的类内距离
                   f"Inter-distance ({suffix})"])           # 类间距离

csv_writer.writerow(header)

# 循环读取多个文件夹中的所有JSON文件
folders = ["Experiment_1/Extract/data/EXP2_feature", 
           "Experiment_1/Extract/data/EXP3_feature", 
           "Experiment_1/Extract/data/EXP4_feature"]

for folder in folders:
    for file in glob.glob(os.path.join(folder, "*.json")):
        with open(file, "r") as f:
            data = json.load(f)

        article = data["article"]            # 文章
        entity = data["entity"]              # 事实词
        dependency = data["dependency"]      # 依存关系
        pos = data["pos"]                    # 词性
        tfidf = data["tfidf"]                # TF-IDF

        categories = [entity, dependency, pos, tfidf]  # 4个类别

        filename = os.path.basename(file) 
        row = [filename]

        for suffix, category in zip(suffixes, categories):  
            class_A = [word for i, word in enumerate(article) if category[i] == 0]  # 非事实词
            class_B = [word for i, word in enumerate(article) if category[i] == 1]  # 事实词
            intra_distance_A, intra_distance_B, inter_distance, intra_distance_all_words = calculate_distances(class_A, class_B, article)   # 计算距离

            row.extend([intra_distance_A, intra_distance_B, intra_distance_all_words, inter_distance])  # 写入CSV文件

        csv_writer.writerow(row)

# 关闭CSV文件
csv_file.close()
