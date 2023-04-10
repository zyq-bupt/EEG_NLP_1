# 查看原文长度，选出比较短的原文
import json
import csv

def have_error(summary_annot_list,error_type):
    for sent_annot_dict in summary_annot_list:#sent_annot_dict 一个句子，三个评价者的评价
        n=0#如果等于3，则表明3个评价者都认为这个句子有error_type类型错误
        # annotation0_list=sent_annot_dict['annotator_0']
        # annotation1_list=sent_annot_dict['annotator_1']
        # annotation2_list=sent_annot_dict['annotator_2']
        for i in range(3):
            annotation_list=sent_annot_dict['annotator_%s'%(i)]
            if error_type in annotation_list:
                n+=1

        if n==3:
            return True
    return False


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    filepath='./corpus/human_annotations_sentence.json'
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
        if article_len<1000:
            doc_hash.append(sample_dict['hash'])

        else:pass
    print(len(set(doc_hash)))
    doc_hash=list(set(doc_hash))
    # print(doc_hash)
    corpus_hash_file='./corpus/corpus_hash.json'
    with open(corpus_hash_file, "w") as f:
        json.dump(doc_hash, f)
    #
    # with open(doc_hash_path, "w") as f:
    #     json.dump(doc_hash_model, f,indent=1)
    #
    # with open(doc_hash_path_csv, mode="w", encoding="utf-8-sig", newline="") as f:
    #     writer=csv.writer(f)
    #     writer.writerows(doc_hash_model)

