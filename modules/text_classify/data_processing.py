import jieba


data_path = "sources/weibo_senti_100k.csv"
data_stop_path = "sources/hit_stopword"
data_list = open(data_path).readlines()[1:]
stops_word = open(data_stop_path).readlines()
stops_word = [line.strip() for line in stops_word]
stops_word.append(" ")
stops_word.append("\n")

voc_dict = {}
min_seq = 1
top_n = 1000
UNK = "<UNK>"
PAD = "<PAD>"

for item in data_list[:]:
    label = item[0]
    content = item[2:].strip()
    seg_list = jieba.cut(content, cut_all=False)
    seg_res = []
    for seg_item in seg_list:
        if seg_item in stops_word:
            continue
        seg_res.append(seg_item)
        if seg_item in voc_dict.keys():
            voc_dict[seg_item] = voc_dict[seg_item] + 1
        else:
            voc_dict[seg_item] = 1

    print(content)
    print(seg_res)

voc_list = sorted([_ for _ in voc_dict.items() if _[1] > min_seq],
                  key=lambda x:x[1],
                  reverse=True)[:top_n]

voc_dict = {word_count[0]: idx for idx, word_count in enumerate(voc_list)}

voc_dict.update({UNK:len(voc_dict), PAD:len(voc_dict) + 1})

print(voc_dict)

ff = open("sources/dict", "w")
for item in voc_dict.keys():
    ff.writelines("{},{}\n".format(item, voc_dict[item]))
ff.close()
