import os
from pytorch_transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import pickle
import random
import numpy as np
import collections
from collections import Counter
import sys

def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

def bert4token(tokenizer, title, attribute, value):
    title = tokenizer.tokenize(title)
    attribute = tokenizer.tokenize(attribute)
    value = tokenizer.tokenize(value)
    tag = ['O']*len(title)

    for i in range(0,len(title)-len(value)):
        if title[i:i+len(value)] == value:
            for j in range(len(value)):
                if j==0:
                    tag[i+j] = 'B'
                else:
                    tag[i+j] = 'I'
    title_id = tokenizer.convert_tokens_to_ids(title)
    attribute_id = tokenizer.convert_tokens_to_ids(attribute)
    value_id = tokenizer.convert_tokens_to_ids(value)
    tag_id = [TAGS[_] for _ in tag]
    return title_id, attribute_id, value_id, tag_id

def nobert4token(tokenizer, title, attribute, value):

    def get_char(sent):
        tmp = []
        s = ''
        for char in sent.strip():
            if char.strip():
                cp = ord(char)
                if _is_chinese_char(cp):
                    if s:
                        tmp.append(s)
                    tmp.append(char)
                    s = ''
                else:
                    s += char
            elif s:
                tmp.append(s)
                s = ''
        if s:
            tmp.append(s)
        return tmp

    title_list = get_char(title)
    attribute_list = get_char(attribute)
    value_list = get_char(value)

    tag_list = ['O']*len(title_list)
    for i in range(0,len(title_list)-len(value_list)):
        if title_list[i:i+len(value_list)] == value_list:
            for j in range(len(value_list)):
                if j==0:
                    tag_list[i+j] = 'B'
                else:
                    tag_list[i+j] = 'I'

    title_list = tokenizer.convert_tokens_to_ids(title_list)
    attribute_list = tokenizer.convert_tokens_to_ids(attribute_list)
    value_list = tokenizer.convert_tokens_to_ids(value_list)
    tag_list = [TAGS[i] for i in tag_list]

    return title_list, attribute_list, value_list, tag_list


max_len = 40
def X_padding(ids):
    if len(ids) >= max_len:  
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) 
    return ids

tag_max_len = 6
def tag_padding(ids):
    if len(ids) >= tag_max_len: 
        return ids[:tag_max_len]
    ids.extend([0]*(tag_max_len-len(ids))) 
    return ids

def rawdata2pkl4nobert(path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    titles = []
    attributes = []
    values = []
    tags = []
    with open(path, 'r') as f:
        for index, line in enumerate(tqdm(f.readlines())):
            line = line.strip('\n')
            if line:
                title, attribute, value = line.split('<$$$>')
                if attribute in ['适用季节','品牌'] and value in title and _is_chinese_char(ord(value[0])):
                    title, attribute, value, tag = nobert4token(tokenizer, title, attribute, value)
                    titles.append(title)
                    attributes.append(attribute)
                    values.append(value)
                    tags.append(tag)
    print([tokenizer.convert_ids_to_tokens(i) for i in titles[:3]])
    print([[id2tags[j] for j in i] for i in tags[:3]])
    print([tokenizer.convert_ids_to_tokens(i) for i in attributes[:3]])
    print([tokenizer.convert_ids_to_tokens(i) for i in values[:3]])

    df = pd.DataFrame({'titles': titles, 'attributes': attributes, 'values': values, 'tags': tags},
                      index=range(len(titles)))
    print(df.shape)
    df['x'] = df['titles'].apply(X_padding)
    df['y'] = df['tags'].apply(X_padding)
    df['att'] = df['attributes'].apply(tag_padding)

    index = list(range(len(titles)))
    random.shuffle(index)
    train_index = index[:int(0.9 * len(index))]
    valid_index = index[int(0.9 * len(index)):int(0.96 * len(index))]
    test_index = index[int(0.96 * len(index)):]

    train = df.loc[train_index, :]
    valid = df.loc[valid_index, :]
    test = df.loc[test_index, :]

    train_x = np.asarray(list(train['x'].values))
    train_att = np.asarray(list(train['att'].values))
    train_y = np.asarray(list(train['y'].values))

    valid_x = np.asarray(list(valid['x'].values))
    valid_att = np.asarray(list(valid['att'].values))
    valid_y = np.asarray(list(valid['y'].values))

    test_x = np.asarray(list(test['x'].values))
    test_att = np.asarray(list(test['att'].values))
    test_value = np.asarray(list(test['values'].values))
    test_y = np.asarray(list(test['y'].values))

    with open('../data/中文_适用季节.pkl', 'wb') as outp:
        pickle.dump(train_x, outp)
        pickle.dump(train_att, outp)
        pickle.dump(train_y, outp)
        pickle.dump(valid_x, outp)
        pickle.dump(valid_att, outp)
        pickle.dump(valid_y, outp)
        pickle.dump(test_x, outp)
        pickle.dump(test_att, outp)
        pickle.dump(test_value, outp)
        pickle.dump(test_y, outp)



def rawdata2pkl4bert(path, att_list):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    with open(path, 'r') as f:
        lines = f.readlines()
        for att_name in tqdm(att_list):
            print('#'*20+att_name+'#'*20)
            titles = []
            attributes = []
            values = []
            tags = []
            for index, line in enumerate(lines):
                line = line.strip('\n')
                if line:
                    title, attribute, value = line.split('<$$$>')
                    if attribute in [att_name] and value in title: #and _is_chinese_char(ord(value[0])):
                        title, attribute, value, tag = bert4token(tokenizer, title, attribute, value)
                        titles.append(title)
                        attributes.append(attribute)
                        values.append(value)
                        tags.append(tag)
            if titles:
                print([tokenizer.convert_ids_to_tokens(i) for i in titles[:3]])
                print([[id2tags[j] for j in i] for i in tags[:3]])
                print([tokenizer.convert_ids_to_tokens(i) for i in attributes[:3]])
                print([tokenizer.convert_ids_to_tokens(i) for i in values[:3]])
                df = pd.DataFrame({'titles':titles,'attributes':attributes,'values':values,'tags':tags}, index=range(len(titles)))
                print(df.shape)
                df['x'] = df['titles'].apply(X_padding)
                df['y'] = df['tags'].apply(X_padding)
                df['att'] = df['attributes'].apply(tag_padding)

                index = list(range(len(titles)))
                random.shuffle(index)
                train_index = index[:int(0.85*len(index))]
                valid_index = index[int(0.85*len(index)):int(0.95*len(index))]
                test_index = index[int(0.95*len(index)):]

                train = df.loc[train_index,:]
                valid = df.loc[valid_index,:]
                test = df.loc[test_index,:]

                train_x = np.asarray(list(train['x'].values))
                train_att = np.asarray(list(train['att'].values))
                train_y = np.asarray(list(train['y'].values))

                valid_x = np.asarray(list(valid['x'].values))
                valid_att = np.asarray(list(valid['att'].values))
                valid_y = np.asarray(list(valid['y'].values))

                test_x = np.asarray(list(test['x'].values))
                test_att = np.asarray(list(test['att'].values))
                test_value = np.asarray(list(test['values'].values))
                test_y = np.asarray(list(test['y'].values))

                att_name = att_name.replace('/','_')
                with open('../data/all/{}.pkl'.format(att_name), 'wb') as outp:
                # with open('../data/top105_att.pkl', 'wb') as outp:
                    pickle.dump(train_x, outp)
                    pickle.dump(train_att, outp)
                    pickle.dump(train_y, outp)
                    pickle.dump(valid_x, outp)
                    pickle.dump(valid_att, outp)
                    pickle.dump(valid_y, outp)
                    pickle.dump(test_x, outp)
                    pickle.dump(test_att, outp)
                    pickle.dump(test_value, outp)
                    pickle.dump(test_y, outp)

def get_attributes(path):
    atts = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line:
                title, attribute, value = line.split('<$$$>')
                atts.append(attribute)
    return [item[0] for item in Counter(atts).most_common()]


if __name__=='__main__':
    TAGS = {'':0,'B':1,'I':2,'O':3}
    id2tags = {v:k for k,v in TAGS.items()}
    path = '../data/raw.txt'
    att_list = get_attributes(path)
    # rawdata2pkl4bert(path, att_list)
    rawdata2pkl4nobert(path)