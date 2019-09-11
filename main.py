import os
import sys
from time import strftime, localtime
from collections import Counter
from config import opt
from pytorch_transformers import BertTokenizer
import random
import numpy as np 
import torch 
import models
from utils import get_dataloader
# from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from seqeval.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def get_attributes(path='./data/raw.txt'):
    atts = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line:
                title, attribute, value = line.split('<$$$>')
                atts.append(attribute)
    return [item[0] for item in Counter(atts).most_common()]

def train(**kwargs):
    log_file = '{}-{}.log'.format(opt.model, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    att_list = get_attributes()

    tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_name)
    tags2id = {'':0,'B':1,'I':2,'O':3}
    id2tags = {v:k for k,v in tags2id.items()}

    opt._parse(kwargs)

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
   
    # step1: configure model
    model = getattr(models, opt.model)(opt)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # step2: data
    train_dataloader,valid_dataloader,test_dataloader = get_dataloader(opt)
    
    # step3: criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    # step4 train
    for epoch in range(opt.max_epoch):
        model.train()
        for ii,batch in tqdm(enumerate(train_dataloader)):
            # train model
            optimizer.zero_grad()
            x = batch['x'].to(opt.device)
            y = batch['y'].to(opt.device)
            att = batch['att'].to(opt.device)
            inputs = [x, att, y]
            loss = model.log_likelihood(inputs)
            loss.backward()
            #CRF
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=3)
            optimizer.step()
            if ii % opt.print_freq == 0:
                print('epoch:%04d,------------loss:%f'%(epoch,loss.item()))

    model.save()

    preds, labels = [], []
    for index, batch in enumerate(valid_dataloader):
        model.eval()
        x = batch['x'].to(opt.device)
        y = batch['y'].to(opt.device)
        att = batch['att'].to(opt.device)
        inputs = [x, att, y]
        predict = model(inputs)

        if index % 5 == 0:
            print(tokenizer.convert_ids_to_tokens([i.item() for i in x[0].cpu() if i.item()>0]))
            length = [id2tags[i.item()] for i in y[0].cpu() if i.item()>0]
            print(length)
            print([id2tags[i] for i in predict[0][:len(length)]])

        # 统计非0的，也就是真实标签的长度
        leng = []
        for i in y.cpu():
            tmp = []
            for j in i:
                if j.item()>0:
                    tmp.append(j.item())
            leng.append(tmp)

        for index, i in enumerate(predict):
            preds.append([id2tags[k] if k>0 else id2tags[3] for k in i[:len(leng[index])]])
            # preds += i[:len(leng[index])]

        for index, i in enumerate(y.tolist()):
            labels.append([id2tags[k] if k>0 else id2tags[3] for k in i[:len(leng[index])]])
            # labels += i[:len(leng[index])]
    # precision = precision_score(labels, preds, average='macro')
    # recall = recall_score(labels, preds, average='macro')
    # f1 = f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds)
    print(report)
    logger.info(report)


if __name__=='__main__':
    import fire
    fire.Fire()