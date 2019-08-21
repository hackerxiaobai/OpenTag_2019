import os
from config import opt
import random
import numpy as np 
import torch 
import models
from utils import get_dataloader
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train(**kwargs):

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
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
            optimizer.step()
            if ii % 2 == 0:
                print('epoch:%04d,------------loss:%f'%(epoch,loss.item()))

if __name__=='__main__':
    import fire
    fire.Fire()