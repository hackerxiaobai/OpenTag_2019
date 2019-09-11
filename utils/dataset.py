import pickle
import torch
from torch.utils.data import Dataset,DataLoader

class MyDataset(Dataset):
    def __init__(self, X, Y, att):
        self.data = [{'x':X[i],'y':Y[i],'att':att[i]} for i in range(X.shape[0])]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_dataloader(opt):
    with open(opt.pickle_path, 'rb') as inp:
        train_x = pickle.load(inp)
        train_att = pickle.load(inp)
        train_y = pickle.load(inp)
        valid_x = pickle.load(inp)
        valid_att = pickle.load(inp)
        valid_y = pickle.load(inp)
        test_x = pickle.load(inp)
        test_att = pickle.load(inp)
        test_value = pickle.load(inp)
        test_y = pickle.load(inp)
    print("train len:",train_x.shape)
    print("test len:",test_x.shape)
    print("valid len", valid_x.shape)

    train_dataset = MyDataset(train_x, train_y, train_att)
    valid_dataset = MyDataset(valid_x, valid_y, valid_att)
    test_dataset = MyDataset(test_x, test_y, test_att)

    try:
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        return train_dataloader,valid_dataloader,test_dataloader
    except:
        pass
    return None

# if __name__=='__main__':
#     train_dataloader, valid_dataloader, test_dataloader = get_dataloader()
#     for batch in train_dataloader:
#         print(batch['x'].shape)
#         print(batch['y'].shape)
#         print(batch['att'].shape)