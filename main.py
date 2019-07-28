import torch
from torch import optim
import torch.utils.data as Data
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.model_selection import KFold
from data_util import Schema
from data_util import BIOLoader
from models import JointERE
import data_util

import numpy as np
import copy
import json
import pickle
import os
import argparse

## BUG檢查
## 文件註解
## 畫圖code

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/storage/jojo2/data/')
parser.add_argument('--dataset', type=str, default='conll04', help='Choose conll04, ADE, ACE04, ACE05')

parser.add_argument('--USE_CUDA', type=bool, default=True)
parser.add_argument('--CUDA_device', type=str, default='1')
parser.add_argument('--embedding', type=str, default='BERT_base', help='Choose BERT_base, BERT_large, \
                                                                        BERT_base_finetune or GloVe ')

parser.add_argument('--n_iter', type=int, default=600, 
					help='Recommand 600 iters for conll04, 100 iters for ADE, 250 iters for ACE04 and 800 iters for ACE05')

parser.add_argument('--batch_size', type=int, default=32, 
					help='If you use bert_finetune, please adjust batch_size to 10, otherwise it will OOM')
parser.add_argument('--bilstm_n_layers', type=int, default=2)
parser.add_argument('--word_dropout', type=float, default=0.2)
parser.add_argument('--bilstm_dropout', type=float, default=0.2)
parser.add_argument('--rel_dropout', type=float, default=0.2)
parser.add_argument('--d_rel', type=int, default=512)
parser.add_argument('--n_r_head', type=int, default=16, help='Numbers of head in relation layer(multi-head attrntion)')
parser.add_argument('--point_out', type=int, default=128, help='Pointer network size')
parser.add_argument('--to_seq_attn', type=bool, default=True, help='Use sequence-wise multi-head attention')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate in Adam')
parser.add_argument('--weight_decay', type=float, default=1e-5)

args = parser.parse_args()


def model_main(training_data, dev_data, param_list, max_len, batch_size, 
               schema, tokenizer, n_iter, embedding, dataset, use_device, n_fold=None, test_data=None):
    
    loader = BIOLoader(training_data, max_len, batch_size, schema, tokenizer, 
                       embedding, shuffle=True, device=use_device)
    dev_loader = BIOLoader(dev_data, max_len, batch_size, schema, tokenizer, 
                           embedding, shuffle=False, device=use_device) 


    model = JointERE(*param_list).to(use_device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    model.fit(loader, dev_loader, n_fold, n_iter=n_iter, optimizer=optimizer, dataset=dataset)
    
    if test_data!=None:
        test_loader = BIOLoader(test_data, max_len, batch_size, schema, tokenizer, 
                                embedding, shuffle=False, device=use_device) 
        print()
        print('Test Set')
        e_score, er_score, all_er_score, acc_zone_block = model.score(test_loader, silent=True, rel_detail=True)
    
    else:
        e_score, er_score, all_er_score, acc_zone_block = model.score(dev_loader, silent=True, rel_detail=True)
        
    return e_score, er_score


def use_cv(cv_dir, param_list, max_len, batch_size, 
           schema, tokenizer, n_iter, embedding, use_device, n_fold, dataset, process_path=None):
    
    average_ent = []
    average_rel = []

    cv_fullpaths = data_util.get_cv_path(cv_dir)
    train_cv_contents = []
    test_cv_contents = []

    kf = KFold(n_splits=n_fold)
    for train, test in kf.split(cv_fullpaths):

        for cv in train:
            if dataset=='ACE04':
                filename_fullpath = data_util.get_cv_path_file(cv_fullpaths[cv], process_path)
                cv_content = data_util.get_cv_context(filename_fullpath, dataset)
            else:
                cv_content = data_util.get_cv_context(cv_fullpaths[cv], dataset)
            train_cv_contents.extend(cv_content)

        for cv in test:
            if dataset=='ACE04':
                filename_fullpath = data_util.get_cv_path_file(cv_fullpaths[cv], process_path)
                cv_content = data_util.get_cv_context(filename_fullpath, dataset)
            else:
                cv_content = data_util.get_cv_context(cv_fullpaths[cv], dataset)
            test_cv_contents.extend(cv_content)

        
        #  do main
        e_score, er_score = model_main(train_cv_contents, test_cv_contents, param_list, max_len, batch_size, 
                                       schema, tokenizer, n_iter, embedding, dataset, use_device, int(test))
        #  do main


        train_cv_contents = []
        test_cv_contents = []

        average_ent.append(e_score)
        average_rel.append(er_score)
        
    
    print()
    print('The result of cross validation')
    print()
    print('{}-fold cross validation in NER'.format(n_fold))
    print("precision  \t recall  \t fbeta_score")
    print("{:.3f} \t\t {:.3f} \t\t {:.3f} \t".format(*np.mean(average_ent, axis=0)))
    print()
    print('{}-fold cross validation in RE'.format(n_fold))
    print("precision  \t recall  \t fbeta_score")
    print("{:.3f} \t\t {:.3f} \t\t {:.3f} \t".format(*np.mean(average_rel, axis=0)))




os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_device
USE_CUDA = args.USE_CUDA
use_device = torch.device('cuda' if USE_CUDA else 'cpu')


#======================================================================


root = args.data_root
dataset = args.dataset
dataset_root = os.path.join(root, dataset)

batch_size = args.batch_size
embedding = args.embedding
n_iter = args.n_iter
d_r_v = args.d_rel//args.n_r_head

if embedding=='BERT_base' or embedding=='BERT_base_finetune':
    d_model = 768
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')

elif embedding=='BERT_large':
    d_model = 1024
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    bert = BertModel.from_pretrained('bert-large-uncased')

elif embedding=='GloVe':
	d_model = 300


if dataset=='conll04':
	max_len = 118
elif dataset=='ADE':
	max_len = 92
elif dataset=='ACE04':
	max_len = 120
elif dataset=='ACE05':
	max_len = 110



schema = Schema(dataset)
param_list = [bert, max_len, d_model, args.bilstm_n_layers, 
              args.word_dropout, args.bilstm_dropout, args.rel_dropout,
              args.d_rel, args.n_r_head, d_r_v, args.point_out, schema, 
              args.embedding, args.to_seq_attn]


print()
print('Start to train the {}'.format(dataset))


if dataset == 'conll04':
    training_data = os.path.join(dataset_root, 'training_set.txt')
    dev_data = os.path.join(dataset_root, 'dev_set.txt')
    test_data = os.path.join(dataset_root, 'test_set.txt')
    
    e_score, er_score = model_main(training_data, dev_data, param_list, max_len, batch_size, 
                               	   schema, tokenizer, n_iter, embedding, dataset, use_device, test_data=test_data)
    
elif dataset == 'ADE':
    # n_iter = 100
    n_fold = 10
    
    cv_dir = os.path.join(dataset_root, 'cv_data')

    use_cv(cv_dir, param_list, max_len, batch_size, 
           schema, tokenizer, n_iter, embedding, use_device, n_fold, dataset)
    
elif dataset == 'ACE04':   
    cv_dir = os.path.join(dataset_root, 'split')
    process_path = os.path.join(dataset_root, 'process_data')
    # n_iter = 250
    n_fold = 5
    
    use_cv(cv_dir, param_list, max_len, batch_size, 
           schema, tokenizer, n_iter, embedding, use_device, n_fold, dataset, process_path)

elif dataset == 'ACE05': 
    # n_iter = 800
    
    split_dir = os.path.join(dataset_root, 'split/')
    process_path = os.path.join(dataset_root, 'process_data')
    
    train_split = os.path.join(split_dir, 'train.txt')
    dev_split = os.path.join(split_dir, 'dev.txt')
    test_split = os.path.join(split_dir, 'test.txt')
    
    train_fn_fullpath = data_util.get_cv_path_file(train_split, process_path)
    training_data = data_util.get_cv_context(train_fn_fullpath, dataset)

    dev_fn_fullpath = data_util.get_cv_path_file(dev_split, process_path)
    dev_data = data_util.get_cv_context(dev_fn_fullpath, dataset)

    test_fn_fullpath = data_util.get_cv_path_file(test_split, process_path)
    test_data = data_util.get_cv_context(test_fn_fullpath, dataset)   
    
    e_score, er_score = model_main(training_data, dev_data, param_list, max_len, batch_size, 
                                   schema, tokenizer, n_iter, embedding, dataset, use_device, test_data=test_data)