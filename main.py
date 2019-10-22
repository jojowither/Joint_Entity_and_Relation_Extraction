import torch
from torch import optim
import torch.utils.data as Data
from pytorch_transformers import *
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
from pprint import pprint



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')



parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/')
parser.add_argument('--dataset', type=str, default='conll04', help='Choose conll04, ADE, ACE04, ACE05')
parser.add_argument('--train_eval', type=str, default='train', help='Choose train or eval')

parser.add_argument('--USE_CUDA', type=str2bool, default=True)
parser.add_argument('--CUDA_device', type=str, default='1')
parser.add_argument('--embedding', type=str, default='BERT_base', help='Choose BERT_base, BERT_large, \
                                                                        BERT_base_finetune, XLNet_base, \
                                                                        XLNet_large and GloVe ')

parser.add_argument('--n_iter', type=int, default=600, 
                    help='Recommand 600 iters for conll04, 100 iters for ADE, 250 iters for ACE04 and 800 iters for ACE05')

parser.add_argument('--batch_size', type=int, default=32, 
                    help='If you use bert_finetune, please adjust batch_size to 10, otherwise it will OOM')
parser.add_argument('--bilstm_n_layers', type=int, default=2)
parser.add_argument('--word_dropout', type=float, default=0.2)
parser.add_argument('--bilstm_dropout', type=float, default=0.2)
parser.add_argument('--rel_dropout', type=float, default=0.2)
parser.add_argument('--d_rel', type=int, default=512)
parser.add_argument('--n_r_head', type=int, default=16, help='Numbers of head in relation layer(multi-head attention)')
parser.add_argument('--point_out', type=int, default=256, help='Pointer network size')
parser.add_argument('--mh_attn', type=str2bool, default=True, help='Use sequence-wise multi-head attention')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate in Adam')
parser.add_argument('--weight_decay', type=float, default=2e-3)
parser.add_argument('--rel_weight', type=float, default=5, help='the weight of relation tags')
parser.add_argument('--rel_weight_base_tag', type=float, default=2, help='the weights of pad tag and none tag')
parser.add_argument('--scheduler_step', type=float, default=60)
parser.add_argument('--scheduler_gamma', type=float, default=0.5)
parser.add_argument('--silent', type=str2bool, default=True, help='Boolean to suppress detailed decoding')
parser.add_argument('--model_dict', type=str, default='NER_RE_best.ACE05.BERT_base.32.pkl')

args = parser.parse_args()




def model_main(training_data, dev_data, param_list, max_len, batch_size, 
               schema, tokenizer, n_iter, embedding, dataset, use_device, n_fold=None, test_data=None):
    
    loader = BIOLoader(training_data, max_len, batch_size, schema, tokenizer, 
                       embedding, shuffle=True, device=use_device)
    dev_loader = BIOLoader(dev_data, max_len, batch_size, schema, tokenizer, 
                           embedding, shuffle=False, device=use_device) 



    model = JointERE(*param_list).to(use_device)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    if args.train_eval=='train':
        model.fit(loader, dev_loader, n_fold, n_iter=n_iter, optimizer=optimizer, dataset=dataset)
    elif args.train_eval=='eval':
        model.load_state_dict(torch.load(args.model_dict))
    
    if test_data!=None:
        test_loader = BIOLoader(test_data, max_len, batch_size, schema, tokenizer, 
                                embedding, shuffle=False, device=use_device) 
        print()
        print('Test Set')
        e_score, er_score, all_er_score, acc_zone_block = model.score(test_loader, silent=args.silent, rel_detail=True)
    
    else:
        e_score, er_score, all_er_score, acc_zone_block = model.score(dev_loader, silent=args.silent, rel_detail=True)
        
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


config = {'embedding': args.embedding,
          'n_iter': args.n_iter,
          'batch_size': args.batch_size,
          'bilstm_n_layers': args.bilstm_n_layers,
          'word_dropout': args.word_dropout,
          'bilstm_dropout': args.bilstm_dropout,
          'rel_dropout': args.rel_dropout,
          'd_rel': args.d_rel,
          'n_r_head': args.n_r_head,
          'point_out': args.point_out,
          'lr': args.lr,
          'weight_decay': args.weight_decay,
          'rel_weight': args.rel_weight,
          'rel_weight_base_tag': args.rel_weight_base_tag,
          'scheduler_step': args.scheduler_step,
          'scheduler_gamma': args.scheduler_gamma, 
          'silent': args.silent
          }
print()
pprint(config)


#======================================================================

train_eval = args.train_eval
root = args.data_root
dataset = args.dataset
dataset_root = os.path.join(root, dataset)

batch_size = args.batch_size
embedding = args.embedding
n_iter = args.n_iter
d_r_v = args.d_rel//args.n_r_head  # demension of relation values

 # BERT_base, BERT_large, BERT_base_finetune, XLNet_base, XLNet_large and GloVe
pre_model_type = embedding.split('_')[0]
lg_or_bs = ''
if pre_model_type!='GloVe':
    lg_or_bs = embedding.split('_')[1]

if pre_model_type=='BERT':
    tokenizer = BertTokenizer
    pre_model = BertModel
    pre_weight = ['bert-base-uncased', 'bert-large-uncased']
elif pre_model_type=='XLNet':
    tokenizer = XLNetTokenizer
    pre_model = XLNetModel
    pre_weight = ['xlnet-base-cased', 'xlnet-large-cased']
elif pre_model_type=='GloVe':
    d_model = 300
    # add a model but no use
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    pre_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

if lg_or_bs=='base':
    d_model = 768
    tokenizer = tokenizer.from_pretrained(pre_weight[0])
    pre_model = pre_model.from_pretrained(pre_weight[0], output_hidden_states=True)
elif lg_or_bs=='large':
    d_model = 1024
    tokenizer = tokenizer.from_pretrained(pre_weight[1])
    pre_model = pre_model.from_pretrained(pre_weight[1], output_hidden_states=True)



if dataset=='conll04':
    max_len = 120
    # max_len = 100
elif dataset=='ADE':
    max_len = 100
elif dataset=='ACE04':
    max_len = 120
elif dataset=='ACE05':
    # max_len = 110
    max_len = 120



schema = Schema(dataset)
param_list = [pre_model, max_len, d_model, args.bilstm_n_layers, 
              args.word_dropout, args.bilstm_dropout, args.rel_dropout,
              args.d_rel, args.n_r_head, d_r_v, args.point_out, schema, args, 
              args.embedding, args.mh_attn]


print()


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