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
parser.add_argument('--model_dict', type=str, default='NER_RE_best.conll04.XLNet_base.32.nobi.backward.Pw_hint.pkl')

parser.add_argument('--USE_CUDA', type=str2bool, default=True)
parser.add_argument('--CUDA_device', type=str, default='1')
parser.add_argument('--embedding', type=str, default='XLNet_base', help='Choose BERT_base, BERT_large, \
                                                                        BERT_base_finetune, XLNet_base, \
                                                                        XLNet_large and GloVe ')

parser.add_argument('--n_iter', type=int, default=300, 
                    help='Recommand 300 iters for conll04, 100 iters for ADE, 125 iters for ACE04 and 150 iters for ACE05')
parser.add_argument('--batch_size', type=int, default=32, 
                    help='If you use bert_finetune, please adjust batch_size to 10, otherwise it will OOM')
parser.add_argument('--bilstm_n_layers', type=int, default=2)
parser.add_argument('--word_dropout', type=float, default=0.2)
parser.add_argument('--bilstm_dropout', type=float, default=0.2)
parser.add_argument('--rel_dropout', type=float, default=0.2)
parser.add_argument('--pair_dropout', type=float, default=0)
parser.add_argument('--d_rel', type=int, default=512)
parser.add_argument('--n_r_head', type=int, default=32, help='Numbers of head in multi-head self-attention')
parser.add_argument('--pair_out', type=int, default=128, help='Pw-hint size')
parser.add_argument('--mh_attn', type=str2bool, default=True, help='Use multi-head attention')
parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate in AdamW')
parser.add_argument('--weight_decay', type=float, default=2e-3)
parser.add_argument('--rel_weight', type=float, default=20, help='the weight of relation tags')
parser.add_argument('--rel_weight_base_tag', type=float, default=20, help='the weights of pad tag and none tag')
parser.add_argument('--scheduler_step', type=float, default=40)
parser.add_argument('--scheduler_gamma', type=float, default=0.5)
parser.add_argument('--clip_value', type=float, default=0.4)
parser.add_argument('--silent', type=str2bool, default=True, help='Boolean to suppress detailed decoding')

parser.add_argument('--bi_fill', type=str2bool, default=False, help='Boolean to fill the relation table in bidirection')
parser.add_argument('--e_d_attn', type=str2bool, default=True, help='Boolean to choose encoder-decoder attention or self-attention on relation layer')
parser.add_argument('--Pw_hint', type=str2bool, default=True, help='Boolean to choose pair-wise hint')
parser.add_argument('--P_hint', type=str2bool, default=True, help='Boolean to use positional hint')
parser.add_argument('--pos_strategy', type=str, default='backward', help='Choose backward, forward or minus on positional hint')
parser.add_argument('--augment_info', type=str2bool, default=False, help='Boolean to augment the distance infomation on encoder')

args = parser.parse_args()




def model_main(training_data, dev_data, param_list, schema, tokenizer, dataset, use_device, n_fold=None, test_data=None):
    
    # param_list = [pre_model, max_len, d_model, d_r_v, schema, use_device, args]
    max_len = param_list[1]
    batch_size = param_list[-1].batch_size
    n_iter = param_list[-1].n_iter
    embedding = param_list[-1].embedding

    loader = BIOLoader(training_data, max_len, batch_size, schema, tokenizer, args,
                       embedding, shuffle=True, device=use_device)
    dev_loader = BIOLoader(dev_data, max_len, batch_size, schema, tokenizer, args,
                           embedding, shuffle=False, device=use_device) 


    model = JointERE(*param_list).to(use_device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    if args.train_eval=='train':
        model.fit(loader, dev_loader, n_fold, n_iter=n_iter, optimizer=optimizer, dataset=dataset)
    elif args.train_eval=='eval':
        model.load_state_dict(torch.load(args.model_dict))
    
    if test_data!=None:
        test_loader = BIOLoader(test_data, max_len, batch_size, schema, tokenizer, args,
                                embedding, shuffle=False, device=use_device) 
        print()
        print('Test Set')
        e_score, er_score, all_er_score, acc_zone_block = model.score(test_loader, silent=args.silent, rel_detail=True)
    
    else:
        e_score, er_score, all_er_score, acc_zone_block = model.score(dev_loader, silent=args.silent, rel_detail=True)
        
    return e_score, er_score


def use_cv(cv_dir, param_list, schema, tokenizer, use_device, n_fold, dataset, process_path=None):
    
    max_len = param_list[1]
    batch_size = param_list[-1].batch_size
    n_iter = param_list[-1].n_iter
    embedding = param_list[-1].embedding

    average_ent = []
    average_rel = []

    cv_fullpaths = data_util.get_cv_path(cv_dir)
    train_cv_contents = []
    test_cv_contents = []

    kf = KFold(n_splits=n_fold)
    for train, test in kf.split(cv_fullpaths):

        for cv in train:
            if dataset=='ACE04' or 'ACE05_cross':
                filename_fullpath = data_util.get_cv_path_file(cv_fullpaths[cv], process_path)
                cv_content = data_util.get_cv_context(filename_fullpath, dataset)
            else:
                cv_content = data_util.get_cv_context(cv_fullpaths[cv], dataset)
            train_cv_contents.extend(cv_content)

        for cv in test:
            if dataset=='ACE04' or 'ACE05_cross':
                filename_fullpath = data_util.get_cv_path_file(cv_fullpaths[cv], process_path)
                cv_content = data_util.get_cv_context(filename_fullpath, dataset)
            else:
                cv_content = data_util.get_cv_context(cv_fullpaths[cv], dataset)
            test_cv_contents.extend(cv_content)

        
        #  do main
        e_score, er_score = model_main(train_cv_contents, test_cv_contents, param_list,  
                                       schema, tokenizer, dataset, use_device, int(test))
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
          'pair_dropout': args.pair_dropout,
          'd_rel': args.d_rel,
          'n_r_head': args.n_r_head,
          'pair_out': args.pair_out,
          'lr': args.lr,
          'weight_decay': args.weight_decay,
          'rel_weight': args.rel_weight,
          'rel_weight_base_tag': args.rel_weight_base_tag,
          'scheduler_step': args.scheduler_step,
          'scheduler_gamma': args.scheduler_gamma, 
          'silent': args.silent,
          'mh_attn':args.mh_attn,
          'clip_value': args.clip_value,
          'bi_fill':args.bi_fill,
          'e_d_attn':args.e_d_attn,
          'Pw_hint':args.Pw_hint,
          'P_hint':args.P_hint,
          'pos_strategy':args.pos_strategy

          }
print()
pprint(config)
print()

#======================================================================

root = args.data_root
dataset = args.dataset
dataset_root = os.path.join(root, dataset)


# BERT_base, BERT_large, BERT_base_finetune, XLNet_base, XLNet_large and GloVe
pre_model_type = args.embedding.split('_')[0]
lg_or_bs = ''
if pre_model_type!='GloVe':
    lg_or_bs = args.embedding.split('_')[1]

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




d_r_v = args.d_rel//args.n_r_head  # demension of relation values
max_len = 100
schema = Schema(dataset)
param_list = [pre_model, max_len, d_model, d_r_v, schema, use_device, args]





if dataset == 'conll04':
    training_data = os.path.join(dataset_root, 'training_set.txt')
    dev_data = os.path.join(dataset_root, 'dev_set.txt')
    test_data = os.path.join(dataset_root, 'test_set.txt')
    
    e_score, er_score = model_main(training_data, dev_data, param_list, 
                                   schema, tokenizer, dataset, use_device, test_data=test_data)
    
elif dataset == 'ADE':
    n_fold = 10
    cv_dir = os.path.join(dataset_root, 'cv_data')

    use_cv(cv_dir, param_list, schema, tokenizer, use_device, n_fold, dataset)
    
elif dataset == 'ACE04':   
    cv_dir = os.path.join(dataset_root, 'split')
    process_path = os.path.join(dataset_root, 'process_data')
    n_fold = 5
    
    use_cv(cv_dir, param_list, schema, tokenizer, use_device, n_fold, dataset, process_path)

elif dataset == 'ACE05': 
    
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
    
    e_score, er_score = model_main(training_data, dev_data, param_list, schema, tokenizer, dataset, use_device, test_data=test_data)


elif dataset == 'ACE05_cross':   
    cv_dir = os.path.join(dataset_root, 'split')
    process_path = os.path.join(dataset_root, 'process_data')
    n_fold = 5
    
    use_cv(cv_dir, param_list, schema, tokenizer, use_device, n_fold, dataset, process_path)