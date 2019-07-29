import torch
import torch.utils.data as Data

import numpy as np
import copy
import json
import pickle
import os

root = 'data/'
dataset = 'conll04'
dataset_root = os.path.join(root, dataset)
root_data = os.path.join(dataset_root, 'conll04.corp')

def readfile(data):
    with open(data, "r", encoding="utf-8") as f:
        content = f.read().splitlines()
        
    return content

def split_sentence(data):
    num=0
    max_len=0
    record_len = {}
    isspace = False
    source_sentences = []
    entity_seqs = []
    relation_seqs = []
        
    sentence = ''
    entity_seq = ''
    relation_seq = ''
    for row_data in data:

        try:
            num_sent = int(row_data[0])
        
        except IndexError:
            if isspace==False:
                isspace=True
            
            else:
                num+=1
                isspace = False
                source_sentences.append(sentence)
                entity_seqs.append(entity_seq)
                relation_seqs.append(relation_seq)
                sentence = ''
                entity_seq = ''
                relation_seq = ''
                
        else:    
            if len(row_data.split('\t'))==9:
                sentence = sentence + row_data.split('\t')[5] + ' '
                entity_seq = entity_seq + row_data.split('\t')[1] + ' '
                
                if int(row_data.split('\t')[2])+1 > max_len:
                    max_len = int(row_data.split('\t')[2])+1
                
            
            elif len(row_data.split('\t'))==3 :
                relation_seq = relation_seq + row_data + ' '
    
    
    # clean the space in the tail
    for i, (s, e, r) in enumerate(zip(source_sentences, entity_seqs, relation_seqs)):
        source_sentences[i] = s[:-1]
        entity_seqs[i] = e[:-1]
        relation_seqs[i] = r[:-1]
    
    return source_sentences, entity_seqs, relation_seqs, max_len


def record_total_len(seq, max_len=200):
    record_len = {i:0 for i in range(1, max_len+1)}
    
    for s in seq:
        record_len[len(s.split(' '))] += 1
        
    return record_len


def filter_non_relation(source_sentences, entity_seqs, relation_seqs):
    
    reserve_idx = []
    for i,r in enumerate(relation_seqs):
        if r!='': 
            reserve_idx.append(i)
            
    filter_sentences = []
    filter_entitys = []
    filter_relations = []
    
    for i in reserve_idx:
        filter_sentences.append(source_sentences[i])
        filter_entitys.append(entity_seqs[i])
        filter_relations.append(relation_seqs[i])
        
    return filter_sentences, filter_entitys, filter_relations


def process_seqs(source_sentences, entity_seqs, relation_seqs):
    
    for idx, (sent, e_seq, r_seq) in enumerate(zip(source_sentences, entity_seqs, relation_seqs)):   
        
        # split to list
        sent_l = sent.split(' ')
        e_seq_l = e_seq.split(' ')
        r_seq_l = r_seq.split(' ')
        w2span_dict = {}
        
        for i, s in enumerate(sent_l):
            if ',' in s:
                sent_l[i] = s.replace(',','COMMA')
            if '-LRB-' in s:
                sent_l[i] = s.replace('-LRB-','(')
            if '-RRB-' in s:
                sent_l[i] = s.replace('-RRB-',')')
                
            # remove '.' in the word, like 'a.m.' -> 'am'
            if '.' in s and len(s)>1:
                sent_l[i] = s.replace('.','')

            
        for i, s in enumerate(sent_l):            
            
            if '/' in s and e_seq_l[i]!='O':
                w2span_dict[i] = s.split('/')        
            
            # remove the dot end of the word, if only appear dot dont remove
            try :
                s[-1]=='.' 
            except IndexError:
                pass
                # Does not affect
                # it is '', and the raw input is '..' or '...'
            else:
                if s[-1]=='.' and len(s)>1:
                    sent_l[i] = s[:-1]
                
        
        keys = sorted(w2span_dict.keys(), reverse=True)
        
        for k in keys:
            entity = e_seq_l[k]
            del sent_l[k]
            del e_seq_l[k]
            B_idx = k
            
            word_len = len(w2span_dict[k])
            for i, w in enumerate(w2span_dict[k]):
                sent_l.insert(k, w)
                if i+1==word_len:
                    e_seq_l.insert(k, 'L-'+entity)
                else:
                    e_seq_l.insert(k, 'I-'+entity)
                k+=1
            
            e_seq_l[B_idx] = 'B-'+entity

        
        
        for i, e in enumerate(e_seq_l):
            if e!='O' and e[0]!='B' and e[0]!='I' and e[0]!='L':
                e_seq_l[i] = 'U-'+e_seq_l[i]
            
            if e=='Loc':
                e_seq_l[i] = 'U-'+e_seq_l[i]

        
        record_loc = {}
        previous_idx = 0
        count_itag = 0
        
        # Record: Previous starting position: {now starting position, now ending position}
        for now_idx, e in enumerate(e_seq_l):
            if e[0]=='U' or e[0]=='B':
                record_loc[now_idx-count_itag] = {'start':now_idx, 'end':now_idx}
                previous_idx = now_idx-count_itag
                
            elif e[0]=='I':
                count_itag += 1
            
            elif e[0]=='L':
                count_itag += 1
                record_loc[previous_idx]['end'] = now_idx
                
                
                   
        now_r = 0
        r_list = [' ' for _ in range(len(sent_l))]
        if r_seq_l==[''] :
            relation_seqs[idx] = r_list
            
        else:
            for triple_r in r_seq_l:
                
                _a = int(triple_r.split('\t')[0])
                _b = int(triple_r.split('\t')[1])
                rel = triple_r.split('\t')[2]
                
                
                # triple in A                              
                end_address = record_loc[_a]['end']
                
                if r_list[end_address]==' ':
                    r_list[end_address] = [rel+'-'+str(now_r)+'-'+'A']
                else:
                    r_list[end_address].append(rel+'-'+str(now_r)+'-'+'A')
                        
                # triple in B
                end_address = record_loc[_b]['end']
                
                if r_list[end_address]==' ':
                    r_list[end_address] = [rel+'-'+str(now_r)+'-'+'B']
                else:
                    r_list[end_address].append(rel+'-'+str(now_r)+'-'+'B')
                now_r += 1
                        
    
                
    
        # Remove the COMMA in the entity
        for i, (s,e) in enumerate(zip(sent_l, e_seq_l)):
#             if s=='COMMA' and e!='O':
#                 del sent_l[i]
#                 del e_seq_l[i]
#                 del r_list[i]
                
            if s=='' :
                sent_l[i] = '.'
#                 del sent_l[i]
#                 del e_seq_l[i]
#                 del r_list[i] 
            
#             if s=='--' :
#                 del sent_l[i]
#                 del e_seq_l[i]
#                 del r_list[i]  
            
        
        source_sentences[idx] = ' '.join(sent_l)
        entity_seqs[idx] = ' '.join(e_seq_l)
        relation_seqs[idx] = r_list

    return source_sentences, entity_seqs, relation_seqs



# concatenate the sentence, entity and relation to the list
def concat_s_e_r(sents, ent_seqs, rel_seqs):
    
    # record the sentence, entity and relation
    all_combine_list = []
    
    for idx, (sent, e_seq, r_seq) in enumerate(zip(sents, ent_seqs, rel_seqs)):    
        sent_l = sent.split(' ')
        e_seq_l = e_seq.split(' ')
        
        data_represent = ''
        for s,e,r in zip(sent_l, e_seq_l, r_seq):
            if type(r) is list:
                r = ' '.join(r)

            data_represent += s+' '+e+' '+r+'\n'
        
        all_combine_list.append(data_represent)
        
    return all_combine_list


data = readfile(root_data)
source_sentences, entity_seqs, relation_seqs, max_len = split_sentence(data)

#======filter non relation sequences======
source_sentences, entity_seqs, relation_seqs = filter_non_relation(source_sentences, entity_seqs, relation_seqs)
# ========================================

source_sentences, entity_seqs, relation_seqs = process_seqs(source_sentences, entity_seqs, relation_seqs)
record_len = record_total_len(source_sentences)
all_combine_data = concat_s_e_r(source_sentences, entity_seqs, relation_seqs)


print('The numbers of data', len(all_combine_data))

test_size = int(len(all_combine_data)*0.2)
dev_size = int((len(all_combine_data)-test_size)*0.1)
train_size = len(all_combine_data)-dev_size-test_size

train_dataset, dev_dataset, test_dataset = Data.random_split(all_combine_data, [train_size, dev_size, test_size])

print('train_dataset', len(train_dataset))
print('dev_dataset', len(dev_dataset))
print('test_dataset', len(test_dataset))

with open(os.path.join(dataset_root, 'training_set.txt'), "w") as f:
    for item in train_dataset:
        f.write("%s\n" % item)

with open(os.path.join(dataset_root, 'dev_set.txt'), "w") as f:
    for item in dev_dataset:
        f.write("%s\n" % item)

with open(os.path.join(dataset_root, 'test_set.txt'), "w") as f:
    for item in test_dataset:
        f.write("%s\n" % item)