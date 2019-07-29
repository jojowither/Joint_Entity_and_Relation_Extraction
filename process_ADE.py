import json
import random
import numpy as np
import pickle
import os

import torch
import torch.utils.data as Data

from itertools import chain
import more_itertools as mit
from functools import reduce


ADE_root = 'data/ADE'
ADE_data = os.path.join(ADE_root, 'DRUG-AE.rel')


def readfile(data):
    with open(data, "r", encoding="utf-8") as f:
        content = f.read().splitlines()
        
    return content


def process_data(ADE_data):
    
    ID = ''
    sentence = ''
    diseases, drug = '', ''
    overlap_count = 0
    pre_sentence = ''
    same_sentence_pair = 0
    
    input_sentence = []
    input_entity = []
    input_relation = []
        
    
    count=0
    
    for data in ADE_data:
        
        data = remove_punctuation(data)
        data = insert_space_around_brackets(data)
        
        data = data.split('|')
        ID = data[0]
        sentence = data[1]
        diseases = data[2]
        drugs = data[5]
        
        
        isOverlap = check_overlap(diseases, drugs)
        if isOverlap:
            overlap_count+=1
            continue
        else:
  
            sentence = sentence.split()
            diseases = diseases.split()
            drugs = drugs.split()
            
            diseases_index = find_index(sentence, diseases)
            drugs_index = find_index(sentence, drugs)
            
            entity_seq = create_BILOUentity(len(sentence), diseases_index, drugs_index)
    
    
        if pre_sentence!=sentence:
            pre_sentence = sentence
            input_sentence.append(sentence)
            input_entity.append(entity_seq)
            
            same_sentence_pair = 0
            rel_seq = create_relation(len(sentence), diseases_index, drugs_index, same_sentence_pair)
            input_relation.append(rel_seq)
            
            
        else:
            input_idx = len(input_sentence)-1
            input_entity[input_idx] = combine_same_entity(input_entity[input_idx], entity_seq)
            
            same_sentence_pair += 1
            rel_seq = create_relation(len(sentence), diseases_index, drugs_index, same_sentence_pair)
            input_relation[input_idx] = combine_same_rel(input_relation[input_idx], rel_seq)      
            
    
    print('Overlap count: ', overlap_count)
    return input_sentence, input_entity, input_relation



def remove_punctuation(sentence):
           
    sentence = sentence.replace(".", '')
    sentence = sentence.replace(",", '')
    sentence = sentence.replace(":", '')
    sentence = sentence.replace(";", '')
    sentence = sentence.replace("%", '')
    sentence = sentence.replace("\"", '')
    sentence = sentence.replace("?", '')
    sentence = sentence.replace("\'s", '')
    sentence = sentence.replace("\'", '')
    
    sentence = sentence.replace("/", ' ')
    sentence = sentence.replace("-", ' ')
    
        
    return sentence


def insert_space_before_comma(sentence):
    comma_list = list(mit.locate(sentence, lambda x: x == ","))
    for c in reversed(comma_list):
        sentence = sentence[:c]+' '+sentence[c:]
        
    return sentence


def insert_space_around_brackets(sentence):
    ''' (pulmonary edema) -> ( pulmonary edema )
    '''
    front_bracket_list = list(mit.locate(sentence, lambda x: x == "("))
    for c in reversed(front_bracket_list):
        sentence = sentence[:c+1]+' '+sentence[c+1:]
    rear_bracket_list = list(mit.locate(sentence, lambda x: x == ")"))
    for c in reversed(rear_bracket_list):
        sentence = sentence[:c]+' '+sentence[c:]
        
    
    front_bracket_list = list(mit.locate(sentence, lambda x: x == "["))
    for c in reversed(front_bracket_list):
        sentence = sentence[:c+1]+' '+sentence[c+1:]
    rear_bracket_list = list(mit.locate(sentence, lambda x: x == "]"))
    for c in reversed(rear_bracket_list):
        sentence = sentence[:c]+' '+sentence[c:]
        
    return sentence
                

        
def check_overlap(diseases, drugs):
    '''overlapping entities (e.g., “lithium” is a
       drug which is related to “lithium intoxication”)'''
    
    if set(diseases.lower().split()) & set(drugs.lower().split()) == set():
        return False    
    else:
        return True
    
    
def remove_hyphen(sentence):
    return [word.split('-') for word in sentence]
    
    
def flatten(listOfLists):
    "Flatten one level of nesting"
    return list(chain.from_iterable(listOfLists))



def find_index(sentence, entitys):
    index_list = []
    ent_lists = []
    for entity in entitys:
        ent_list = list(mit.locate(sentence, lambda x: x == entity))
        ent_lists.append(set(ent_list))
    
    for i, e_list in enumerate(ent_lists):
        ent_lists[i] = set([e-i for e in e_list])
        
    result = reduce(set.intersection, ent_lists).pop()
    
    idxx = list(map(lambda x: x+result, range(len(entitys))))
        
    return idxx


def create_BILOUentity(seq_len, diseases_index, drugs_index):
    
    entity_seq = ['O']*seq_len
    entity_seq = addBILOU(entity_seq, diseases_index, 'diseases')
    entity_seq = addBILOU(entity_seq, drugs_index, 'drugs')
    
    return entity_seq

            
def addBILOU(e_seq, index, e_type):
    
    for i, idx in enumerate(index):
        
        if len(index)==1:
            e_seq[idx] = 'U-'+ e_type
            
        elif i==0:
            e_seq[idx] = 'B-'+ e_type
            
        elif i+1==len(index):
            e_seq[idx] = 'L-'+ e_type
            
        else:
            e_seq[idx] = 'I-'+ e_type
            
    return e_seq


def combine_same_entity(e1_seq, e2_seq):
    '''將同一句的實體seq做合併
    '''
    new_e1 = []
    for e1,e2 in zip(e1_seq, e2_seq):
        if e1=='O' and e2=='O':
            new_e1.append('O')
        elif e1!='O':
            new_e1.append(e1)
        elif e2!='O':
            new_e1.append(e2)
            
    return new_e1


def create_relation(seq_len, diseases_index, drugs_index, pair_count):
    rel_seq = [' ']*seq_len
    rel_seq = add_rel(rel_seq, diseases_index, pair_count, 'diseases')
    rel_seq = add_rel(rel_seq, drugs_index, pair_count, 'drugs')
    
    return rel_seq

def add_rel(rel_seq, index, pair_count, r_type):
    
    if r_type=='drugs':
        AorB = '-A'
    else:
        AorB = '-B'
    
    rel_seq[index[-1]] = ['ADE-'+str(pair_count)+AorB]
        
    return rel_seq


def combine_same_rel(r1_seq, r2_seq):
    
    for i, (r1,r2) in enumerate(zip(r1_seq, r2_seq)):
        if r1==' ' and r2==' ':
            pass
        elif r1==' ' and r2!=' ':
            r1_seq[i] = [''.join(r2)]
        elif r1!=' ' and r2!=' ':
            r1_seq[i].append(''.join(r2))
            
    return r1_seq


# concatenate the sentence, entity and relation to the list
def concat_s_e_r(sents, ent_seqs, rel_seqs):
    
    # record the sentence, entity and relation
    all_combine_list = []
    
    for idx, (sent, e_seq, r_seq) in enumerate(zip(sents, ent_seqs, rel_seqs)):    
        
        data_represent = ''
        for s,e,r in zip(sent, e_seq, r_seq):
            if type(r) is list:
                r = ' '.join(r)

            data_represent += s+' '+e+' '+r+'\n'
        
        all_combine_list.append(data_represent)
        
    return all_combine_list


ADE_data = readfile(ADE_data)
input_sentence, input_entity, input_relation = process_data(ADE_data)
all_combine_data = concat_s_e_r(input_sentence, input_entity, input_relation)
cv = []

cv_size = int(len(all_combine_data)*0.1)
remain_cv_size = len(all_combine_data)-cv_size*9

for _ in range(9):
    cv.append(cv_size)
cv.append(remain_cv_size)

cv_data =  Data.random_split(all_combine_data, cv)

cv_root = os.path.join(ADE_root, 'cv_data')
os.makedirs(cv_root, exist_ok=True)  

dataname = ['cv{}.txt'.format(i) for i in range(10)]

for name, cv in zip(dataname, cv_data):
    with open(os.path.join(cv_root, name), "w") as f:
        for item in cv:
            f.write("%s\n" % item)