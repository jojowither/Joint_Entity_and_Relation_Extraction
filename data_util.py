import torch
import torch.utils.data as Data
# from pytorch_pretrained_bert import BertTokenizer, BertModel

import numpy as np
import copy
import json
import pickle
import os

class Schema():
    
    UNK_TOKEN = "<UNKNOWN>"
    PAD_TAG = "<PAD>"    # entity pad & sentence pad
    REL_PAD = 'Rel-Pad'
    REL_NONE = 'Rel-None'

    def __init__(self, dataset='conll04'):
        
        self.dataset = dataset
        if dataset=='conll04':
            self.Entity_tags = ['Peop', 'Loc', 'Org', 'Other']
            self.Relation_tags = ['Located_In', 'Work_For', 'OrgBased_In', 'Live_In', 'Kill']
            
        elif dataset=='ADE':
            self.Entity_tags = ['drugs', 'diseases']
            self.Relation_tags = ['ADE']
            
        elif dataset=='ACE04':
            self.Entity_tags = ['PER', 'ORG', 'GPE', 'LOC', 'FAC', 'WEA', 'VEH']
            self.Relation_tags = ['PHYS', 'PER_SOC', 'EMP_ORG', 'ART', 'OTHER_AFF', 'GPE_AFF']
            
        elif dataset=='ACE05' or 'ACE05_cross':
            self.Entity_tags = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
            self.Relation_tags = ['ART', 'GEN_AFF', 'ORG_AFF', 'PART_WHOLE', 'PER_SOC', 'PHYS']

            

        self.ent2ix = EntTagDict(self)
        self.rel2ix = RelTagDict(self)
        
        self.ix2ent = {v: k for k, v in self.ent2ix.items()}
        self.ix2rel = {v: k for k, v in self.rel2ix.items()}
        
        self.tag2eid = {tag:idx for idx, tag in enumerate(self.Entity_tags)}
        self.tag2rid = {tag:idx for idx, tag in enumerate(self.Relation_tags)}
        
        self.eid2tag = {v: k for k, v in self.tag2eid.items()}
        self.rid2tag = {v: k for k, v in self.tag2rid.items()}



class TagDict(dict):
    '''
    Base tag-index dictionary data structure
    It also stores a list to provide index-tag lookup.
    '''

    def __init__(self, tags_list):
        self.tags = self.define(tags_list)
        super().__init__(((t, i) for i, t in enumerate(self.tags)))

    def inv(self, idx):
        return self.tags[idx]

    def define(self, schema):
        raise NotImplementedError()




class EntTagDict(TagDict):

    def define(self, schema):
        '''
        Define entity tags in presumed BILOU scheme.
        Input:
            schema:
                An instance of data_util.Schema
        Output:
            bio_tags:
                A list of tags of Begining/Intermediate of entity and non-entity
        '''

        tag_type = ['B', 'I', 'L', 'O', 'U']

        bio_tags = []
        for t in tag_type:
            for e in schema.Entity_tags:
                if t != 'O':
                    bio_tags.append(t + '-' + e)


        bio_tags.sort()
        bio_tags = [schema.UNK_TOKEN, schema.PAD_TAG] + bio_tags + ['O']

        return bio_tags

        
        

class RelTagDict(TagDict):

    def define(self, schema):
        
        bi_r_tag = []
        
        for r_tag in schema.Relation_tags:
            bi_r_tag.append(r_tag+"#A2B")
            bi_r_tag.append(r_tag+"#B2A")
            
        
        bi_r_tag.sort()
        relation_tags = [schema.REL_PAD, schema.REL_NONE]  + bi_r_tag
        return relation_tags



# ====================================================

class BIOLoader(Data.DataLoader):
    
    def __init__(self, data, max_len, batch_size, schema, tokenizer, args,
                 embedding='XLNet_base', shuffle=False, device=torch.device('cpu')):
        
        '''
        Load corpus and dictionary if available to initiate a torch DataLoader
        Input:
            data_path:
                The string of path to BIO-format corpus.
            max_len:
                The maximal tokens allowed in a sentence.
            batch_size:
                The batch_size parameter as a torch DataLoader.
            schema:
                An instance of data_util.Schema
            word_to_ix: optional
                The word (token) dictionary to index. Use the dict to map sentences
                into indexed sequences if provided, or try to load it from disk if 
                the path is provided.
                If the dictionary does not present in the path, this class would
                write the newly parsed dictionary to the path.
            shuffle: optional
                The shuffle parameter as a torch Dataloader.
            embedding: optional
                Use 'BERT_base', 'BERT_large', 'BERT_base_finetune' or 'GloVe '
            device: optional
                The device at which the dataset is going to be loaded.
        '''
        
        self.max_len = max_len
        self.device = device
        self.tokenizer = tokenizer
        self.bi_fill = args.bi_fill                   
                
        self.raw_input, *results = self.preprocess(data, schema)
        
        
        self.embedding = embedding
        if embedding!='GloVe':
            embedding_indexeds = self.get_pretrain_input()
        else:           
            embedding_indexeds = self.get_w2v_input()
        results = [embedding_indexeds]+results

                                                                     
        torch_dataset = Data.TensorDataset(*(x.to(device) for x in results))
        
        super().__init__(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False
        )
    
    
    

    def preprocess(self, data, schema):
        if schema.dataset == 'conll04':
            data = readfile(data)
        sent_list, ent_list, rel_list = split_to_list(data, schema)
        reserved_index = filter_len(sent_list, self.max_len)
        filter_word, filter_ent, filter_rel = filter_sentence(reserved_index, sent_list, ent_list, rel_list)
        f_w, f_e, f_r = deep_copy_lists(filter_word, filter_ent, filter_rel)
        input_padded, ent_padded, rel_padded = pad_all(f_w, f_e, f_r, self.max_len)
        #================================================
        ent_var = prepare_all(ent_padded, schema.ent2ix)
        rel_var = prepare_rel(rel_padded, schema.rel2ix, self.bi_fill)
        #================================================

        self.batch_index = torch.from_numpy(np.asarray(reserved_index))
              
        return sent_list, ent_var, rel_var, self.batch_index



    def get_pretrain_input(self, _raw_input=None):
        self.indexeds = []

        if _raw_input==None:
            _raw_input = self.raw_input

            self.texts_with_OOV = ['']*len(_raw_input)
            # this wordpiece_ranges defines wordpiece and sentencepiece
            self.wordpiece_ranges = ['']*len(_raw_input)

            for idx in self.batch_index:
                self.token_process(' '.join(_raw_input[idx]), idx)

        else:
            self.texts_with_OOV = ['']*len(_raw_input)
            self.wordpiece_ranges = ['']*len(_raw_input)
            self.token_process(_raw_input)
        
        return torch.stack(self.indexeds)


    def token_process(self, text, idx=None):
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        indexed_tokens = self.pretrain_pad(indexed_tokens)

        if self.embedding.split('_')[0]=='BERT':
            wordpiece_range = self.wordpiece_combine(text, tokenized_text)

        elif self.embedding.split('_')[0]=='XLNet':
            wordpiece_range = self.sentencepiece_combine(text, tokenized_text)

        self.indexeds.append(torch.tensor(indexed_tokens))
        
        if idx:
            self.texts_with_OOV[idx] = tokenized_text
            self.wordpiece_ranges[idx] = wordpiece_range
        else:
            self.texts_with_OOV[0] = tokenized_text
            self.wordpiece_ranges[0] = wordpiece_range


    def pretrain_pad(self, indexed_tokens):
        # self.max_len+74 == the range of maxlen + the slice after wordpiece
        indexed_tokens += [0 for i in range(self.max_len+94 - len(indexed_tokens))]
        return indexed_tokens


    def wordpiece_combine(self, raw_text, tokenized_text):
        raw_text = raw_text.lower().split()
        
        wordpiece_range = []
        wordpiece = []
        piece_count = 0
        tokenized_pos = 0
        record_str = ''
        
        for i, raw in enumerate(raw_text):
            for j, tokenized in enumerate(tokenized_text[i+tokenized_pos:]):
                j = i+tokenized_pos+j
                if raw==tokenized:
                    break
                if raw!=tokenized:
                    if tokenized[:2]=='##':
                        tokenized = tokenized[2:]
                    if raw.find(tokenized)!=-1 :   
                        if record_str==raw:
                            pass

                        else:
                            wordpiece.append(j)
                            piece_count+=1
                            record_str+=tokenized

                        if j+1==len(tokenized_text):
                            wordpiece_range.append(wordpiece)
                            wordpiece = []
                            tokenized_pos = piece_count-len(wordpiece_range)
                            record_str = ''
                            break

                    if raw.find(tokenized)==-1:  
                        wordpiece_range.append(wordpiece)
                        wordpiece = []
                        tokenized_pos = piece_count-len(wordpiece_range)
                        record_str = ''
                        break
                        
        
        return wordpiece_range


    def sentencepiece_combine(self, raw_text, tokenized_text):
        raw_text = raw_text.split()
        wordpiece = []
        wordpiece_range = []   
        piece_count = 0
        tokenized_pos = 0
        record_str = ''

        for i, tokenized in enumerate(tokenized_text):
            if len(tokenized)==1 and tokenized =='▁':
                wordpiece_range.append(wordpiece)
                wordpiece = []
                wordpiece.append(i)

            elif tokenized[0] =='▁':
                wordpiece_range.append(wordpiece)
                wordpiece = []
                wordpiece.append(i)

            else:
                wordpiece.append(i)

        wordpiece_range.append(wordpiece)

        return wordpiece_range[1:]



    
    
    def get_w2v_input(self):
        from torchnlp.word_to_vector import GloVe
        vectors = GloVe() 
        indexeds = []      
        
        for idx in self.batch_index:
            text = self.raw_input[idx]
            pad_num = self.max_len-len(text)
            indexeds.append(torch.cat((vectors[text],torch.zeros([pad_num,300])),0))
            
        return torch.stack(indexeds)




# ==================================================

def readfile(data):
    with open(data, "r", encoding="utf-8") as f:
        content = f.read().splitlines()
        
    return content
        
    


def get_word_and_label(_content, start_w, end_w, schema):
    word_list = []
    ent_list = []
    rel_list = []
    
    for word_set in _content[start_w:end_w]:
        word_set = word_set.split()
        if len(word_set)==1:
            word_list.append(' ')
            ent_list.append('O')
            rel_list.append(schema.REL_NONE)
        
        else:
            word_list.append(word_set[0])
            ent_list.append(word_set[1])
                        
            try:
                testerror = word_set[2]
            except IndexError:
                rel_list.append(schema.REL_NONE)
            else:
                rel_list.append(word_set[2:])
                
    
    return word_list, ent_list, rel_list


def split_to_list(content, schema):
    init = 0
    word_list = []
    ent_list = []
    rel_list = []

    for now_token, c in enumerate(content):
        if c == '':
            words, ents, rels = get_word_and_label(content, init, now_token, schema)
            init = now_token + 1
            word_list.append(words)
            ent_list.append(ents)
            rel_list.append(rels)
            
    return word_list, ent_list, rel_list

# ==================================================

def word2index(sent_list):
    vocab = { Schema.UNK_TOKEN, Schema.PAD_TAG }
    vocab.update((word for sent in sent_list for word in sent))
    
    return { w: i for i, w in enumerate(vocab) }

def dict_inverse(tag_to_ix):
    return {v: k for k, v in tag_to_ix.items()}
    

def index2tag(indexs, ix_to):
    return [ix_to[i] for i in indexs.cpu().numpy()]


# ==================================================

def find_max_len(word_list):
    max_len = 0
    for i in range(len(word_list)):
        if max_len < len(word_list[i]):
            max_len = len(word_list[i])
            
    return max_len

# ====== filter the length of sentence more than MAX_LEN =======

def filter_len(word_list, max_len):
    reserved_index = []
    for i in range(len(word_list)):
        if len(word_list[i]) < max_len:
            reserved_index.append(i)
            
    return reserved_index


def filter_sentence(reserved_index, word_list, ent_list, rel_list):

    filter_word = list(word_list[i] for i in reserved_index)
    filter_ent = list(ent_list[i] for i in reserved_index)
    filter_rel = list(rel_list[i] for i in reserved_index)
    return filter_word, filter_ent, filter_rel

# ==================================================

def pad_seq(seq, pad, max_len):
    seq += [pad for i in range(max_len - len(seq))]
    return seq

def pad_all(filter_word, filter_ent, filter_rel, max_len):
    input_padded = [pad_seq(s, Schema.PAD_TAG, max_len) for s in filter_word]
    ent_padded = [pad_seq(s, Schema.PAD_TAG, max_len) for s in filter_ent]
    rel_padded = [pad_seq(s, Schema.REL_PAD, max_len) for s in filter_rel]
    
    return input_padded, ent_padded, rel_padded

def deep_copy_lists(filter_word, filter_ent, filter_rel):
    f_w = copy.deepcopy(filter_word)
    f_e = copy.deepcopy(filter_ent)
    f_r = copy.deepcopy(filter_rel)
    
    return f_w, f_e, f_r

# ==================================================

def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        try:
            idxs.append(to_ix[w])            
        except KeyError:
            idxs.append(to_ix[Schema.UNK_TOKEN])
    
    return torch.tensor(idxs, dtype=torch.long)

def prepare_all(seqs, to_ix):
    seq_list = []
    for i in range(len(seqs)):
        seq_list.append(prepare_sequence(seqs[i], to_ix))
        
    seq_list = torch.stack(seq_list)
        
    return seq_list


def prepare_rel(rel_padded, to_ix, bi_fill):
    '''
    Prepare relation label data structure
    Output:
        rel_ptr: BATCH*LEN*LEN
            Labels for whether a relation exists from the former to the later token
    '''
 
    num_seqs, max_len, num_rels = len(rel_padded), len(rel_padded[-1]), len(to_ix)
    rel_ptr = torch.ones(num_seqs, max_len, max_len, dtype=torch.long)

    for i, rel_seq in enumerate(rel_padded):
        rel_dict = {}
        for j, token_seq in enumerate(rel_seq):
            if token_seq != Schema.REL_PAD:
                rel_ptr[i][j][:j+1] = to_ix[Schema.REL_NONE]
                if token_seq != Schema.REL_NONE:  
                    for k, rel in enumerate(token_seq):
                        rel_token = rel.split('-')
                        if rel_token[1] not in rel_dict:
                            rel_dict[rel_token[1]] = {'rel':rel_token[0], 'loc':rel_token[2], 'idx':j}
                        
                        else:
                            record_loc = rel_dict[rel_token[1]]['loc'] 
                            record_idx = rel_dict[rel_token[1]]['idx']
                            

                            if record_loc=='A':
                                rel_ptr[i][j][record_idx] = to_ix[rel_token[0]+"#B2A"]
                                if bi_fill:
                                    rel_ptr[i][record_idx][j] = to_ix[rel_token[0]+"#A2B"]

                            elif record_loc=='B':
                                rel_ptr[i][j][record_idx] = to_ix[rel_token[0]+"#A2B"] 
                                if bi_fill:
                                    rel_ptr[i][record_idx][j] = to_ix[rel_token[0]+"#B2A"]
                            
    return rel_ptr



def get_cv_path(cv_dir):
    cv_fullpath = []
    
    for root, dirs, files in os.walk(cv_dir):
        for f in files:
            cv_fullpath.append(os.path.join(root, f))
            
    return cv_fullpath
        

    
def get_cv_path_file(cv_fullpath, process_path):
    
    filename_fullpath = []
    with open(cv_fullpath, "r", encoding="utf-8") as f:
        filename = f.read().splitlines()
        
        
    for fn in filename:
        fn = os.path.join(process_path, fn+'.txt')
        filename_fullpath.append(fn)
        
    return filename_fullpath


def get_cv_context(filename_fullpath, dataset):
    cv_content = []
    
    if dataset=='ACE04' or dataset=='ACE05' or dataset=='ACE05_cross':
        for fn in filename_fullpath:
            with open(fn, "r", encoding="utf-8") as f:
                content = f.read().splitlines()
                cv_content.extend(content)
                
    elif dataset=='ADE':
        with open(filename_fullpath, "r", encoding="utf-8") as f:
            content = f.read().splitlines()
            cv_content.extend(content)

    return cv_content



def calculate_maxlen(cv_contents):
    count_sentence = {i:0 for i in range(489)}
    start_idx = 0
    end_idx = 0
    maxlen = 0
    
    for i, word in enumerate(cv_contents):
        if word=='':
            end_idx = i
            if maxlen<(end_idx-start_idx):
                maxlen = end_idx-start_idx
            count_sentence[end_idx-start_idx]+=1
            start_idx = i+1
            
    
    return count_sentence