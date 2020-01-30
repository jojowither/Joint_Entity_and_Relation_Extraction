import torch
from torch import nn, optim
import torch.nn.functional as F

import time
import numpy as np
import copy
from tqdm import tqdm
import pickle

# from evaluation import evaluate_data, decode_ent, decode_rel
from evaluation import *
from data_util import BIOLoader



class JointERE(nn.Module):
    def __init__(self, pre_model, maxlen, d_model, d_r_v,  
    	schema, use_device, args, embedding='XLNet_base', mh_attn=True):
        '''
        JointERE
            Joint Entity and Relation mention Extraction on Traditional Chinese text
        Input:
            schema:
                An instance of data_util.Schema with definition of entities and relations
            embedding:
                'BERT_base', 'BERT_large', 'BERT_base_finetune' or 'GloVe'
        '''
        
        
        super().__init__()
        
        # Load pre-trained model (weights)
        self.pre_model = pre_model
        self.pre_model.eval()
             
        self.maxlen = maxlen
        self.ent_size = len(schema.ent2ix)                   #es
        self.rel_size = len(schema.rel2ix)                   #rs 
        self.d_model = d_model
        self.tag_embed_dim = self.ent_size                 #TE
        self.schema = schema
        self.n_r_head = args.n_r_head
        self.mh_attn = mh_attn
        self.embedding = embedding
        self.rel_weight = args.rel_weight
        self.rel_weight_base_tag = args.rel_weight_base_tag
        self.scheduler_step = args.scheduler_step
        self.scheduler_gamma = args.scheduler_gamma
        self.activation = _get_activation_fn(args.activation_fn)
        self.args = args
        self.use_device = use_device
    

        self.word_dropout = nn.Dropout(args.word_dropout)      
    
        self.bilstm = nn.LSTM(d_model, d_model // 2, num_layers=args.bilstm_n_layers, 
                              bidirectional=True, batch_first=True, dropout=args.bilstm_dropout) 
        # for param in self.bilstm.parameters():
        #     if len(param.size()) >= 2:
        #         # nn.init.orthogonal_(param.data)
        #         nn.init.normal_(param.data, mean=0, std=np.sqrt(1.0 / (d_model)))
        #     else:
        #         nn.init.normal_(param.data)

           
        self.fc2tag = nn.Linear(d_model+self.tag_embed_dim, self.ent_size)
        self.init_linear(self.fc2tag)
        

        self.softmax = nn.LogSoftmax(dim=-1)
        self.tag_embed = nn.Embedding(self.ent_size, self.tag_embed_dim)
        nn.init.orthogonal_(self.tag_embed.weight)

        self.t2rel = nn.Linear(2*self.tag_embed_dim+d_model, args.d_rel)
        self.init_linear(self.t2rel)
        

        if self.args.P_hint:
            self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(self.maxlen, args.d_rel, 
                                                         padding_idx=0), freeze=True)
        else:
            self.position_enc = None
        self.relation_layer = Relation_Layer(d_r_v, self.rel_size, self.mh_attn, self.args, self.position_enc)
        

        # finetune
        try:
            self.ft = self.embedding.split('_')[2]
        except:
            if self.embedding.split('_')[0]!='GloVe':
                self.ft = 'pre_model'
            else:
                self.ft = 'constant_embedding'
        

        
    def init_linear(self, m):
        nn.init.xavier_normal_(m.weight)
        nn.init.normal_(m.bias)
        
    def one_hot(self, ent_choice):
        y_onehot = ent_choice.new_zeros(ent_choice.size(0), self.ent_size, dtype=torch.float)
        return y_onehot.scatter_(-1, ent_choice.unsqueeze(-1), 1)   

    
    def check_text_len(self, loader, cb_wp_texts, batch_index):

        for i, (t,bi) in enumerate(zip(cb_wp_texts, batch_index)):
            s = loader.raw_input[bi]
                       
            if len(s)!=len(t):
                print(loader.raw_input[bi])
                print(len(loader.raw_input[bi]))
                print(t)
                print(len(t))
                print(loader.wordpiece_ranges[bi])
                                  
            assert len(s)==len(t)         

        
        
    def pad2maxlen(self, enc):
        tensor_zero = enc.new_zeros(self.maxlen - len(enc), self.d_model)
        enc = torch.cat((enc, tensor_zero), 0)
        
        return enc
            
    
    def cb_wordpiece_and_rm_pad(self, enc_output, batch_index, loader):

        cb_wordpiece_tensor = []
        cb_wordpiece_text = []

        for enc, bi in zip(enc_output, batch_index):
            
            text = copy.deepcopy(loader.texts_with_OOV[bi])
            wp_range = copy.deepcopy(loader.wordpiece_ranges[bi])
            

            premodel_len, feature = enc.size()
            text_len = len(text)
            enc = enc[:text_len]

            
            for wp_range_l in reversed(wp_range):
                
                enc[wp_range_l[0]] = torch.mean(torch.stack([enc[idx] for idx in wp_range_l]), dim=0)
                text[wp_range_l[0]] += ''.join([text[idx] for idx in wp_range_l[1:]])

                for idx in reversed(wp_range_l[1:]):
                    del text[idx] 

                enc = torch.cat((enc[0:wp_range_l[0]+1], enc[wp_range_l[-1]+1:]), 0)
            
            assert enc.size()[0]==len(text)  

            
            cb_wp_tensor = self.pad2maxlen(enc)
            cb_wordpiece_tensor.append(cb_wp_tensor)
            cb_wordpiece_text.append(text)
            
        return torch.stack(cb_wordpiece_tensor), cb_wordpiece_text
    
    
        
    def forward(self, embed_input, batch_index, loader, batch_ent=None):
        '''Assume I/O resides on the same device, and so does this module'''
        
        batch_size = embed_input.size()[0]
        entity_tensor = torch.zeros(batch_size, self.maxlen, self.ent_size, device=embed_input.device)  #B x ML x es
        rel_tensor = torch.zeros(batch_size, self.maxlen, self.maxlen, self.rel_size, device=embed_input.device)  #B x ML x ML x rs


        if self.ft=='pre_model':
            with torch.no_grad():
                embed_input = self.pre_model(embed_input)
                embed_input = torch.sum(torch.stack([layer for layer in embed_input[2][:-1]]), dim=0)
                embed_input, cb_wp_texts = self.cb_wordpiece_and_rm_pad(embed_input, batch_index, loader)
                ## 
                # self.check_text_len(loader, cb_wp_texts, batch_index) 
                
        elif self.ft=='finetune':
            embed_input, _ = self.pre_model(embed_input)
            embed_input, cb_wp_texts = self.cb_wordpiece_and_rm_pad(embed_input, batch_index, loader)
            ##
            # self.check_text_len(loader, cb_wp_texts, batch_index) 

            
             
        embed_input = self.word_dropout(embed_input)                                          
        enc_output = self.bilstm(embed_input)[0]      # B x ML x d_model

        encoder_sequence_l = [] 
        
        label = enc_output.new_zeros(batch_size, self.tag_embed_dim)

        for length in range(self.maxlen):
            now_token = enc_output[:,length,:]
            now_token = torch.squeeze(now_token, 1)
            
            combine_pre = torch.cat((label, now_token), 1)                    #B x (TE+d_model)    
            ent_output = self.softmax(self.fc2tag(combine_pre))               #B x es

                       
            # pass the gold entity embedding to the next time step, if available
            if batch_ent is not None:
                label = self.tag_embed(batch_ent[:, length])
                
            else:
                label = self.tag_embed(ent_output.argmax(-1))                 #B x TE
            
            combine_entity = torch.cat((combine_pre, label), 1)                    #B x (2*TE+d_model)
            # Relation layer
            encoder_sequence_l.append(combine_entity)  
            encoder_sequence = torch.stack(encoder_sequence_l).transpose(0, 1)     #B x len x (2*TE+d_model)     
   
            # Calculate attention weights 
            rel_weights = self.relation_layer(self.activation(self.t2rel(encoder_sequence)))     
    
            entity_tensor[:,length,:] = ent_output            
            rel_tensor[:,length,:length+1,:] = rel_weights


        if self.args.bi_fill:
            for length in reversed(range(self.maxlen)):
                ## sentence order is reversed
                ## flip the order of the sentence
                encoder_sequence = torch.stack(encoder_sequence_l[length:]).transpose(0, 1).flip(1)
                rel_weights = self.relation_layer(self.activation(self.t2rel(encoder_sequence))).flip(1)
                rel_tensor[:,length,length:,:] = rel_weights

        return entity_tensor, rel_tensor
        
        
    def entity_loss(self):
        return EntityNLLLoss()
    
    def relation_loss(self):
        return RelationNLLLoss(self.rel_weight, self.rel_weight_base_tag, 
                               self.schema.rel2ix, self.args.bi_fill, self.use_device)


    
    def fit(self, loader, dev_loader, n_fold=None, optimizer=None, n_iter=50, true_ent=False,
            save_model=None, dataset=None):
        
        '''
        Fit JointERE and select paramteres based on validation score
        Input:
            loader:
                The instance of data_util.BIOLoader containing training data
            dev_loader:
                The instance of data_util.BIOLoader containing development/validation data
            optimizer: optional
                An specified optimizer from torch.optim for training model.
                By default Adam(lr=0.01, weight_decay=1e-4, amsgrad=True) would be used.
            n_iter: optional
                The total traing epochs to fit and search the best model.
                The default value is 50 epochs.
            stable_iter: optional
                The epoch after which the model begins to evaluate and select model.
                The default value is 10 epochs.
            save_model: optional
                The path to store model parameters during model selection
                If unspecified, the path defaults to 'checkpoints/relation_extraction_best.{time}.pkl'
        '''
        
        criterion_tag = self.entity_loss()
        criterion_rel = self.relation_loss()

        bi = 'bi' if self.args.bi_fill else 'nobi'
        aug = 'aug' if self.args.augment_info else 'noaug'
        pos = 'no_P_hint' if not self.args.P_hint else self.args.pos_strategy
        pw = 'Pw_hint' if self.args.Pw_hint else 'no_Pw_hint'
        e_score, er_score = (0, 0, 0), (0, 0, 0)
        self.best_er_score = (0, 0, 0)
        dev_F1_list = []

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, 
                                              gamma=self.scheduler_gamma)
        
        if save_model is None:
            hash_id = int(time.time())
            print('Start to train/eval the {}'.format(dataset))
            print('Number of heads: ', self.n_r_head)
            n_r_head = ''
            if self.n_r_head<10:
                n_r_head = str(0)+str(self.n_r_head)
            else:
                n_r_head = str(self.n_r_head)
                
            if n_fold:
                save_model = 'NER_RE_best.{}.{}.{}.{}.{}.{}.{}_fold.pkl'.format(dataset, self.embedding, n_r_head, bi, pos, pw, n_fold)
            else:
                save_model = 'NER_RE_best.{}.{}.{}.{}.{}.{}.pkl'.format(dataset, self.embedding, n_r_head, bi, pos, pw)
        
        for epoch in tqdm(range(n_iter)):
            
            for embed_input, batch_ent, batch_rel, batch_index in loader:
                self.train() 

                # forward
                if true_ent:
                    ent_output, rel_output = self.forward(embed_input, batch_index, loader, batch_ent)
                
                else:
                    ent_output, rel_output = self.forward(embed_input, batch_index, loader)
                

                # backward
                batch_loss_ent = criterion_tag(ent_output, batch_ent)
                batch_loss_rel = criterion_rel(rel_output, batch_rel)   

                batch_loss = batch_loss_ent + batch_loss_rel

                batch_loss.backward()
                
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                torch.nn.utils.clip_grad_value_(self.parameters(), self.args.clip_value)
                
                # update parameters
                optimizer.step()               
                optimizer.zero_grad()


            scheduler.step()



            for embed_input, batch_ent, batch_rel, batch_index in dev_loader:
                self.eval()
                
                ent_output, rel_output = self.forward(embed_input, batch_index, dev_loader)
                
                batch_loss_ent_dev = criterion_tag(ent_output, batch_ent)
                batch_loss_rel_dev = criterion_rel(rel_output, batch_rel)  
            
            
            print("epoch: %d | ent loss %.4f | rel loss %.4f | total loss %.4f" \
          % (epoch+1, batch_loss_ent, batch_loss_rel, batch_loss))
            print("      %s  | val ent loss %.4f | val rel loss %.4f"
          % (" "*len(str(epoch+1)), batch_loss_ent_dev, batch_loss_rel_dev))

            print('lr {}'.format(optimizer.param_groups[0]['lr']))


            if epoch>-1:
                e_score, er_score = self.score(dev_loader, silent=True)
                # to plot epoch with F1
                dev_F1_list.append(er_score)
                
                if er_score[-1]>0.35 and er_score[-1]>self.best_er_score[-1]:
                    self.best_er_score = er_score
                    print("Saving model with f1={:.4f}".format(self.best_er_score[2]))
                    torch.save(self.state_dict(), save_model)
                            
                            
                                  
                                                          
        # if self.best_er_score[2] > 0:
        #     self.load_state_dict(torch.load(save_model))
        
        if self.mh_attn :
            with open("{}_head_er_scores_{}.{}.{}.{}.{}.txt".format(self.n_r_head, dataset, self.embedding, bi, pos, pw), "wb") as fp:  
                pickle.dump(dev_F1_list, fp)
        
        else:
            with open("no_M-h_Attn_er_scores_{}.{}.{}.{}.{}.txt".format(dataset, self.embedding, bi, pos, pw), "wb") as fp:  
                pickle.dump(dev_F1_list, fp)
        return self

    
    def score(self, loader, isTrueEnt=False, silent=False, rel_detail=False, print_final=True):
        '''
        Compute Precision/Recall/F-1 score of dataset in BIOLoader X
        Input:
            loader: a BIOLoader containing a dataset of sentences.
            isTrueEnt: optional. Boolean to give the ground truth entity to evaluate
            silent: optional. Boolean to suppress detailed decoding
            rel_detail: optional. Boolean to show the each relation's precision, recall and F1 score.
        Output:
            e_score: P/R/F-1 score of entity prediction
            er_score: P/R/F-1 score of entity and relation prediction
        '''

        if rel_detail==True:
            e_score, er_score, all_er_score, acc_zone_block = evaluate_data(self, loader, self.schema, isTrueEnt, 
                                                            silent, rel_detail, print_final)
            return e_score, er_score, all_er_score, acc_zone_block
        
        else:
            e_score, er_score = evaluate_data(self, loader, self.schema, isTrueEnt, silent, 
                                              rel_detail, print_final)
            return e_score, er_score


    def predict(self, loader):

        while True:
            sentence = input('Enter the sentence to predict. Type !quit to break: ')
            if sentence=='!quit':
                break

            embed_input = BIOLoader.get_pretrain_input(loader, _raw_input = sentence).to(self.use_device)
            ent_output, rel_output = self.forward(embed_input, [0], loader)

            len_of_sent = len(sentence)
            e = ent_argmax(ent_output[:len_of_sent]).cpu().numpy()[0]

            predict_ent = [self.schema.ix2ent[i] for i in e]
            pred_ent_list, _ = decode_ent(e, self.schema)

            r = rel_argmax(rel_output[:len_of_sent]).tolist()[0]
            pred_r_list = decode_rel(predict_ent, r, self.schema) 
            pred_rel_list = decode_rel_to_eval(pred_r_list, self.schema, pred_ent_list)

            print()
            print(sentence)
            print('Predict Output')
            print(predict_ent[:len_of_sent])
            print(pred_r_list[:len_of_sent])
            print()
            print('Predict in Triplet')
            print('Entity:   ', pred_ent_list)
            print('Relation: ', pred_rel_list)
            print("=====================================")

    
                      

    
class Relation_Layer(nn.Module):
    def __init__(self, d_v, rel_size, mh_attn, args, position_enc):
        super().__init__()
        
        self.head = args.n_r_head      
        self.d_rel = args.d_rel
        self.pair_out = args.pair_out
        self.rel_dropout = args.rel_dropout
        self.bi_fill = args.bi_fill
        
        self.e_d_attn = args.e_d_attn
        self.mh_attn = mh_attn
        self.rel_size = rel_size

        self.Pw_hint = args.Pw_hint       
        self.P_hint = args.P_hint
        self.pos_strategy = args.pos_strategy
        self.position_enc = position_enc
        self.augment_info = args.augment_info

        
        self.multi_head_attn = Multi_head_Attn(self.d_rel, self.head, d_v, self.rel_dropout, 
                                                             self.bi_fill, self.e_d_attn, self.augment_info)   
        self.layer_norm = nn.LayerNorm(self.d_rel)

        if self.Pw_hint: 
            self.w1 = nn.Linear(self.d_rel, self.pair_out)        
            self.w2 = nn.Linear(self.d_rel, self.pair_out)         
            self.tanh = nn.Tanh()   
            self.v = nn.Linear(self.pair_out, self.rel_size)
            
            nn.init.xavier_normal_(self.w1.weight)
            nn.init.xavier_normal_(self.w2.weight) 
            nn.init.xavier_normal_(self.v.weight)

        else:
            self.no_pw = nn.Linear(self.d_rel, self.rel_size)
            nn.init.xavier_normal_(self.no_pw.weight)

    
        self.softmax = nn.LogSoftmax(dim=-1)
        self.dropout1 = nn.Dropout(args.pair_dropout)
        self.dropout2 = nn.Dropout(args.pair_dropout)
        
        
    def forward(self, encoder_outputs):
        if self.P_hint:
            pos = torch.tensor(list(range(encoder_outputs.size(1))), device=encoder_outputs.device)
            if self.pos_strategy == 'backward':
                encoder_outputs = self.layer_norm(encoder_outputs + self.position_enc(pos).flip(0))
            elif self.pos_strategy == 'forward':
                encoder_outputs = self.layer_norm(encoder_outputs + self.position_enc(pos))
            elif self.pos_strategy == 'minus':
                encoder_outputs = self.layer_norm(encoder_outputs + positional_dist(self.position_enc(pos)))

        else:
        	encoder_outputs = self.layer_norm(encoder_outputs)

        if self.mh_attn:
            encoder_outputs = self.multi_head_attn(encoder_outputs)
                     
        if self.Pw_hint:
            decoder = encoder_outputs[:,-1,:].unsqueeze(1)                        #B x 1 x (d_model+TE) 
            encoder_score = self.dropout1(self.w1(encoder_outputs))               #B x now len x pair_out
            decoder_score = self.dropout2(self.w2(decoder))                       #B x 1 x pair_out
            
            energy = encoder_score+decoder_score       
            energy = self.tanh(energy)                                           #B x now len x pair_out 
            energy = self.v(energy)                                              #B x now len x rel_size
        else:
            energy = self.no_pw(encoder_outputs)
   
        p = self.softmax(energy)                        
        
        return p                                                             #B x now len x rel_size
    


class Multi_head_Attn(nn.Module):
    def __init__(self, d_model, n_head, d_v, attn_dropout=0.1, bi_fill=False, e_d_attn=True, augment_info=False):
        super().__init__()     

        self.n_head = n_head
        self.d_v = d_v
        self.bi_fill = bi_fill   
        self.e_d_attn = e_d_attn
        self.augment_info = augment_info
                
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.w_qs = nn.Linear(d_model, n_head * d_v)
        self.w_ks = nn.Linear(d_model, n_head * d_v)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
             
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(1.0 / (d_model + d_v)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(1.0 / (d_model + d_v)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(1.0 / (d_model + d_v)))

        self.layer_norm = nn.LayerNorm(d_model)



    def forward(self, encoder_outputs):
        d_v, n_head = self.d_v, self.n_head
        temperature = encoder_outputs.size(2)
        temperature = np.power(temperature, 0.5)
        residual = encoder_outputs        
        
        if self.e_d_attn:
            query = encoder_outputs[:,-1,:].unsqueeze(1)       #B x 1 x (d_model)    
        else:
            query = encoder_outputs                        #B x now len x (d_model)   
        
        key = encoder_outputs                              #B x now len x (d_model)
        value = encoder_outputs

        sz_b, len_q, _ = query.size()
        sz_b, len_k, _ = key.size()

        query = self.w_qs(query).view(sz_b, len_q, n_head, d_v)
        key = self.w_ks(key).view(sz_b, len_k, n_head, d_v)
        value = self.w_vs(value).view(sz_b, len_k, n_head, d_v)

        query = query.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_v) # [(n*B) x 1 x dv] or [(n*B) x now len x dv]
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_v) # (n*B) x now len x dv
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_v) # (n*B) x now len x dv

        if self.augment_info:
            key, value = augment_dist_info(key), augment_dist_info(value)


        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / (temperature+1e-5)
        
        attn = self.softmax(attn)       # [(n*B) x 1 x now len] or [(n*B) x now len x now len]
        attn = self.dropout(attn)

        if self.e_d_attn:
            output = attn.transpose(1,2)*value                        # (n*B) x now len x dv
        else:
             output = torch.bmm(attn, value)                           # (n*B) x now len x dv 
        
        output = output.view(n_head, sz_b, len_k, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_k, -1) # B x now len x (n*dv)
        
        output = output + residual         #B x now len x (d_model) 
        output = self.layer_norm(output)
        
        return output    
    

def augment_dist_info(encoder):
    _, nowlen, d_model = encoder.size()
    rate_list = []
    for i in range(nowlen):
        observe_dist = nowlen-i-1
        dist_rate = observe_dist/nowlen + 1
        rate_list.append(dist_rate)

    return (encoder.transpose(1,2)*
        torch.tensor(rate_list, dtype=encoder.dtype, device=encoder.device)).transpose(1,2)

def positional_dist(pos_embd):
    last_pos = pos_embd[-1]
    return last_pos-pos_embd



def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.cuda.FloatTensor(sinusoid_table)


    
    
class EntityNLLLoss(nn.NLLLoss):  
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, outputs, labels):
        loss = super(EntityNLLLoss, self).forward(outputs.transpose(1, 2).unsqueeze(2),
                                                  labels.unsqueeze(1))
        return loss
    

    
    
class RelationNLLLoss(nn.NLLLoss):    
    def __init__(self, weight, base_weight, rel2ix, bi_fill, device):
        self.rel_weight = weight
        self.base_weight = base_weight
        self.rel2ix = rel2ix
        self.bi_fill = bi_fill
        weight = torch.ones(len(self.rel2ix), device=device)*self.rel_weight
        weight[0], weight[1] = self.base_weight, self.base_weight
        super().__init__(reduction='none', weight=weight)
        
        
    def forward(self, outputs, labels):
        loss = super(RelationNLLLoss, self).forward(outputs.permute(0,-1,1,2),labels)
        return mean_sentence_loss(loss, self.bi_fill)


    
    
def mean_sentence_loss(loss, bi_fill):

    batch, length = loss.size(0), loss.size(1)

    if bi_fill:
        num_tokens = torch.ones(batch, length, device=loss.device)*length
    else:
        num_tokens = torch.ones(batch, length, device=loss.device)*\
                     torch.tensor(list(range(1,length+1)), dtype=loss.dtype, device=loss.device)


    # loss.size() torch.Size([32, 110, 110])
    # num_tokens  torch.Size([32, 110])
    # tensor([[  1.,   2.,   3.,  ..., 108., 109., 110.],
    #         [  1.,   2.,   3.,  ..., 108., 109., 110.],....)
    # loss  torch.Size([32, 110])
    
    return loss.sum(dim=-1).div(num_tokens).mean()


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "selu":
        return F.selu    
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/selu/gelu, not %s." % activation)



