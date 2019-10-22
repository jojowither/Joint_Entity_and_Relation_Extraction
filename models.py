import torch
from torch import nn, optim
import torch.nn.functional as F

import time
import numpy as np
import copy
from tqdm import tqdm
import pickle

from evaluation import evaluate_data, decode_ent, decode_rel



class JointERE(nn.Module):
    def __init__(self, pre_model, maxlen,
            d_model, bilstm_n_layers, word_dropout, bilstm_dropout, rel_dropout, d_rel, n_r_head, 
            d_r_v, point_output, schema, args, embedding='BERT_base', mh_attn=True):
        '''
        JointERE
            Joint Entity and Relation mention Extraction on Traditional Chinese text
        Input:
            point_output:
                The column dimension of the value matrix in the pointer network
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
        self.label_embed_dim = self.ent_size                 #LE
        self.point_output = point_output
        self.schema = schema
        self.n_r_head = n_r_head
        self.mh_attn = mh_attn
        self.embedding = embedding
        self.rel_weight = args.rel_weight
        self.rel_weight_base_tag = args.rel_weight_base_tag
        self.scheduler_step = args.scheduler_step
        self.scheduler_gamma = args.scheduler_gamma
    

        self.word_dropout = nn.Dropout(word_dropout)
        self.lstm_dropout = nn.Dropout(bilstm_dropout)  #optional
        
    
        self.bilstm = nn.LSTM(d_model, d_model // 2, num_layers=bilstm_n_layers, 
                              bidirectional=True, batch_first=True, dropout=bilstm_dropout) 

        
    
        self.fc2tag = nn.Linear(d_model+self.label_embed_dim, self.ent_size)
        self.init_linear(self.fc2tag)
        
        
        self.softmax = nn.LogSoftmax(dim=-1)
        self.label_embed = nn.Embedding(self.ent_size, self.label_embed_dim)
        nn.init.orthogonal_(self.label_embed.weight)

        self.t2rel = nn.Linear(2*self.label_embed_dim+d_model, d_rel)
        self.init_linear(self.t2rel)
        
        self.pointer = Token_wise_Pointer_Network(d_rel, point_output, n_r_head, d_r_v, 
                                                  rel_dropout, self.rel_size, self.mh_attn)


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
#                 enc[wp_range_l[0]] = torch.sum(torch.stack([enc[idx] for idx in wp_range_l]), dim=0)
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
                # print(embed_input[0].size())  # torch.Size([32, 192, 768])
                # print(embed_input[1].size())  # torch.Size([32, 768])
                # print(len(embed_input[2]))    # 13
                # print(embed_input[2][0].size())  # torch.Size([32, 192, 768])
                # print(embed_input[2][12].size())
                embed_input = torch.sum(torch.stack([layer for layer in embed_input[2][:-1]]), dim=0)
                # print(embed_input.size())
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
#         enc_output = self.lstm_dropout(enc_output)

        encoder_sequence_l = [] 
        
        label = enc_output.new_zeros(batch_size, self.label_embed_dim)

        for length in range(self.maxlen):
            now_token = enc_output[:,length,:]
            now_token = torch.squeeze(now_token, 1)
            
            combine_pre = torch.cat((label, now_token), 1)                    #B x (LE+d_model)    
            ent_output = self.softmax(self.fc2tag(combine_pre))               #B x es

                       
            # pass the gold entity embedding to the next time step, if available
            if batch_ent is not None:
                label = self.label_embed(batch_ent[:, length])
                
            else:
                label = self.label_embed(ent_output.argmax(-1))                 #B x LE
            
            
            
            combine_entity = torch.cat((combine_pre, label), 1)                    #B x (2*LE+d_model)
            # relation layer
            encoder_sequence_l.append(combine_entity)  
            encoder_sequence = torch.stack(encoder_sequence_l).transpose(0, 1)     #B x len x (2*LE+d_model)     
   
            # Calculate attention weights 
            # point_weights = self.pointer(self.t2rel(encoder_sequence))
            point_weights = self.pointer(F.selu(self.t2rel(encoder_sequence)))
      
    
            entity_tensor[:,length,:] = ent_output            
            rel_tensor[:,length,:length+1,:] = point_weights
 
        
        return entity_tensor, rel_tensor
        
        
    def entity_loss(self):
        return EntityNLLLoss()
    
    def relation_loss(self, device):
        return RelationNLLLoss(self.rel_weight, self.rel_weight_base_tag, self.schema.rel2ix, device)


    
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
        criterion_rel = self.relation_loss(loader.device)
        self.loader = loader
        self.dev_loader = dev_loader
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
                save_model = 'NER_RE_best.{}.{}.{}.{}_fold.pkl'.format(dataset, self.embedding, n_r_head, n_fold)
            else:
                save_model = 'NER_RE_best.{}.{}.{}.pkl'.format(dataset, self.embedding, n_r_head)
        
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

                # batch_loss = batch_loss_ent + self.rel_loss_weight*batch_loss_rel
                batch_loss = batch_loss_ent + batch_loss_rel

                batch_loss.backward()
                
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                # torch.nn.utils.clip_grad_value_(self.parameters(), 1)
                
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
            with open("{}_head_er_scores_{}.txt".format(self.n_r_head, dataset), "wb") as fp:  
                pickle.dump(dev_F1_list, fp)
        
        else:
            with open("no_M-h_Attn_er_scores_{}.txt".format(dataset), "wb") as fp:  
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
    
    

                   

    
class Token_wise_Pointer_Network(nn.Module):
    def __init__(self, attn_input, attn_output, head, d_v, drop_out, rel_size, mh_attn):
        super().__init__()
        
        self.mh_attn = mh_attn
        self.attn_input = attn_input
        self.attn_output = attn_output
        self.rel_size = rel_size

        self.multi_head_attn = Sequence_wise_Multi_head_Attn(self.attn_input, head, d_v, drop_out)   
        
        self.w1 = nn.Linear(self.attn_input, self.attn_output)        
        self.w2 = nn.Linear(self.attn_input, self.attn_output)         
        self.tanh = nn.Tanh()   
        self.v = nn.Linear(self.attn_output, self.rel_size)
        
        nn.init.xavier_normal_(self.w1.weight)
        nn.init.xavier_normal_(self.w2.weight) 
        nn.init.xavier_normal_(self.v.weight)

        
        self.softmax = nn.LogSoftmax(dim=-1)
#         self.layer_norm = nn.LayerNorm(self.attn_output)
#         self.dropout = nn.Dropout(drop_out)
        
        
    def forward(self, encoder_outputs):
        if self.mh_attn:
            encoder_outputs = self.multi_head_attn(encoder_outputs)
        
        decoder = encoder_outputs[:,-1,:].unsqueeze(1)                       #B x 1 x (d_model+LE) 
        encoder_score = self.w1(encoder_outputs)                             #B x now len x POINT_OUT
        decoder_score = self.w2(decoder)                                     #B x 1 x POINT_OUT
        
        energy = encoder_score+decoder_score
        
        energy = self.tanh(energy)                                           #B x now len x POINT_OUT          
        energy = self.v(energy)                                              #B x now len x rel_size
        
        p = self.softmax(energy)                        
        
        return p                                                             #B x now len x rel_size
    


class Sequence_wise_Multi_head_Attn(nn.Module):
    def __init__(self, d_model, n_head, d_v, attn_dropout=0.1):
        super().__init__()     

        self.n_head = n_head
        self.d_v = d_v   
                
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.w_qs = nn.Linear(d_model, n_head * d_v)
        self.w_ks = nn.Linear(d_model, n_head * d_v)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        # nn.init.xavier_normal_(self.w_qs.weight)
        # nn.init.xavier_normal_(self.w_ks.weight)
        # nn.init.xavier_normal_(self.w_vs.weight)        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(1.0 / (d_model + d_v)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(1.0 / (d_model + d_v)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(1.0 / (d_model + d_v)))

        
#         self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, encoder_outputs):
        d_v, n_head = self.d_v, self.n_head
        temperature = encoder_outputs.size(2)
        temperature = np.power(temperature, 0.5)
        residual = encoder_outputs        
        
        query = encoder_outputs[:,-1,:].unsqueeze(1)                       #B x 1 x (d_model)
        key = encoder_outputs                                              #B x now len x (d_model)
        value = encoder_outputs

        sz_b, len_q, _ = query.size()
        sz_b, len_k, _ = key.size()

        query = self.w_qs(query).view(sz_b, len_q, n_head, d_v)
        key = self.w_ks(key).view(sz_b, len_k, n_head, d_v)
        value = self.w_vs(value).view(sz_b, len_k, n_head, d_v)

        query = query.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_v) # (n*B) x 1 x dv
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_v) # (n*B) x now len x dv
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_v) # (n*B) x now len x dv


        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / (temperature+1e-5)
        
        attn = self.softmax(attn)                                          #(n*B) x 1 x now len
        attn = self.dropout(attn)

        output = attn.transpose(1,2)*value                        # (n*B) x now len x dv 
        output = output.view(n_head, sz_b, len_k, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_k, -1) # B x now len x (n*dv)
        
#         output = self.layer_norm(output + residual)                       #B x now len x (d_model) 
        output = output + residual                       #B x now len x (d_model) 
        
        return output    
    
  
    
    
class EntityNLLLoss(nn.NLLLoss):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, outputs, labels):
        loss = super(EntityNLLLoss, self).forward(outputs.transpose(1, 2).unsqueeze(2),
                                                  labels.unsqueeze(1))
        return loss
    

    
    
class RelationNLLLoss(nn.NLLLoss):    
    def __init__(self, weight, base_weight, rel2ix, device):
        self.rel_weight = weight
        self.base_weight = base_weight
        self.rel2ix = rel2ix
        weight = torch.ones(len(self.rel2ix), device=device)*self.rel_weight
        weight[0], weight[1] = self.base_weight, self.base_weight
        super().__init__(reduction='none', weight=weight)
        
        
    def forward(self, outputs, labels):
        loss = super(RelationNLLLoss, self).forward(outputs.permute(0,-1,1,2),labels)

        return mean_sentence_loss(loss)


    
    
def mean_sentence_loss(loss):

    batch, length = loss.size(0), loss.size(1)
    num_tokens = torch.ones(batch, length, device=loss.device)*\
                 torch.tensor(list(range(1,length+1)), dtype=loss.dtype, device=loss.device)
    # num_tokens = loss.norm(0, -1)


    # loss.size() torch.Size([32, 110, 110])
    # num_tokens  torch.Size([32, 110])
    # tensor([[  1.,   2.,   3.,  ..., 108., 109., 110.],
    #         [  1.,   2.,   3.,  ..., 108., 109., 110.],....)
    # loss  torch.Size([32, 110])
    
    return loss.sum(dim=-1).div(num_tokens).mean()



