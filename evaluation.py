import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
import warnings
# import matplotlib.pyplot as plt


def evaluate_data(model, data_loader, schema, isTrueEnt=False, silent=False, rel_detail=False, print_final=True):
    '''
    Evaluate model predictions from data_loader with P/R/F-1 scores
    
    Input:
        model: an instance of models.JointERE
        data_loader: an instance of torch.utils.data.DataLoader
        schema: an instance of data_util.Schema
        isTrueEnt: optional. Boolean to give the ground truth entity to evaluate
        silent: optional. Boolean to suppress detailed decoding
        rel_detail: optional. Boolean to show the each relation's precision, recall and F1 score.
        
    Output:
        e_score: a tuple of P/R/F-1 score of entity prediction
        er_scores: a list of tuples of P/R/F-1 score of entity+relation,
            which has length corresponding to number of threshold
    '''
    
    y_ent_true_all, y_ent_pred_all = [], []
    y_rel_true_all, y_rel_pred_all = [], []
    tps, fps, tns, fns = 0, 0, 0, 0
    
    anay_rel_true = {}
    anay_rel_pred = {}
    anay_rel_pospred = {}
    for i in range(len(schema.Relation_tags)):
        anay_rel_true[i]=[]
        anay_rel_pred[i]=[]
        anay_rel_pospred[i]=[]
        
        
    # for accuracy_for_sentences_of_varying_lengths     
    sent_lens = []
    all_t_r_lists, all_p_r_lists = [], []
    
    
    if silent:
        warnings.simplefilter('ignore')
    else:
        warnings.filterwarnings('always')
        
        
    with torch.no_grad():
        for embed_input, batch_ent, batch_rel, batch_index in data_loader:
            model.eval()
            
            
            if isTrueEnt:
                ent_output, rel_output = model(embed_input, batch_index, data_loader, batch_ent)
            else:
                ent_output, rel_output = model(embed_input, batch_index, data_loader)
            
            

            anay_true, anay_pred, anay_pospred, *(score_num) = batch_decode(ent_output.cpu(), rel_output.cpu(), 
                                                                           batch_index,  data_loader.raw_input, 
                                                                           batch_ent.cpu(), batch_rel.cpu(), 
                                                                           schema, silent=silent, rel_detail=rel_detail)
            
            y_true_ent, y_pred_ent, y_true_rel, y_pred_rel, tp, fp, tn, fn, acc_for_len = score_num
            
            
            
            y_ent_true_all.extend(y_true_ent)
            y_ent_pred_all.extend(y_pred_ent)
            y_rel_true_all.extend(y_true_rel)
            y_rel_pred_all.extend(y_pred_rel)
            
            
            tps += tp
            fps += fp
            tns += tn
            fns += fn           
            
            if rel_detail:
                for r in anay_true:
                    anay_rel_true[r].extend(anay_true[r])
                for r in anay_pred:
                    anay_rel_pred[r].extend(anay_pred[r])
                for r in anay_pospred:
                    anay_rel_pospred[r].extend(anay_pospred[r])
                    
                    
                sent_len, t_r_lists, p_r_lists = acc_for_len
                sent_lens.extend(sent_len)
                all_t_r_lists.extend(t_r_lists)
                all_p_r_lists.extend(p_r_lists)
                            
            
    e_score = precision_recall_fscore_support(y_ent_true_all, y_ent_pred_all, average='micro', 
                                               labels=range(len(schema.Entity_tags)))[:-1]
    
    er_score = precision_recall_fscore_support(y_rel_true_all, y_rel_pred_all, average='micro', 
                                               labels=range(len(schema.Relation_tags)))[:-1]
    
    
    if print_final:
        print()
        print("Entity detection score")
        print("precision  \t recall  \t fbeta_score")
        print("{:.3f} \t\t {:.3f} \t\t {:.3f} \t".format(*e_score))

        print("Entity+Relation detection score ")
        print("precision  \t recall  \t fbeta_score  \t")
        print("{:.3} \t\t {:.3} \t\t {:.3} \t".format(*er_score))

#         print('confusion matrix ')
#         print('TP  \t fp  \t tn  \t fn')
#         print('{:.0f} \t {:.0f} \t {:.0f} \t {:.0f} \t'.format(tps, fps, tns, fns))

        print()
    
        
    if rel_detail==True:        

        all_er_score = precision_recall_fscore_support(y_rel_true_all, y_rel_pred_all, average=None, 
                                           labels=range(len(schema.Relation_tags)))[:-1]

        
        for num_rel in range(len(schema.Relation_tags)):
            
            print('======================================================')
            print('Relation type %s' % (schema.rid2tag[num_rel]))
            print("%s \t %s \t %s \t" % ('precision ', 'recall ', 'fbeta_score '))
            print('%.3f \t\t %.3f \t\t %.3f \t' % (all_er_score[0][num_rel], all_er_score[1][num_rel], all_er_score[2][num_rel]))
            print()
        
        acc_zone_block = accuracy_for_sentences_of_varying_lengths(all_t_r_lists, all_p_r_lists, sent_lens)
            

        return e_score, er_score, all_er_score, acc_zone_block
    
    else:            
        return e_score, er_score
                
                
                
def batch_decode(ent_output, rel_output, batch_index, word_lists, true_ent, true_rel, schema, silent, rel_detail):
    '''
    Aggregate serialized true/predicted class from batch data and model predictions
    
    Input:
        ent_output:
        rel_output:
            raw model predictions
        batch_index:
            current index to the raw input sentences
        wordlists:
            the raw input sentences in characters
        true_ent:
        true_rel:
            ground truth for training model
        schema:
            an instance of data_util.Schema
        silent: optional.
            Boolean to suppress detailed decoding
            
    Output:
         
        anay_true:
            a 2-dimension list with each relation type and the distance of each entity pair in true data.
        anay_pred:
            a 2-dimension list with each relation type and the distance of each entity pair in predict.
        anay_pospred:
            a 2-dimension list with each relation type and the distance of each entity pair 
            in predict and the distance is true.
        rel_error_count:
            the number of error relation
        y_true_ent:
        y_pred_ent:
            a list of char.-offset aligned entity class
        y_true_rel_list:
            a list of entity-pair aligned relation class
        y_pred_rel_list:
            a list of lists of entity-pair aligned relation class,
            predicted with and ordered as in parameter thresholds
        
        tp, fp, tn, fn : the number of true postive, false postive, true negtive, false negtive
        
    '''
    
    true_ent_lists, true_rel_lists = [], []
    pred_ent_lists, pred_rel_lists = [], []   
    sentences_len = []
    

    anay_true, anay_pred, anay_pospred = {}, {}, {}    # add postive relation
    for i in range(len(schema.Relation_tags)):
        anay_true[i]=[]
        anay_pred[i]=[]
        anay_pospred[i]=[]
    
    for e,r,i,te,tr in zip(ent_output, rel_output, batch_index, true_ent, true_rel):
        
        len_of_list = len(word_lists[i])
        word_list = word_lists[i]
        
        te = te[:len_of_list].data.numpy()
        e = ent_argmax(e[:len_of_list]).data.numpy()

        
        true_ent = [schema.ix2ent[i] for i in te]
        predict_ent = [schema.ix2ent[i] for i in e]
        
        
        true_ent_list, _ = decode_ent(te, schema)
        pred_ent_list, _ = decode_ent(e, schema)
        
                    
        tr = tr.tolist()
        r = rel_argmax(r[:len_of_list]).tolist()
        
                
        true_r_list = decode_rel(true_ent, tr, schema)  
        pred_r_list = decode_rel(predict_ent, r, schema)      
        
        
        true_rel_list = decode_rel_to_eval(true_r_list, schema, true_ent_list)
        pred_rel_list = decode_rel_to_eval(pred_r_list, schema, pred_ent_list)
        

        
        true_ent_lists.append(true_ent_list)
        pred_ent_lists.append(pred_ent_list)
        true_rel_lists.append(true_rel_list)
        pred_rel_lists.append(pred_rel_list)
        
        sentences_len.append(len(word_list))

        
        if not silent:
            print(word_list)
            print(true_ent)
            print(true_r_list)
            print()
            print('Predict output')
            print(predict_ent)
            print(pred_r_list)
            print()
            print('True')
            print(true_ent_list)
            print(true_rel_list)
            print('predict')
            print(pred_ent_list)
            print(pred_rel_list)
            print("=====================================")
            
        
        if rel_detail:
            postive_predict_rel = list(set(true_rel_list).intersection(pred_rel_list))
            
            analyze_dict_true = calculate_distance(true_rel_list)   
            analyze_dict_pred = calculate_distance(pred_rel_list) 
            analyze_dict_pos_pred = calculate_distance(postive_predict_rel) 
            
            for r in analyze_dict_true:
                anay_true[r].extend(analyze_dict_true[r])
            for r in analyze_dict_pred:
                anay_pred[r].extend(analyze_dict_pred[r])
            for r in analyze_dict_pos_pred:
                anay_pospred[r].extend(analyze_dict_pos_pred[r])
                        
        
    ent_score, y_true_ent, y_pred_ent = get_scores(true_ent_lists, pred_ent_lists, 
                                                    range(len(schema.Entity_tags)), output_y=True)
    
    rel_score, y_true_rel, y_pred_rel = get_scores(true_rel_lists, pred_rel_lists, 
                                                    range(len(schema.Relation_tags)), output_y=True)
        
    
    tp, fp, tn, fn = relation_error_analysis(true_rel_lists, pred_rel_lists)
    
    if not silent:
        print('Batch entity score')
        print("%s \t %s \t %s \t" % ('precision ', 'recall ', 'fbeta_score '))
        print(ent_score)
        print()
        print('Batch relation score')
        print("%s \t %s \t %s \t" % ('precision ', 'recall ', 'fbeta_score '))
        print(rel_score)
        print()
        print('p_r_fscore')
        print("%s \t %s \t %s \t" % ('precision ', 'recall ', 'fbeta_score '))
        print(p_r_fscore(tp, fp, tn, fn), tp, fp, tn, fn)
        print('===========================================') 
    
    
    return anay_true, anay_pred, anay_pospred, y_true_ent, y_pred_ent, y_true_rel, y_pred_rel, tp, fp, tn, fn,\
            (sentences_len, true_rel_lists, pred_rel_lists)
            



def ent_argmax(output):
    return output.argmax(-1)
                
def rel_argmax(output):
    output = output.argmax(-1)
    return output              


    
# ==================================================
# Decoding utilities
# ==================================================
   
def decode_ent(ent_output, schema):
    '''
    Aggregate entities from predicted tags
    Input:
    pred_ent=a list of entity tags in a sentence
    schema=the dictionary defining entities and relations
    Output: 
    ent_list=[(ent_start, ent_end, ent_type=eid_in_schema)]
    err_count=the number of bad tags
    '''
    
    ent_list = []
    ent_start = 0
    ent_end = 0
    state = {
        'ENT_SPAN': 0,
        'NO_ENT': 1
    }
    err_count = 0
    
    
    ent_type = ''
    sid = state['NO_ENT']
    
    for idx, e_idx in enumerate(ent_output):
        bilou = schema.ix2ent[e_idx].split('-')[0]
        e_tag = schema.ix2ent[e_idx].split('-')[-1]

        
        if sid == state['NO_ENT']:
            if bilou == 'B':
                ent_start = idx
                ent_type = schema.tag2eid[e_tag]
                sid = state['ENT_SPAN']
                
            elif bilou == 'I':
                err_count += 1

            elif bilou == 'U':
                ent_start = idx
                ent_type = schema.tag2eid[e_tag]
                ent_list.append((ent_start, ent_start, ent_type))

                
        elif sid == state['ENT_SPAN']:
            if bilou == 'L':
                ent_end = idx 
                ent_list.append((ent_start, ent_end, ent_type))
                sid = state['NO_ENT']
        
    return ent_list, err_count



def decode_rel(ent_output, rel_output, schema):
    r_list = ['']*len(ent_output)
    num_reocrd = -1       
    
    for now in range(len(rel_output)):
        for loc, rel in enumerate(rel_output[now][:now+1]):
            
            if rel!=schema.rel2ix['Rel-None'] and rel!=schema.rel2ix['Rel-Pad']:
                

                tag = schema.ix2rel[rel]
                r_type, AorB = tag.split('#')
                num_reocrd+=1
                
                if AorB[0]=='A':
                    nowAorB = 'A'
                    preAorB = 'B'
                else:
                    nowAorB = 'B'
                    preAorB = 'A'
                                
                if r_list[loc]=='':
                    r_list[loc] = []
                if r_list[now]=='':
                    r_list[now] = []
                    
                pre_complete_rel = r_type+'-'+str(num_reocrd)+'-'+preAorB
                now_complete_rel = r_type+'-'+str(num_reocrd)+'-'+nowAorB
                
                r_list[loc].append(pre_complete_rel)
                r_list[now].append(now_complete_rel)
                
    return r_list
    
    



                
def decode_rel_to_eval(r_list, schema, ent_list):
    
    pair_idx = {}
    for i, r in enumerate(r_list):
        if type(r) is list:
            for single_r in r:
                tag, pair, AorB = single_r.split('-')
                if pair not in pair_idx:
                    pair_idx[pair] = [i]
                else:
                    pair_idx[pair].append(i)
                    pair_idx[pair].append(tag)
                    
    
    
    eval_rel_list = []
    ent_last_idx_list = [e[1] for e in ent_list]
    for pair in sorted(pair_idx.keys()):
        ent1, ent2, r_type = pair_idx[pair]
        
        try:
            ent1idx = ent_last_idx_list.index(ent1)
        except:
            continue
        
        try:
            ent2idx = ent_last_idx_list.index(ent2)
        except:
            continue
        else:
            r_type = schema.tag2rid[r_type]       
            rel_boundary = [ent_list[ent1idx], ent_list[ent2idx], r_type]       
            eval_rel_list.append(tuple(rel_boundary))
        
    return eval_rel_list




# ==================================================
# Scorers
# ==================================================
        
        
def get_scores(true_lists, pred_lists, labels, output_y=False):
    y_true, y_pred = [], []
    for t_list, p_list in zip(true_lists, pred_lists):
        yt, yp = align_yt_yp(t_list, p_list, labels)
        y_true.extend(yt)
        y_pred.extend(yp)
        
    scores = precision_recall_fscore_support(y_true, y_pred, average='micro', labels=labels)
    return scores, y_true, y_pred if output_y else scores

def align_yt_yp(truths, predictions, labels):
    '''
    Input:
        truths/predictions: list of true and predicted tuples, 
        with the leading entries as the structure and the last entry as the class,
        e.g., [(e1, e2, rel), ...]
        labels: sequence of valid class
    Output:
        yt: list of true class given a structure
        yp: list of predicted class given a structure
    '''
    yt, yp = [], []
    _ID_NONE = len(labels)
    true_dict = { t[:-1]: t[-1] for t in truths }
    for p in predictions:
        yt.append(true_dict.pop(p[:-1], _ID_NONE))
        yp.append(p[-1])
    for target in true_dict.values():
        yt.append(target)
        yp.append(_ID_NONE)

    return yt, yp



def is_neg_triple(t):
    return np.imag(t[-1]) > 0

def negate_triple(t):
    # Mark negative triples with imaginary relation id
    return (t[0], t[1], np.real(t[-1]).item() + 1j)

def posit_triple(t):
    return (t[0], t[1], np.real(t[-1]).item())

def has_edge(base_ptrs, rel, e):
    '''
    Assume a relation exist between an entity pair, 
    if all the tokens in the base entity point to those in entity e.
    '''
    tok_has_ptr_to_e = [tok_ptrs[rel].ge(e[0]).dot(tok_ptrs[rel].le(e[1])).item() > 0 
                        for tok_ptrs in base_ptrs]
    return len(tok_has_ptr_to_e) > 0 and all(tok_has_ptr_to_e)


def relation_error_analysis(true_rel_lists, rel_lists):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i, r_list in enumerate(rel_lists):
        true_pos = len([t for t in r_list if t in true_rel_lists[i]])
        all_true = len([t for t in true_rel_lists[i] if not is_neg_triple(t)])
        all_pos = len(r_list)
        tp += true_pos
        fn += all_true - true_pos
        fp += all_pos - true_pos
        tn += len([t for t in true_rel_lists[i] if is_neg_triple(t) and posit_triple(t) not in r_list])
    return tp, fp, tn, fn

def p_r_fscore(tp, fp, tn, fn, beta=1, eps=1e-8):
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f_beta = (1 + beta**2) * ((p * r) / (((beta**2) * p) + r + eps))
    return p, r, f_beta


def check_every_rel(rel_true, rel_pred, none_rel):
    
    check_true = []
    check_pred = []
    for i in range(none_rel):
        check_true.append([])
        check_pred.append([])
   
    for r_t,r_p in zip(rel_true, rel_pred):
        if r_t==r_p:
            check_true[r_t].append(r_t)
            check_pred[r_p].append(r_p)
        
        elif r_t==none_rel:
            check_true[r_p].append(r_t)
            check_pred[r_p].append(r_p)
        
        elif r_p==none_rel:
            check_true[r_t].append(r_t)
            check_pred[r_t].append(r_p)
    
# #         See the results of the relation classification
# #         _ID_NONE appears above is a false positive
# #         _ID_NONE appears below is a false negative
#     print(check_true)
#     print(check_pred)

    
    return check_true, check_pred
        

def show_every_rel_score(check_true, check_pred, schema):
    
    for num_rel in range(len(check_true)):
        each_scores = precision_recall_fscore_support(check_true[num_rel], \
                    check_pred[num_rel], average='micro', labels = range(len(schema['relation'])))
        
        
        print()
        print('======================================================')
        print('Relation type %d' % (num_rel))
        print("%s \t %s \t %s \t" % ('precision ', 'recall ', 'fbeta_score '))
        print('%.3f \t\t %.3f \t\t %.3f \t' % (each_scores[0], each_scores[1], each_scores[2]))
        print()


        
# ===================================================================        
        
        
def analyze_loader(data_loader, schema, silent=False):
    
    all_rel = {}
    for i in range(len(schema['relation'])):
        all_rel[i]=[]
    
    for batch_x, batch_ent, batch_rel, batch_index in data_loader:
        batch_rel, rel_error_count = analyze_batch(batch_index, data_loader.raw_input,batch_ent.cpu(), 
                                                  batch_rel.cpu(), schema, silent=silent)
        
        for r in batch_rel:
            all_rel[r].extend(batch_rel[r])
    
    
    return all_rel
        
        
        
def analyze_batch(batch_index, word_lists, true_ent, true_rel, schema, silent):
    
    rel_error_count=0
    batch_rel = {}
    for i in range(len(schema['relation'])):
        batch_rel[i]=[]
  
    
    for i,te,tr in zip(batch_index, true_ent, true_rel):

        len_of_list = len(word_lists[i])
        word_list = word_lists[i]
        
        true_ent = [schema.ent2ix.inv(i) for i in te[:len_of_list]]     
        true_ent_list, _ = decode_ent(te[:len_of_list], schema)   
        true_r_list, appear_error = decode_rel(true_ent, tr, schema)  
        
        if appear_error:
            rel_error_count+=1
            continue
            
        true_r_list = [list(set(i)) if type(i) is list else i for i in true_r_list]
        true_r_list = true_r_list[:len_of_list]
        true_rel_list = decode_rel_to_eval(true_r_list, schema, true_ent_list)
     
        
        if not silent:
            print(word_list)
            print(true_ent)
            print(true_r_list)
            print()

            print()
            print('True')
            print(true_ent_list)
            print(true_rel_list)
            
            print("=====================================")
            
        analyze_dict = calculate_distance(true_rel_list)    
        for r in analyze_dict:
            batch_rel[r].extend(analyze_dict[r])
            
    return batch_rel, rel_error_count
        
        
# ===================================================================          
        
            
            
def calculate_distance(true_rel_list):
    analyze_dict = {}
    
    for r_triplet in true_rel_list:
        distant = r_triplet[1][0] - r_triplet[0][0]
        r_type = r_triplet[2]
        
        if r_type in analyze_dict:
            analyze_dict[r_type].append(distant)
        else:
            analyze_dict[r_type] = [distant]
        
    
    return analyze_dict      
        
        
        



def accuracy_for_sentences_of_varying_lengths(all_t_r_lists, all_p_r_lists, sent_lens):
    
    true_zone_block = {10:0, 20:0, 30:0, 40:0, 50:0, '>50':0}
    all_zone_block = {10:0, 20:0, 30:0, 40:0, 50:0, '>50':0}
    acc_zone_block = {10:0, 20:0, 30:0, 40:0, 50:0, '>50':0}
    
    for t_r, p_r, s_len in zip(all_t_r_lists, all_p_r_lists, sent_lens):
        if s_len<=10:
            all_zone_block[10]+=1
            if t_r==p_r:
                true_zone_block[10]+=1
                
        elif s_len<=20:
            all_zone_block[20]+=1
            if t_r==p_r:
                true_zone_block[20]+=1
        
        elif s_len<=30:
            all_zone_block[30]+=1
            if t_r==p_r:
                true_zone_block[30]+=1
                
        elif s_len<=40:
            all_zone_block[40]+=1
            if t_r==p_r:
                true_zone_block[40]+=1

        elif s_len<=50:
            all_zone_block[50]+=1
            if t_r==p_r:
                true_zone_block[50]+=1
                
        else:
            all_zone_block['>50']+=1
            if t_r==p_r:
                true_zone_block['>50']+=1

    for k in acc_zone_block.keys():
        acc_zone_block[k] = true_zone_block[k]/all_zone_block[k]
        
    
    return acc_zone_block
        
        
    
