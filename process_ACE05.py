import xml.etree.ElementTree as ET
import sys, re
import more_itertools as mit
import os


def get_filename(filelist_path):
    file_name = []
    with open(filelist_path, "r", encoding="utf-8") as f:
        content = f.read().splitlines()    
    for c in content[1:-2]:
        if 'timex2norm' in c.split('\t')[1]:
            file_name.append(c.split('\t')[0])
    
    return file_name



def get_apf_and_sgm(data_root):
    apf_file = []
    sgm_file = []
    filename_list = []
    
    removed_subset = ['cts', 'un']
    dirs = os.listdir(data_root)
    
    for d in dirs:
        if d in removed_subset:
            continue
        subset_path = os.path.join(data_root, d)
        timex2norm_path = os.path.join(subset_path, 'timex2norm')
        for f in os.listdir(timex2norm_path):
            file_path = os.path.join(timex2norm_path, f)
            if file_path.endswith('apf.xml'):
                apf_file.append(file_path)

            if file_path.endswith('.sgm'):
                sgm_file.append(file_path)
        
        filelist_path = os.path.join(subset_path, 'FileList')
        file_name = get_filename(filelist_path)
        filename_list.extend(file_name)
                               
    return apf_file, sgm_file, filename_list


def sort_apf_and_sgm(filename_list, apf_file, sgm_file):
    apf_file_sort = []
    sgm_file_sort = []

    for file in filename_list:
        for apf, sgm in zip(apf_file, sgm_file):
            if file==apf.split('/')[-1].replace('.apf.xml', ''):
                apf_file_sort.append(apf)
            if file==sgm.split('/')[-1].replace('.sgm', ''):
                sgm_file_sort.append(sgm)
                                
    return apf_file_sort, sgm_file_sort



def process_apf(apf_file):
    apf_tree = ET.parse(apf_file)
    apf_root = apf_tree.getroot()

    named_entities = {}
    check_nes = {}
    ne_starts={}
    ne_ends={}
    ne_map = {}
    for entity in apf_root.iter('entity'):
        ne_type = entity.attrib["TYPE"]
        for mention in entity.iter('entity_mention'):
            ne_id = mention.attrib["ID"]
            for child in mention:
                if child.tag == 'head':
                    for charseq in child:
                        start = int(charseq.attrib["START"])
                        end = int(charseq.attrib["END"])+1
                        text = re.sub(r"\n", r" ", charseq.text)
                        ne_tuple = (ne_type, start, end, text)
                        if ne_tuple in check_nes:
                            sys.stderr.write("duplicated entity %s\n" % (ne_id))
                            ne_map[ne_id] = check_nes[ne_tuple]
                            continue
                        check_nes[ne_tuple] = ne_id
                        named_entities[ne_id] = [ne_id, ne_type, start, end, text]
                        if not start in ne_starts:
                            ne_starts[start] = []
                        ne_starts[start].append(ne_id)
                        if not end in ne_ends:
                            ne_ends[end] = []
                        ne_ends[end].append(ne_id)

    rels = {}
    check_rels = []
    for relation in apf_root.iter('relation'):
        rel_type = relation.attrib["TYPE"]
        for mention in relation.iter('relation_mention'):
            rel_id = mention.attrib["ID"]
            rel = [rel_id, rel_type, "", ""]
            ignore = False
            for arg in mention.iter('relation_mention_argument'):
                arg_id = arg.attrib["REFID"]
                if arg.attrib["ROLE"] != "Arg-1" and arg.attrib["ROLE"] != "Arg-2":
                    continue
                if arg_id in ne_map:
                    arg_id = ne_map[arg_id]
                rel[int(arg.attrib["ROLE"][-1])+1] = arg_id
                if not arg_id in named_entities:
                    ignore = True
                    # ignored duplicated entity
            if ignore:
                sys.stderr.write("ignored relation %s\n" % (rel_id))
                continue
            if rel[1:] in check_rels:
                sys.stderr.write("duplicated relation %s\n" % (rel_id))
                continue
            check_rels.append(rel[1:])
            rels[rel_id] = rel
            
    return named_entities, rels, ne_starts, ne_ends


def process_doc(doc, named_entities, ne_starts, ne_ends):
    doc = open(doc).read()
    doc = re.sub(r"<[^>]+>", "", doc)
    doc = re.sub(r"(\S+)\n(\S[^:])", r"\1 \2", doc)

    offset = 0
    size = len(doc)
    current = 0
    regions = []
    for i in range(size):
        if i in ne_starts or i in ne_ends:
            inc = 0
            if (doc[i-1] != " " and doc[i-1] != "\n") and (doc[i] != " " and doc[i] != "\n"):
                regions.append(doc[current:i])
                inc = 1
                current = i
            if i in ne_starts:
                for ent in ne_starts[i]:
                    named_entities[ent][2] += offset + inc
            if i in ne_ends:
                for ent in ne_ends[i]:
                    named_entities[ent][3] += offset
            offset+=inc
    regions.append(doc[current:])
    doc = " ".join(regions)

    for ne in named_entities.values():
        if "\n" in doc[int(ne[2]):int(ne[3])]:
            l = []
            l.append(doc[0:int(ne[2])])
            l.append(doc[int(ne[2]):int(ne[3])].replace("\n", " "))
            l.append(doc[int(ne[3]):])
            doc = "".join(l)

    for ne in named_entities.values():
        assert doc[int(ne[2]):int(ne[3])].replace("&AMP;", "&").replace("&amp;", "&").replace(" ", "") == \
        ne[4].replace(" ",""), "%s <=> %s" % (doc[int(ne[2]):int(ne[3])], ne[4])
        
    return doc


def create_entity_seqs(doc_len, named_entities):
    ent_seqs = ['O']*doc_len
    
    for ne in named_entities.values():
        e_type = ne[1]
        start_idx = ne[2]
        end_idx = ne[3]        
        words = ne[4]
        
        words = words.split(' ')
        if len(words)==1:
            for i in range(start_idx, end_idx):
                ent_seqs[i] = 'U-'+e_type
                
        else:
            start_range = len(words[0])
            end_range = len(words[-1])
            for i in range(start_idx, start_idx+start_range):
                ent_seqs[i] = 'B-'+e_type
            for i in range(start_idx+start_range, end_idx-end_range):
                ent_seqs[i] = 'I-'+e_type
            for i in range(end_idx-end_range, end_idx):
                ent_seqs[i] = 'L-'+e_type    
            
        
    return ent_seqs


def create_relation_seqs(doc_len, named_entities, rels):
    rel_seqs = ['']*doc_len
    rel_count = 0
    
    for r in rels.values():
        r_type = r[1].replace('-', '_')
        arg1 = r[2]
        arg2 = r[3]
        
        arg1_ent_start = named_entities[arg1][2]
        arg1_ent_end = named_entities[arg1][3]
        arg2_ent_start = named_entities[arg2][2]
        arg2_ent_end = named_entities[arg2][3]
        
        rel_seqs = rel_append_type(rel_seqs, r_type, rel_count, 'A', arg1_ent_start, arg1_ent_end)
        rel_seqs = rel_append_type(rel_seqs, r_type, rel_count, 'B', arg2_ent_start, arg2_ent_end)
        
        rel_count+=1
        
    return rel_seqs
        
            
            
def rel_append_type(rel_seqs, r_type, rel_count, AorB, ent_start, ent_end):
    for i in range(ent_start, ent_end):
        if rel_seqs[i]=='':
            rel_seqs[i] = []
            rel_seqs[i].append(r_type+'-'+str(rel_count)+'-'+AorB)
        else:
            rel_seqs[i].append(r_type+'-'+str(rel_count)+'-'+AorB)
            
    return rel_seqs


def process_char2word(doc, ent_seqs, rel_seqs):
    sentences = doc.split('\n')
    sentence_list = []
    start_idx = 0
    end_idx = 0
    word_range = []
    in_word = False
    
    newline_list = list(mit.locate(doc, lambda x: x == "\n"))
    space_list = list(mit.locate(doc, lambda x: x == " "))
    
    newline_and_space_list = sorted(newline_list+space_list)
    
    for s in sentences:
        s = s.strip()
        sentence_list.append(s.split(' '))
    sentence_list = list(filter(lambda x: x!= [''], sentence_list))
    
    remove_space_list = []
    for sent in sentence_list:
        sent = list(filter(lambda x: x!= '', sent))
        for i,s in enumerate(sent):
            if len(s.split())>1:
                sent[i] = s.split()[0]
        remove_space_list.append(sent)
        
    
    
    for i in range(len(doc)):
        if i not in newline_and_space_list and in_word==False:
            start_idx = i
            in_word = True
        
        elif i in newline_and_space_list and in_word==True:
            end_idx = i
            in_word = False
            word_range.append((start_idx, end_idx))
    
    
    ent_lists = []
    rel_lists = []
    for wr in word_range:
        ent_lists.append(ent_seqs[wr[0]])
        if ent_seqs[wr[0]]!='':
            if ent_seqs[wr[0]][0]=='L' or ent_seqs[wr[0]][0]=='U':
                rel_lists.append(rel_seqs[wr[0]])
            else:
                rel_lists.append('')
        else:
            rel_lists.append('')
    
    
    return remove_space_list, ent_lists, rel_lists


def combine_sentence_ent_rel(sentence_lists, ent_lists, rel_lists):
    combine_input = []
    idx = 0
       
    for sentence in sentence_lists:
        data_represent = ''
        for s in sentence:
            data_represent += s+' '+ent_lists[idx]+' '+' '.join(rel_lists[idx])+'\n'                
            idx+=1
        combine_input.append(data_represent)
        
    return combine_input



def check_doc_type(apf_file_name, combine_input):
            
    if apf_file_name[:7] == 'CNN_ENG' or apf_file_name[:9] == 'CNNHL_ENG':
        return combine_input[3:-1]
    else:
        return combine_input[3:]


ACE05_root = 'data/ACE05/'
process_path = os.path.join(ACE05_root, 'process_data')
os.makedirs(process_path, exist_ok=True)    

data_root = os.path.join(ACE05_root, 'data/English')
apf_file, sgm_file, filename_list = get_apf_and_sgm(data_root)

apf_files, sgm_files = sort_apf_and_sgm(filename_list, apf_file, sgm_file)


for apf_file, doc in zip(apf_files, sgm_files):
    named_entities, rels, ne_starts, ne_ends = process_apf(apf_file)
    doc = process_doc(doc, named_entities, ne_starts, ne_ends)
    
#     for ne in named_entities.values():
#         print("T%s\t%s %d %d\t%s" % tuple(ne))

#     for rel in rels.values():
#         print("R%s\t%s Arg1:T%s Arg2:T%s" % tuple(rel))

    ent_seqs = create_entity_seqs(len(doc), named_entities)
    rel_seqs = create_relation_seqs(len(doc), named_entities, rels)
    
    sentence_lists, ent_lists, rel_lists = process_char2word(doc, ent_seqs, rel_seqs)
    combine_input = combine_sentence_ent_rel(sentence_lists, ent_lists, rel_lists)
    
    apf_file_name = apf_file.split('/')[-1].replace('.apf.xml', '')
    combine_input = check_doc_type(apf_file_name, combine_input)
    
    with open(os.path.join(process_path, apf_file_name+'.txt'), "w") as f:
        for item in combine_input:
            f.write("%s\n" % item)