# Multi-head Attention with Hint Mechanisms for Joint Extraction of Entity and Relation

---

## Requirements
- Python 3.7.3
- pytorch 1.2.0
- transformers 2.3.0
- numpy 1.16.2
- scikit-learn 0.20.3
- tqdm 4.31.1


---

## Task
Given a raw text, (i) give the entity tag of each word (e.g., NER) and (ii) the relations between the entities in the sentence. 

---

## Data
- Conll04 https://cogcomp.seas.upenn.edu/page/resource_view/43
- ADE https://sites.google.com/site/adecorpus/home/document
- ACE04 https://catalog.ldc.upenn.edu/LDC2005T09
- ACE05 https://catalog.ldc.upenn.edu/LDC2006T06
 
Download data and put them in their respective folders in the path ```/data/ ```.

---

## Preprocessing
Preprocess the data to the form the model can train.

```
python3 process_conll04.py
python3 process_ADE.py
python3 process_ACE04.py
python3 process_ACE05.py
```

---

## Train
For each dataset

```
python3 main.py --CUDA_device 0 --dataset conll04 --word_dropout 0.25 
python3 main.py --CUDA_device 0 --dataset ADE --n_r_head 16 --n_iter 100 --scheduler_step 15
python3 main.py --CUDA_device 0 --dataset ACE04 --n_iter 125 --scheduler_step 20
python3 main.py --CUDA_device 0 --dataset ACE05 --batch_size 30 --word_dropout 0.25 --rel_dropout 0.15 --pair_dropout 0.15 --pair_out 400 --n_iter 150 --scheduler_step 20
```

---
## Eval
For each dataset, change parameters ```--dataset``` and ```--model_dict```.

```
python3 main.py --CUDA_device 0 --train_eval_predict eval --dataset conll04 --silent False --model_dict NER_RE_best.conll04.XLNet_base.32.nobi.backward.Pw_hint.pkl
```

---

## Predict
After training, give any sentence and predict the NER and RE.

For each dataset, change parameters ```--dataset``` and ```--model_dict```.

```
python3 main.py --CUDA_device 0 --train_eval_predict predict --dataset conll04 --silent False --model_dict NER_RE_best.conll04.XLNet_base.32.nobi.backward.Pw_hint.pkl
```

---

## Notes
Please cite our work when using this software.
