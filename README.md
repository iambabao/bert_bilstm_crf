# BERT BiLSTM CRF

## Data Prepare
- Download pretrained model `cased_L-12_H-768_A-12` from [here](https://github.com/google-research/bert), put it in `bert_bilstm_crf/`.
- prepare your data like [data_raw.example.json](./data/data_raw.example.json), and name it as `data_raw.json`. One json per line, data in `argument` is what you want to annotate.
- your NER tags should be in `dict_tag.json`, the format is like [dict_tag.example.json](./data/dict_tag.example.json).

run
```bash
python preprocess.py
```
to generate `data_train.json`, `data_valid.json`, `data_test.json` and `dict_pos.json` from `raw_data.json`.

## Train
run
```bash
python run.py -m bert_ner --mode train --batch 16 --epoch 10 --lr 3e-5
```

model and valid results will be stored in `/result/bert_ner/`.

## Test
run
```bash
python run.py -m bert_ner --mode test
```

test results will be stored in `/result/bert_ner/`.

## Inference
prepare your data like [input.example.txt](./data/input.example.txt), and name it as `input.txt`.

run
```bash
python inference.py -m bert_ner -i ./data/input_txt -o ./data/output.json
```

## ENV
main environment

```text
Package              Version  
-------------------- ---------
bottle               0.12.17  
gensim               3.4.0    
jieba                0.39     
joblib               0.13.2   
nltk                 3.4.5    
numpy                1.16.4   
scikit-learn         0.21.2   
scipy                1.3.1    
tensorboard          1.14.0   
tensorflow           1.14.0   
tensorflow-estimator 1.14.0   
```
