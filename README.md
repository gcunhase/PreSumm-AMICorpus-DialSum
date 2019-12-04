## About

#### Disclaimer
The PreSumm model, presented in the EMNLP 2019 paper titled "[Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)" [[original code](https://www.github.com/nlpyang/PreSumm)], is not my work. Please credit the appropriate authors for that model.

#### Purpose of this repository
* Need to use PreSumm as baseline model for comparison with a custom dataset
* Using the pre-trained model `BertExtAbs`, fine-tune PreSumm with the custom dataset 

#### Dataset
[AMI DialSum Meeting Corpus](gihub.com/MiuLab/DialSum) [[paper](arxiv.org/abs/1809.05715)]

#### My modifications
* Data pre-processing
    * `src/ami_dialsum_corpus_story.py`: script to format the AMI DialSum Corpus into the same CNN/DM `.story` format
    * In `src/prepo/data_builder.py`: added function `format_to_lines_amidialsum()` for use in *'Step 4'* of pre-processing the custom data
        * Needed because the custom data is already split into train/test/dev and there was no function to deal with this situation
    * In `src/preprocess.py`: addition of `-mode format_to_lines_amidialsum`
    
* Fine-tuning PreSumm
    * Load the [BertExtAbs pre-trained weights](https://drive.google.com/open?id=1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr) as `-load_from_extractive ../models/bertsumextabs_cnndm_final_model_step_148000.pt`
 
* Notes on extra steps needed to run PyRouge 

## Requirements
Python 3.5.2

```
pip install -r requirements.txt
```

#### PyRouge notes
* Clone ROUGE-1.5.5 perl package
```
cd src
git clone https://github.com/andersjo/pyrouge
cp -r pyrouge/tools ../tools
rm -r pyrouge
```

* Clone `pyrouge` (Rouge Python package):
```
git clone https://github.com/bheinzerling/pyrouge
mv tools pyrouge/
cd pyrouge
python setup.py install
pyrouge_set_rouge_path '/mnt/gwena/Gwena/PreSumm/src/pyrouge/tools/ROUGE-1.5.5/'
python -m pyrouge.test
```

* If `python -m pyrouge.test` encounters an error, the problem might be with the wordnet database. To fix:
```
cd pyrouge/tools/ROUGE-1.5.5/data/
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExceptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
```
> Source: github.com/tagucci/pythonrouge/issues/4

## Data Preparation For AMI Meeting Corpus
### Option 1: download the processed AMI Meeting data
* Download [here](): unzip the zipfile and put all `.pt` files into `bert_data`

### Option 2: process the data yourself
> Follow pre-processing steps in nlpyang's [original README file](https://www.github.com/nlpyang/PreSumm)

#### Steps modified for AMI Meeting DialSum Corpus:
* Step 0: [Download AMI DialSum Corpus](https://github.com/MiuLab/DialSum)

* Steps 1 and 2 are the same

* Step 3. Sentence Splitting and Tokenization

```
python preprocess.py -mode tokenize -raw_path ../raw_data/ami_dialsum_corpus_stories/train -save_path ../raw_data/ami_dialsum_corpus_tokenized/train -log_file ../logs/ami_dialsum_corpus_train.log
python preprocess.py -mode tokenize -raw_path ../raw_data/ami_dialsum_corpus_stories/test -save_path ../raw_data/ami_dialsum_corpus_tokenized/test -log_file ../logs/ami_dialsum_corpus_test.log
python preprocess.py -mode tokenize -raw_path ../raw_data/ami_dialsum_corpus_stories/valid -save_path ../raw_data/ami_dialsum_corpus_tokenized/valid -log_file ../logs/ami_dialsum_corpus_valid.log
```

* Step 4. Format to Simpler Json Files
    * AMI Corpus doesn't have `-map_path` for `train, val, test`
    * Modification: `-mode format_to_lines` changed to `-mode format_to_lines_amidialsum`

```
python preprocess.py -mode format_to_lines_amidialsum -raw_path ../raw_data/ami_dialsum_corpus_tokenized -save_path ../json_data/ami_dialsum_corpus -n_cpus 1 -use_bert_basic_tokenizer false -log_file ../logs/ami_dialsum_corpus.log
```

* Step 5. Format to PyTorch Files: make dir `../bert_data/ami_dialsum_corpus_bin` and run the following
```
python preprocess.py -mode format_to_bert -raw_path ../json_data -save_path ../bert_data/ami_dialsum_corpus_bin  -lower -n_cpus 1 -log_file ../logs/preprocess.log
```

## Model Training

* Download best performing model with PreSumm: [CNN/DM BertExtAbs](https://drive.google.com/open?id=1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr)

* **First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

### Abstractive Setting with BertExtAbs
```
python train.py -task abs -mode train -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -dec_dropout 0.2 -model_path ../models/amidialsum_model -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 200 -batch_size 140 -train_steps 2000 -report_every 10 -accum_count 2 -use_bert_emb true -use_interval true -warmup_steps_bert 200 -warmup_steps_dec 100 -max_pos 512 -visible_gpus 2,3,4,5 -log_file ../logs/abs_bert_amidialsum  -load_from_extractive ../models/bertsumextabs_cnndm_final_model_step_148000.pt   

python train.py -task abs -mode train -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -dec_dropout 0.2 -model_path ../models/amidialsum_model_longer -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 20000 -batch_size 140 -train_steps 200000 -report_every 100 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 2000 -warmup_steps_dec 1000 -max_pos 512 -visible_gpus 2,3,4,5 -log_file ../logs/abs_bert_amidialsum_longer  -load_from_extractive ../models/bertsumextabs_cnndm_final_model_step_148000.pt   
```
> Original: -train_steps 200000 -warmup_steps_bert 20000 -warmup_steps_dec 10000


## Model Evaluation
### AMI DialSum Corpus
```
python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 400 -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -log_file ../logs/val_abs_bert_amidialsum -model_path ../models/amidialsum_model -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_amidialsum

python train.py -task abs -mode test -test_from model_step_2000.pt -batch_size 3000 -test_batch_size 400 -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -log_file ../logs/val_abs_bert_amidialsum -model_path ../models/amidialsum_model -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_amidialsum
```



## NOTES

* View loss curve: `tensorboard --logdir=models/amidialsum_model/`


* TODO: save best checkpoint only
* Current: -mode train and load pre-trained weights given by authors
    * What is -finetune_bert param?




//-----------------------------------



## OTHER NOTES
### Steps modified for AMI Meeting Corpus:
* Step 3. Sentence Splitting and Tokenization

```
python preprocess.py -mode tokenize -raw_path ../raw_data/ami_meeting_raw/abstractive -save_path ../raw_data/ami_meeting_tokenized -log_file ../logs/ami_meeting.log
```

* Step 4. Format to Simpler Json Files
    * AMI Corpus doesn't have `-map_path` for `train, val, test`
    * Modification: `-mode format_to_lines` changed to `-mode format_xsum_to_lines`
    * Added `-data_split_json_is_full_path -data_split_json ../raw_data/ami_meeting_data_split.json`

```
python preprocess.py -mode format_xsum_to_lines -data_split_json_is_full_path -data_split_json ../raw_data/ami_meeting_data_split.json -raw_path ../raw_data/ami_meeting_tokenized -save_path ../json_data/ami_meeting -n_cpus 1 -use_bert_basic_tokenizer false -log_file ../logs/ami_meeting.log
```

`MAP_PATH` is the  directory containing the urls files (`../urls`)