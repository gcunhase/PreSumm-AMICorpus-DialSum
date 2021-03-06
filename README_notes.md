## About
Additional notes regarding code.

#### My code modifications
* Data pre-processing
    * `src/ami_dialsum_corpus_story.py`: script to format the AMI DialSum Corpus into the same CNN/DM `.story` format
    * In `src/prepo/data_builder.py`: added function `format_to_lines_amidialsum()` for use in *'Step 4'* of pre-processing the custom data
        * Needed because the custom data is already split into train/test/dev and there was no function to deal with this situation
    * In `src/preprocess.py`: addition of `-mode format_to_lines_amidialsum`
    
* Fine-tuning PreSumm
    * Load the [BertExtAbs pre-trained weights](https://drive.google.com/open?id=1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr) as `-load_from_extractive ../models/bertsumextabs_cnndm_final_model_step_148000.pt`

* Evaluation / PyRouge:
    * Installation: notes on extra steps needed to run PyRouge
    * In `src/others/pyrouge.py` in line 560: changed option `-n` to 3 instead of 2 so ROUGE-3 can also be calculated
    * In `src/others/utils.py`: return dictionary updated with extra information ("rouge_3_f_score" and "rouge_3_recall")
    * In `src/models/predictor.py`: update `tensorboard_writer` with ROUGE-3

## Requirements
### PyRouge notes
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
### Processing the data yourself
* Step 1: Dataset
    * [Download AMI DialSum Corpus](https://github.com/MiuLab/DialSum)
    * Delete `<EOS>` tags
    * Convert to `.story` with `src/ami_dialsum_corpus_story.py`

* Step 2:
```
export CLASSPATH=~/Gwena/PreSumm/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
```

* Step 3. Sentence Splitting and Tokenization

```
python preprocess.py -mode tokenize -raw_path ../raw_data/ami_dialsum_corpus_stories/train -save_path ../raw_data/ami_dialsum_corpus_tokenized/train -log_file ../logs/ami_dialsum_corpus_train.log
python preprocess.py -mode tokenize -raw_path ../raw_data/ami_dialsum_corpus_stories/test -save_path ../raw_data/ami_dialsum_corpus_tokenized/test -log_file ../logs/ami_dialsum_corpus_test.log
python preprocess.py -mode tokenize -raw_path ../raw_data/ami_dialsum_corpus_stories/valid -save_path ../raw_data/ami_dialsum_corpus_tokenized/valid -log_file ../logs/ami_dialsum_corpus_valid.log
```

* Step 4. Format to Simpler Json Files
    * AMI Corpus doesn't have `-map_path` for `train, val, test`
    * Modification: `-mode format_to_lines` changed to `-mode format_to_lines_amidialsum`
    * Makedir `../json_data/ami_dialsum` and run the following

    ```
    python preprocess.py -mode format_to_lines_amidialsum -raw_path ../raw_data/ami_dialsum_corpus_tokenized -save_path ../json_data/ami_dialsum/ami_dialsum_corpus -n_cpus 1 -use_bert_basic_tokenizer false -log_file ../logs/ami_dialsum_corpus.log
    ```

* Step 5. Format to PyTorch Files: make dir `../bert_data/ami_dialsum_corpus_bin` and run the following
```
python preprocess.py -mode format_to_bert -raw_path ../json_data/ami_dialsum -save_path ../bert_data/ami_dialsum_corpus_bin  -lower -n_cpus 1 -log_file ../logs/preprocess.log
```

## Model Training

* Download best performing model with PreSumm: [CNN/DM BertExtAbs](https://drive.google.com/open?id=1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr)

* **First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

### Abstractive Setting with BertExtAbs
* Trained for only 2,000 steps
```
python train.py -task abs -mode train -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -dec_dropout 0.2 -model_path ../models/amidialsum_model -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 200 -batch_size 140 -train_steps 2000 -report_every 10 -accum_count 2 -use_bert_emb true -use_interval true -warmup_steps_bert 200 -warmup_steps_dec 100 -max_pos 512 -visible_gpus 1 -log_file ../logs/abs_bert_amidialsum  -load_from_extractive ../models/bertsumextabs_cnndm_final_model_step_148000.pt   
```
* Trained for only 2,000 steps with BERT fine-tune
```
python train.py -finetune_bert -task abs -mode train -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -dec_dropout 0.2 -model_path ../models/amidialsum_bertfinetune_model -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 6000 -report_every 10 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 200 -warmup_steps_dec 100 -max_pos 512 -visible_gpus 1,2,3,4,5 -log_file ../logs/abs_bert_amidialsum_bertfinetune  -load_from_extractive ../models/bertsumextabs_cnndm_final_model_step_148000.pt   
```

* Trained for 4,000 steps (modified settings) HEREEEE
```
python train.py -task abs -mode train -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -dec_dropout 0.2 -model_path ../models/amidialsum_model_bertextabsWeights_lrbert0.0002 -sep_optim true -lr_bert 0.0002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 20000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 1000 -warmup_steps_dec 500 -max_pos 512 -visible_gpus 1,2,3,4,5 -log_file ../logs/abs_bert_amidialsum_bertextabsWeights_lrbert0.0002  -load_from_extractive ../models/bertsumextabs_cnndm_final_model_step_148000.pt
python train.py -task abs -mode train -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -dec_dropout 0.2 -model_path ../models/amidialsum_model_bertextabsWeights_lrbert0.00002 -sep_optim true -lr_bert 0.00002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 10000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 1000 -warmup_steps_dec 500 -max_pos 512 -visible_gpus 1,2,3,4,5 -log_file ../logs/abs_bert_amidialsum_bertextabsWeights_lrbert0.00002  -load_from_extractive ../models/bertsumextabs_cnndm_final_model_step_148000.pt
python train.py -task abs -mode train -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -dec_dropout 0.2 -model_path ../models/amidialsum_model_bertextabsWeights_lrbert0.00002_lrdec0.00002 -sep_optim true -lr_bert 0.00002 -lr_dec 0.00002 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 20000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 2000 -warmup_steps_dec 1000 -max_pos 512 -visible_gpus 1,2,3,4,5 -log_file ../logs/abs_bert_amidialsum_bertextabsWeights_lrbert0.00002_lrdec0.00002  -load_from_extractive ../models/bertsumextabs_cnndm_final_model_step_148000.pt
```


* Trained for longer (original settings)
```
python train.py -task abs -mode train -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -dec_dropout 0.2 -model_path ../models/amidialsum_model_longer -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 1000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 1,2,3,4,5 -log_file ../logs/abs_bert_amidialsum_longer  -load_from_extractive ../models/bertsumextabs_cnndm_final_model_step_148000.pt   

python train.py -finetune_bert -task abs -mode train -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -dec_dropout 0.2 -model_path ../models/amidialsum_model_longer_bertfinetune -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 1,2,3,4,5 -log_file ../logs/abs_bert_amidialsum_longer_bertfinetune  -load_from_extractive ../models/bertsumextabs_cnndm_final_model_step_148000.pt   
```
> Original: -train_steps 200000 -warmup_steps_bert 20000 -warmup_steps_dec 10000


* Longer, using `-train_from`
```
python train.py -finetune_bert -task abs -mode train -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -dec_dropout 0.2 -model_path ../models/amidialsum_model_longer_bertfinetune -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 1000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 1,2,3,4,5 -log_file ../logs/abs_bert_amidialsum_longer_bertfinetune  -train_from ../models/bertsumextabs_cnndm_final_model_step_148000.pt  -load_from_extractive ../models/bertsumextabs_cnndm_final_model_step_148000.pt 

Lower lr
python train.py -finetune_bert -task abs -mode train -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -dec_dropout 0.2 -model_path ../models/amidialsum_model_longer_bertfinetune -sep_optim true -lr_bert 0.0002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 2000 -warmup_steps_dec 1000 -max_pos 512 -visible_gpus 1,2,3,4,5 -log_file ../logs/abs_bert_amidialsum_longer_bertfinetune  -train_from ../models/bertsumextabs_cnndm_final_model_step_148000.pt  -load_from_extractive ../models/bertsumextabs_cnndm_final_model_step_148000.pt 
```

### Don't lead pre-trained weights
```
python train.py -task abs -mode train -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -dec_dropout 0.2 -model_path ../models/amidialsum_model_longer_fromScratch -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 1000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 1,2,3,4,5 -log_file ../logs/abs_bert_amidialsum_fromScratch  

python train.py -task abs -mode test -test_from ../models/amidialsum_model_longer_bertfinetune_fromScratch/model_step_4000.pt -batch_size 3000 -test_batch_size 400 -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -log_file ../logs/test_abs_bert_amidialsum_longer_bertfinetune_fromScratch -model_path ../models/amidialsum_model_longer_bertfinetune_fromScratch -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_amidialsum_longer_bertfinetune_fromScratch
```

### Abstractive Settings with BertAbs
* Download pre-processed CNN/DM data from original PreSum repository [[download](https://drive.google.com/open?id=1DN7ClZCCXsk2KegmC6t4ClBwtAf5galI)]
* Train:
```
python train.py -task abs -mode train -bert_data_path ../bert_data/bert_data_cnndm_final/cnndm -dec_dropout 0.2  -model_path ../models/cnndm_bertabs -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus -1  -log_file ../logs/cnndm_bertabs
```

## Model Evaluation
### AMI DialSum Corpus
* Output in `logs` directory
    * Target transcripts: `abs_bert_amidialsum.[CKP_TRAIN_STEP].raw_src`
    * Target summaries: `abs_bert_amidialsum.[CKP_TRAIN_STEP].gold`
    * Generated summaries: `abs_bert_amidialsum.[CKP_TRAIN_STEP].candidate`
    * Log (includes ROUGE scores): `test_abs_bert_amidialsum`
    
* Check specific checkpoint by using `-test_from ../models/amidialsum_model/model_step_2000.pt`:
```
python train.py -task abs -mode test -test_from ../models/amidialsum_model/model_step_2000.pt -batch_size 3000 -test_batch_size 400 -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -log_file ../logs/test_abs_bert_amidialsum -model_path ../models/amidialsum_model -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_amidialsum

python train.py -task abs -mode test -test_from ../models/amidialsum_model_bertextabsWeights_lrbert0.00002/model_step_2000.pt -batch_size 3000 -test_batch_size 400 -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -log_file ../logs/test_abs_bert_amidialsum_bertextabsWeights_lrbert0.00002 -model_path ../models/amidialsum_model_bertextabsWeights_lrbert0.00002 -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_amidialsum_bertextabsWeights_lrbert0.00002

python train.py -task abs -mode test -test_from ../models/amidialsum_bertfinetune_model/model_step_4000.pt -batch_size 3000 -test_batch_size 400 -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -log_file ../logs/test_abs_bert_amidialsum_bertfinetune -model_path ../models/amidialsum_bertfinetune_model -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_amidialsum_bertfinetune_model

python train.py -task abs -mode test -test_from ../models/amidialsum_model_longer/model_step_1000.pt -batch_size 3000 -test_batch_size 400 -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -log_file ../logs/test_abs_bert_amidialsum_longer -model_path ../models/amidialsum_model_longer -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_amidialsum_longer

python train.py -task abs -mode test -test_from ../models/amidialsum_model_longer_bertfinetune/model_step_178000.pt -batch_size 3000 -test_batch_size 400 -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -log_file ../logs/test_abs_bert_amidialsum_longer_bertfinetune -model_path ../models/amidialsum_model_longer_bertfinetune -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_amidialsum_longer_bertfinetune
```
> amidialsum_model at step 2000: ROUGE-F(1/2/3/l): 9.40/6.80/9.33, ROUGE-R(1/2/3/l): 50.47/44.60/50.24

> amidialsum_model at step 60000: ROUGE-F(1/2/3/l): 11.45/7.70/11.29, ROUGE-R(1/2/3/l): 65.04/54.02/64.47

* Testing with Pre-training model (not fine tuning)
```
python train.py -task abs -mode test -test_from ../models/bertsumextabs_cnndm_final_model_step_148000.pt -batch_size 3000 -test_batch_size 400 -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -log_file ../logs/test_abs_bert_amidialsum_cnndm_pretrain -model_path ../models -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_amidialsum_cnndm_pretrain

python train.py -task abs -mode test -test_from ../models/bertsumextabs_cnndm_final_model_step_148000.pt -batch_size 3000 -test_batch_size 400 -bert_data_path ../bert_data/cnndm_bin/cnndm -log_file ../logs/test_abs_bert_cnndm_pretrain -model_path ../models -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_cnndm_pretrain
```
 

* Check all saved checkpoints:
```
python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 400 -bert_data_path ../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus -log_file ../logs/val_abs_bert_amidialsum -model_path ../models/amidialsum_model -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_amidialsum
```


## Notes

* View loss curve: `tensorboard --logdir=models/amidialsum_model/`

* TODO: save best checkpoint only
* Current: -mode train and load pre-trained weights given by authors
    * What is -finetune_bert param?
