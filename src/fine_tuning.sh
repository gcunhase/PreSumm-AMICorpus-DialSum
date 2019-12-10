
## ------ Settings --------
FINE_TUNE=false
ROOT_NAME=abs_bert_amidialsum_bertextabsWeights_lrbert0.00002_lrdec0.00002  # abs_bert_amidialsum_longer  # abs_bert_amidialsum_bertextabsWeights
ROOT_NAME_MODEL=amidialsum_model_bertextabsWeights_lrbert0.00002_lrdec0.00002  # amidialsum_model_longer  # amidialsum_model_bertextabsWeights
TRAIN_STEPS=20000

# Should be -1 in the first run, then change to any number of GPUs available
VISIBLE_GPUS=-1  # 1,2,3,4,5

# Parameters
LOAD_FROM_EXTRACTIVE="../models/bertsumextabs_cnndm_final_model_step_148000.pt"  # TEST_FROM
MODEL_PATH="../models/${ROOT_NAME_MODEL}"
ROOT_RESULT="../logs/${ROOT_NAME_MODEL}"
mkdir $ROOT_RESULT
RESULT_PATH="${ROOT_RESULT}/eval"  # Root name for all generated files
LOG_FILE="${ROOT_RESULT}/test_${TRAIN_STEPS}"

BERT_DATA=../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus  # Path to data


## ------ Train (fine-tuning with AMI DialSum Corpus) --------
python train.py -task abs -mode train -bert_data_path $BERT_DATA -dec_dropout 0.2 -model_path $MODEL_PATH -sep_optim true -lr_bert 0.00002 -lr_dec 0.00002 -save_checkpoint_steps 2000 -batch_size 140 -train_steps $TRAIN_STEPS -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 2000 -warmup_steps_dec 1000 -max_pos 512 -visible_gpus $VISIBLE_GPUS -log_file $RESULT_PATH  -load_from_extractive $LOAD_FROM_EXTRACTIVE
