

# Settings
FINE_TUNE=false
ROOT_NAME=abs_bert_amidialsum_bertextabsWeights_lrbert0.00002_lrdec0.00002  # abs_bert_amidialsum_longer  # abs_bert_amidialsum_bertextabsWeights
ROOT_NAME_MODEL=amidialsum_model_bertextabsWeights_lrbert0.00002_lrdec0.00002  # amidialsum_model_longer  # amidialsum_model_bertextabsWeights
TRAIN_STEPS=20000  # 4000
N_GPU=0

# Parameters
TEST_FROM="../models/${ROOT_NAME_MODEL}/model_step_${TRAIN_STEPS}.pt"
MODEL_PATH="../models/${ROOT_NAME_MODEL}"
ROOT_RESULT="../logs/${ROOT_NAME_MODEL}"
mkdir $ROOT_RESULT
RESULT_PATH="${ROOT_RESULT}/eval"  # Root name for all generated files
LOG_FILE="${ROOT_RESULT}/test_${TRAIN_STEPS}"

BERT_DATA=../bert_data/ami_dialsum_corpus_bin/ami_dialsum_corpus  # Path to data

# Eval
python train.py -task abs -mode test -test_from $TEST_FROM -batch_size 3000 -test_batch_size 400 -bert_data_path $BERT_DATA -log_file $LOG_FILE -model_path $MODEL_PATH -sep_optim true -use_interval true -visible_gpus $N_GPU -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path $RESULT_PATH
