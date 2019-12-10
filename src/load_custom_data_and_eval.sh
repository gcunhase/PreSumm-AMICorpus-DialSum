
## ------ Pre-processing custom data --------
# Settings
ROOT_DATA_PATH=../raw_data
CUSTOM_DATA_PATH=custom_data_article_nlm_5
JSON_DATA_PATH="../json_data/${CUSTOM_DATA_PATH}"
BERT_DATA="../bert_data/${CUSTOM_DATA_PATH}_bin"
PRE_PROCESS=true

# Export CoreNLP tokenizer
# export CLASSPATH=../stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar

# Pre-process data
if [ "$PRE_PROCESS" = true ]; then
    echo "Pre-process data"

    # Delete previous processed data
    CUSTOM_DATA_TOKENIZED="${ROOT_DATA_PATH}/${CUSTOM_DATA_PATH}_tokenized"
    rm -r $CUSTOM_DATA_TOKENIZED
    rm -r $JSON_DATA_PATH
    rm -r $BERT_DATA

    # Pre-process
    python preprocess.py -mode tokenize -raw_path "${ROOT_DATA_PATH}/${CUSTOM_DATA_PATH}" -save_path $CUSTOM_DATA_TOKENIZED -log_file "../logs/${CUSTOM_DATA_PATH}_train.log"
    mkdir $JSON_DATA_PATH
    python preprocess.py -mode format_to_lines_customdata -raw_path $CUSTOM_DATA_TOKENIZED -save_path "${JSON_DATA_PATH}/${CUSTOM_DATA_PATH}" -n_cpus 1 -use_bert_basic_tokenizer false -log_file "../logs/${CUSTOM_DATA_PATH}.log"
    mkdir $BERT_DATA
    python preprocess.py -mode format_to_bert -raw_path $JSON_DATA_PATH -save_path $BERT_DATA  -lower -n_cpus 1 -log_file ../logs/preprocess.log
else
    echo "Data has been pre-processed"
fi


## ------ Evaluating pre-processed custom data --------
# Settings

# Should be -1 in the first run, then change to any number of GPUs available
VISIBLE_GPU=-1  # 6

TEST_FROM="../models/bertsumextabs_cnndm_final_model_step_148000.pt"
MODEL_PATH="../models"
ROOT_RESULT="../logs/bertsumextabs_cnndm_${CUSTOM_DATA_PATH}"
mkdir $ROOT_RESULT
RESULT_PATH="${ROOT_RESULT}/eval"  # Root name for all generated files
LOG_FILE="${ROOT_RESULT}/eval.log"
BERT_DATA_FULL="${BERT_DATA}/${CUSTOM_DATA_PATH}"  # Path to data + prefix

# Eval
python train.py -task abs -mode test -dont_calculate_rouge -test_from $TEST_FROM -batch_size 3000 -test_batch_size 400 -bert_data_path $BERT_DATA_FULL -log_file $LOG_FILE -model_path $MODEL_PATH -sep_optim true -use_interval true -visible_gpus $VISIBLE_GPU -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path $RESULT_PATH
# -task abs -mode test -dont_calculate_rouge -test_from ../models/bertsumextabs_cnndm_final_model_step_148000.pt -batch_size 3000 -test_batch_size 400 -bert_data_path ../bert_data/custom_data_hong_kong_bin/custom_data_hong_kong -log_file ../logs/bertsumextabs_cnndm_custom_data_hong_kong/eval.log -model_path ../models -sep_optim true -use_interval true -visible_gpus 6 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/bertsumextabs_cnndm_custom_data_hong_kong/eval

