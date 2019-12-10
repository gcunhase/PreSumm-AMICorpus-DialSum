
## ------ Pre-processing custom data --------
# Settings
ROOT_DATA_PATH=../raw_data
DATA_PATH=ami_dialsum_corpus_stories

ROOT_RESULT="../logs/${DATA_PATH}_bertsumextabs"
mkdir $ROOT_RESULT
RESULT_PATH="${ROOT_RESULT}/eval"  # Root name for all generated files
LOG_FILE="${ROOT_RESULT}/eval.log"
BERT_DATA_FULL="${BERT_DATA}/${DATA_PATH}"  # Path to data + prefix


# Export CoreNLP tokenizer
# export CLASSPATH=../stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar

# Sentence Splitting and Tokenization
for DATA_TYPE in "train" "test" "valid"; do
    RAW_PATH="${ROOT_DATA_PATH}/${DATA_PATH}/${DATA_TYPE}"
    SAVE_PATH="${ROOT_DATA_PATH}/${DATA_PATH}_tokenized/${DATA_TYPE}"
    LOG_FILE="${ROOT_RESULT}/${DATA_TYPE}.log"
    python preprocess.py -mode tokenize -raw_path $RAW_PATH -save_path $SAVE_PATH -log_file $LOG_FILE
done

# Format to Simpler Json Files
JSON_DATA_PATH="../json_data/${DATA_PATH}"
mkdir $JSON_DATA_PATH
SAVE_PATH="${JSON_DATA_PATH}/${DATA_PATH}"
RAW_PATH="${ROOT_DATA_PATH}/${DATA_PATH}_tokenized"
LOG_FILE="${ROOT_RESULT}/${DATA_PATH}.log"
python preprocess.py -mode format_to_lines_amidialsum -raw_path $RAW_PATH -save_path $SAVE_PATH -n_cpus 1 -use_bert_basic_tokenizer false -log_file $LOG_FILE


# Step 5. Format to PyTorch Files
RAW_PATH=$JSON_DATA_PATH
SAVE_PATH="../bert_data/${DATA_PATH}_bin"
mkdir $SAVE_PATH
LOG_FILE="${ROOT_RESULT}/preprocess.log"
python preprocess.py -mode format_to_bert -raw_path $RAW_PATH -save_path $SAVE_PATH -lower -n_cpus 1 -log_file $LOG_FILE
