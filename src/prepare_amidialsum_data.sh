
## ------ Pre-processing custom data --------
# Settings
ROOT_DATA_PATH=../raw_data
DATA_PATH=ami_dialsum_corpus_stories

ROOT_RESULT="../logs/${DATA_PATH}_bertsumextabs"
mkdir $ROOT_RESULT
RESULT_PATH="${ROOT_RESULT}/eval"  # Root name for all generated files
LOG_FILE="${ROOT_RESULT}/eval.log"
BERT_DATA_FULL="${BERT_DATA}/${DATA_PATH}"  # Path to data + prefix


JSON_DATA_PATH="../json_data/${DATA_PATH}"
BERT_DATA="../bert_data/${DATA_PATH}_bin"
PRE_PROCESS=true

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
# Makedir `../json_data/ami_dialsum` and run the following
python preprocess.py -mode format_to_lines_amidialsum -raw_path ../raw_data/ami_dialsum_corpus_tokenized -save_path ../json_data/ami_dialsum/ami_dialsum_corpus -n_cpus 1 -use_bert_basic_tokenizer false -log_file ../logs/ami_dialsum_corpus.log



# Step 5. Format to PyTorch Files: make dir `../bert_data/ami_dialsum_corpus_bin` and run the following
python preprocess.py -mode format_to_bert -raw_path ../json_data/ami_dialsum -save_path ../bert_data/ami_dialsum_corpus_bin  -lower -n_cpus 1 -log_file ../logs/preprocess.log