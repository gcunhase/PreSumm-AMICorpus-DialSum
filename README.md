## About

#### Disclaimer
The PreSumm model, presented in the EMNLP 2019 paper titled "[Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)" [[original code](https://www.github.com/nlpyang/PreSumm)], is not my work. Please credit the appropriate authors for that model.

#### Purpose of this repository
* Need to use PreSumm as baseline model for comparison with a custom dataset.
* Using the pre-trained model `BertExtAbs`, fine-tune PreSumm with the custom dataset.
* [Additional notes are available](./README_notes.md), including [my code modifications](./README_notes.md#my-modifications) in detail.

### Contents
[Requirements](#requirements) • [How to Use](#how-to-use) • [How to Cite](#acknowledgement)

## Requirements
Python 3.5.2, PyRouge [[notes](./README_notes.md)]

```
pip install -r requirements.txt
```

## How to Use
1. **First run**: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.

2. Download best performing model with PreSumm: [CNN/DM BertExtAbs](https://drive.google.com/open?id=1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr)

### A. Evaluate on untrained BertSumExtAbs 
```
./src/load_custom_data_an_eval.sh
```

### B. Fine-tune BertSumExtAbs For AMI DialSum Meeting Corpus

B.1. [Download CoreNLP](https://stanfordnlp.github.io/CoreNLP) and export:
```
export CLASSPATH=./stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
```

B.2. Prepare dataset
* [Download AMI DialSum Corpus](https://github.com/MiuLab/DialSum) [[paper](arxiv.org/abs/1809.05715)]
* Delete `<EOS>` tags
* Convert to `.story` with `src/ami_dialsum_corpus_story.py`
* Run
    ```
    ./src/prepare_amidialsum_data.sh
    ```

B.3. Fine-tune model with AMI DIalSum dataset (modified settings such as `train_steps`, `lrbert`, `lrdec`, `warmup*`, ...)
```
./src/fine_tuning.sh
```

B.4. Evaluate
```
./src/eval.sh
```

## Acknowledgement
