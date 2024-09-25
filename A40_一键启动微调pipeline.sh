#!/bin/bash

# 定义数据集名字。
MODEL_ID="SSB0273_50"

## 数据集位置：A2_prepared_audios/$MODEL_ID

## 1 ASR模型识别文本
python A3_scripts/A33_ASR_ScriptsGen.py \
        --wavdir A2_prepared_audios/$MODEL_ID \
        --output_txt A5_finetuned_trainingout/$MODEL_ID/filelists/script.txt \
        --lang ZH

## 2 复制配置文件
cp configs/config.json A5_finetuned_trainingout/$MODEL_ID

## 3 根据文件产生音素标注、训练验证集、
python preprocess_text.py \
    --transcription-path A5_finetuned_trainingout/$MODEL_ID/filelists/script.txt \
    --cleaned-path A5_finetuned_trainingout/$MODEL_ID/filelists/script.txt.cleaned \
    --train-path A5_finetuned_trainingout/$MODEL_ID/filelists/script.txt.cleaned.train \
    --val-path A5_finetuned_trainingout/$MODEL_ID/filelists/script.txt.cleaned.val \
    --config-path A5_finetuned_trainingout/$MODEL_ID/config.json

# 4 用bert模型产生音素的 .pt文件
python bert_gen.py -c A5_finetuned_trainingout/$MODEL_ID/config.json

# 5 melspec 
python spec_gen.py --script A5_finetuned_trainingout/$MODEL_ID/filelists/script.txt.cleaned.train

# 6 启动train ms 
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_ms.py -c A5_finetuned_trainingout/$MODEL_ID/config.json \
-m A5_finetuned_trainingout/$MODEL_ID \
-mb A1_pretrained_models/Bert-VITS2_2.3