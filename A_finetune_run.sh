
## A2_prepared_audios/SSB0005_50

## 1 ASR模型识别文本
python A3_scripts/A33_ASR_ScriptsGen.py \
        --wavdir A2_prepared_audios/SSB0005_50\
        --output_txt A5_finetuned_trainingout/SSB0005_50/filelists/script.txt \
        --lang ZH

## 2 复制配置文件
cp configs/config.json A5_finetuned_trainingout/SSB0005_50

## 3 根据文件产生音素标注、训练验证集、

python preprocess_text.py \
    --transcription-path A5_finetuned_trainingout/SSB0005_50/filelists/script.txt \
    --cleaned-path A5_finetuned_trainingout/SSB0005_50/filelists/script.txt.cleaned \
    --train-path A5_finetuned_trainingout/SSB0005_50/filelists/script.txt.cleaned.train \
    --val-path A5_finetuned_trainingout/SSB0005_50/filelists/script.txt.cleaned.val \
    --config-path A5_finetuned_trainingout/SSB0005_50/config.json

# 4  用bert模型产生音素的  .pt文件
python bert_gen.py -c A5_finetuned_trainingout/SSB0005_50/config.json

# 5 melspec 

