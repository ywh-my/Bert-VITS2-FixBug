import torch
import commons
import soundfile 
from text import cleaned_text_to_sequence, get_bert

# from clap_wrapper import get_clap_audio_feature, get_clap_text_feature
from typing import Union
from text.cleaner import clean_text
import utils
from models import SynthesizerTrn
from text.symbols import symbols

def get_text(text, language_str, hps, device, style_text=None, style_weight=0.7):
    style_text = None if style_text == "" else style_text
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = get_bert(
        norm_text, word2ph, language_str, device, style_text, style_weight
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert_ori
        ja_bert = torch.randn(1024, len(phone))
        en_bert = torch.randn(1024, len(phone))
    elif language_str == "JP":
        bert = torch.randn(1024, len(phone))
        ja_bert = bert_ori
        en_bert = torch.randn(1024, len(phone))
    elif language_str == "EN":
        bert = torch.randn(1024, len(phone))
        ja_bert = torch.randn(1024, len(phone))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, en_bert, phone, tone, language


def infer(
    text,sdp_ratio,noise_scale,noise_scale_w,length_scale,sid,language,hps,net_g,reference_audio=None,skip_start=False,
    skip_end=False,
    style_text=None,
    style_weight=0.7,
):
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
        device,
        style_text=style_text,
        style_weight=style_weight,
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        # emo = emo.to(device).unsqueeze(0)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                en_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del (
            x_tst,
            tones,
            lang_ids,
            bert,
            x_tst_lengths,
            speakers,
            ja_bert,
            en_bert,
        )  # , emo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio
    
# 1 导入hps加载函数
from utils import get_hparams_from_file

# 2 设置输出日志配置
import logging
logging.basicConfig(
    level=logging.INFO,  # 设置日志等级为 INFO 及以上
    format='%(asctime)s - %(levelname)s - %(message)s'  # 设置日志格式
)
if __name__=="__main__":
    ## 一、 超参数加载。
    hps = get_hparams_from_file(config_path="A5_finetuned_trainingout/SSB0005_50/config.json")
    device = "cuda:0"
    model_path = "A5_finetuned_trainingout/SSB0005_50/models/G_8000.pth"
    ## 
    speaker_name  = "SSB0005_50"
    language = "ZH"
    length_scale = 1.2
    infer_text = "今夜的月光如此清亮，不做些什么真是浪费。随我一同去月下漫步吧，不许拒绝"
    infer_id = 0
    sdp_ratio=0.4
    output_path = f'A4_model_output/{speaker_name}_{infer_id}.wav'

    ## 二、模型类实例初始化。 （已经初始化并加载了bert模型，但是初始化了vits模型，没加载预训练参数）

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=0.01,
        noise_scale_delta=2e-6,
        **hps.model,
    ).to(device )
    print('bert vits 的net_g初始化')
    
    ## 三、加载vits模型的预训练参数
    _ =  utils.load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    net_g = net_g.to(device)

    ## 四、根据各种参数，进行合成。
    audio  = infer(
    text=infer_text,
    sdp_ratio=sdp_ratio, # 重要参数~~~
    noise_scale=0.667,
    noise_scale_w=0.8,
    length_scale=length_scale,
    sid=speaker_name,
    language=language,
    hps=hps,
    net_g = net_g,
    reference_audio=None,
    skip_start=False,
    skip_end=False,
    style_text=None,
    style_weight=0.7,
    )
    

    # 五、写入语音。
    soundfile.write(output_path,audio,samplerate=44100)
    print(f"tts 合成：{output_path}")

    pass


