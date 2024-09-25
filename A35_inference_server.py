import torch
import commons
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


from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import soundfile as sf
import time


# 假设这里有你需要的模块
# from synthesizer import SynthesizerTrn, utils, infer, symbols, get_hparams_from_file

app = FastAPI()

# 全局变量保存模型和超参数，确保只加载一次
model = None
hps = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 请求体定义
class InferRequest(BaseModel):
    speaker_name: str
    language: str
    length_scale: float
    infer_text: str
    infer_id: int
    sdp_ratio: float
    output_path: str


# 加载模型
def load_model():
    global model, hps
    if model is None and hps is None:
        hps = get_hparams_from_file(config_path="configs/config.json")
        model_path = "A1_pretrained_models/Bert-VITS2_2.3/G_0.pth"
        
        # 初始化模型
        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            mas_noise_scale_initial=0.01,
            noise_scale_delta=2e-6,
            **hps.model,
        ).to(device)
        
        print('bert vits 的net_g初始化')

        # 加载预训练模型参数
        utils.load_checkpoint(model_path, model, None, skip_optimizer=True)
        model = model.to(device)
        print("模型已加载")

# 加载模型（仅服务程序启动时加载）
load_model()
# 推理接口
@app.post("/infer_bertvits2/")
async def infer_speech(infer_req: InferRequest):
    
    st = time.perf_counter()
    # 提取请求体中的参数
    speaker_name = infer_req.speaker_name
    language = infer_req.language
    length_scale = infer_req.length_scale
    infer_text = infer_req.infer_text
    infer_id = infer_req.infer_id
    sdp_ratio = infer_req.sdp_ratio
    output_path = infer_req.output_path

    # 进行语音合成
    audio = infer(
        text=infer_text,
        sdp_ratio=sdp_ratio,
        noise_scale=0.667,
        noise_scale_w=0.8,
        length_scale=length_scale,
        sid=speaker_name,
        language=language,
        hps=hps,
        net_g=model,
        reference_audio=None,
        skip_start=False,
        skip_end=False,
        style_text=None,
        style_weight=0.7,
    )

    # 保存生成的音频
    sf.write(output_path, audio, samplerate=44100)
    print(f"语音已保存至: {output_path}")

    # 计算推理速度、效率
    audiolen = len(audio) / 44100 # 单位秒
    textlen = len(infer_text)  # 单位 token数

    et =time.perf_counter()

    usetime  = f"{(et-st):.4f}"
    outs1 = f"{{audiolen:{audiolen:.4f},textlen:{textlen:.4f},usetime:{usetime}}}"


    return {"output_path": output_path,"out_info1":outs1}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8102)

