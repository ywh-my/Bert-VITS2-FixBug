
import os

from asr_model_list import get_vad_punc_model,get_model02
import click
from pathlib import Path
import time
import logging 
# 配置日志记录，设置级别为 INFO
logging.basicConfig(level=logging.INFO)

import time
import librosa
from typing import List, Tuple
import wave
import numpy as np

def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()
    





class BaseASRModel(object):
    def __init__(self) -> None:
        
        # 模型、识别、

        self.recognizer  = None
        self.vad = None
        self.punct = None
        self.window_size = None
        self.recognizer_name = None 

    def asr_model_init(self,model_func):
        self.recognizer ,self.recognizer_name  = model_func()
        logging.info("load asr model")

    def get_asr_modelname(self,):
        return self.recognizer_name

    def vad_punc_model_init(self,):
        self.vad,self.punct,self.window_size = get_vad_punc_model()
        
        logging.info("load vad,punc model")


    def single_wav_recognize(self,input_wavfile):

        samples, sample_rate = read_wave(input_wavfile) ## <class 'numpy.ndarray'>
        duration = len(samples) / sample_rate
        # 采样率修正为16K
        if sample_rate != 16000:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        ## VAD 开始 ，VAD 会去除静音部分。将语音切割为多段。

        speech_samples = []
        time1 = time.time()
        while len(samples) > self.window_size:
            self.vad.accept_waveform(samples[:self.window_size])
            samples = samples[self.window_size:]
            while not self.vad.empty():
                speech_samples.append(self.vad.front.samples)
                self.vad.pop()
        self.vad.flush()
        while not self.vad.empty():
            speech_samples.append(self.vad.front.samples)
            self.vad.pop()
        #print(len(speech_samples),type(speech_samples))
        time2 = time.time()
        ### ASR 开始。 
        results = []
        for i in range(len(speech_samples)):
            s = self.recognizer.create_stream()
            s.accept_waveform(sample_rate, speech_samples[i]) 
            self.recognizer.decode_stream(s)
            results.append(s.result.text)
        #print("asr result:",results)
        time3 = time.time()

        ## 最终判断一下是不是要做 标点符号模型。 （某些ASR模型直接输出了 标点符号。）
        endsens = ["。","，",".",",","!","?","！","!"]
        results = [ x for x in results if x != ""]
        try:
            if results[0][-1] not in endsens:
                texts_with_punct = [ self.punct.add_punctuation(t) for t in results]
                time4 = time.time()
                final_result = "".join(texts_with_punct)
                
                t1,t2,t3 = (time2-time1),(time3-time2),(time4-time3)
                logging.info(f"语音时长:{duration:.4f},识别耗时:VAD:{t1:.4f}秒,ASR:{t2:.4f}秒,PUNC:{t3:.4f}秒")
                logging.info(f"识别结果:{final_result}")
                return {"text":final_result,"consume":{"vad":t1,"asr":t2,"punc":t3}}
            else:
                final_result = "".join(results)
                t1,t2,t3 = (time2-time1),(time3-time2),0.0
                logging.info(f"语音时长:{duration:.4f},识别耗时:VAD:{t1:.4f}秒,ASR:{t2:.4f}秒,PUNC:{t3:.4f}秒")
                logging.info(f"识别结果:{final_result}")
                return {"text":final_result,"consume":{"vad":t1,"asr":t2,"punc":t3}}
        except Exception as e:
            print(f"无正常结果，跳过该语音,{e}")


    
    
## out = aa.single_wav_recognize(input_wavfile=params.input_s)

@click.command()
@click.option('--wavdir', type=str, help='Your Wav datadir')
@click.option('--output_txt', type=str, help='Your Annotation text')
@click.option('--lang', type=str, help='Yourlanguage')
def biaozhu(wavdir, output_txt, lang):
    wavdir = Path(wavdir)
    
    # 检查 wavdir 目录是否存在
    if not wavdir.exists():
        raise FileNotFoundError(f"目录 {wavdir} 不存在")
    
    # 检查 output_txt 文件是否存在，存在则删除
    if os.path.exists(output_txt):
        os.remove(output_txt)
        logging.info(f"已删除存在的文件: {output_txt}")

    # 创建outputtext的父目录
    output_txt = Path(output_txt)
    output_txt_parent = output_txt.parent
    output_txt_parent.mkdir(exist_ok=True,parents=True)


    wavfiles = sorted([x for x in wavdir.rglob("*.wav")], key=lambda x: x.stem)

    # 逐个识别语音
    ttnum = len(wavfiles)
    kk = 0
    with open(output_txt, 'a', encoding="utf-8") as f1:
        speakername = wavdir.name
        for wavf in wavfiles:
            try:
                Annotation = aa.single_wav_recognize(input_wavfile=str(wavf))
                writestr = f"{str(wavf)}|{speakername}|{lang}|{Annotation['text']}"

                f1.write(f"{writestr}\n")

                kk += 1
                logging.info(f"第 {kk}/{ttnum} 个识别完毕")

            except Exception as e:
                print(f"无正常结果，跳过该标注,{e}")
                continue

           



    pass


if __name__ =="__main__":
            
    st = time.time()
    ## ASR识别pipeline类 的初始化
    aa = BaseASRModel()
    aa.vad_punc_model_init()
    aa.asr_model_init(model_func=get_model02)

    biaozhu()

    et = time.time()
    print(f'总用时间：{et-st}s')

    """
    python A3_scripts/A33_ASR_ScriptsGen.py \
        --wavdir A2_prepared_audios/SSB0005_50\
        --output_txt A5_finetuned_trainingout/SSB0005_50/filelists/script.txt \
        --lang ZH
    """
    pass




