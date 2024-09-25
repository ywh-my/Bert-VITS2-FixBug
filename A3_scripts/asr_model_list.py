import sherpa_onnx


def get_vad_punc_model():
    # load PUCN model 
    pcmodel = "A1_pretrained_models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx"

    config = sherpa_onnx.OfflinePunctuationConfig(
        model=sherpa_onnx.OfflinePunctuationModelConfig(ct_transformer=pcmodel))
    
    punct = sherpa_onnx.OfflinePunctuation(config)

    ## load VAD模型 
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = "A1_pretrained_models/VAD_model/silero_vad.onnx"
    config.sample_rate = 16000

    window_size = config.silero_vad.window_size
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
    
    return vad,punct,window_size


def get_model02():
    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer="A1_pretrained_models/sherpa-onnx-paraformer-zh-2023-03-28/model.onnx",
            tokens="A1_pretrained_models/sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt",
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            debug=False,
            provider='cuda'
        )
    name = "sherpa-onnx-paraformer-zh-2023-03-28"
    return recognizer , name 
