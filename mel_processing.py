import torch
import torch.utils.data
import librosa

import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore")
MAX_WAV_VALUE = 32768.0

"""
librosa                      0.10.2.post1

"""

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


import torch
import librosa

def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device

    # If mel_basis is not cached, generate and cache it
    if fmax_dtype_device not in mel_basis:
        mel = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)

    # Apply the Mel filterbank to the spectrogram
    mel_spec = torch.matmul(mel_basis[fmax_dtype_device], spec)

    # Apply spectral normalization (assuming you have this function)
    mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec



mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec







def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):  
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device

    if fmax_dtype_device not in mel_basis:
        mel = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)

    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    # Pad the input to match the expected STFT behavior
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # Perform the Short-Time Fourier Transform (STFT)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    # Compute magnitude spectrogram
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    # Apply the mel filterbank
    mel_spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec


if __name__ =="__main__":


    audio_path = "随机片段01.wav"
    y, sr = librosa.load(audio_path,sr=22050)
    y=  torch.FloatTensor(y).unsqueeze(0)
    print(y.shape) # [1,T]
    mel_spech = mel_spectrogram_torch(y=y,
                          n_fft=1024,
                          num_mels=80,
                          sampling_rate=22050,
                          hop_size=256,
                          win_size=1024,
                          fmax=None,
                          fmin=0)
    print(mel_spech.shape)

    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(mel_spech[0])   
    plt.savefig('对数梅尔谱图.png')


    pass
