"""
Preprocess raw audio sample
"""
from typing import Union
import numpy as np
import torch
import torchaudio
from torch import Tensor, nn
import torchaudio.transforms as T
from . import utils

SR = 16000  # sample rate

class BasePreprocess(nn.Module):
    """
    Preprocess takes loaded audio (source sample (1, S))
    and gives tensor for model
    """
    def __init__(self, sr:int=SR, device:str="cpu"):
        super().__init__()
        self.sr = sr
        self.to(device=device)

    def __call__(self, samples:Tensor) -> Tensor:
        """
        Features extraction
        NOTE: batch of samples in input
        becuase Preprocess may be used inside model
        Args:
            samples (tensor (*, 1, S)):
        Return:
            tensor (*, 1, S): source samples
                              from torchaudio load with SR
        """
        return samples
    

class LogmelPreprocess(BasePreprocess):
    def __init__(self, 
                 sr:int=SR,
                 wnd_step:float=0.008,
                 wnd_len:float=0.010,
                 nfilt:int=40,
                 nfft:int=512,
                 device:str="cpu"):
        super().__init__(sr=sr, device=device)
        self.features = T.MelSpectrogram(sample_rate=sr,
                                       n_fft=nfft,
                                       f_max=sr // 2,
                                       win_length=int(wnd_len * sr),
                                       hop_length=int(wnd_step * sr),
                                       n_mels=nfilt)
        self.features.to(device=device)

    def __call__(self, samples:Tensor) -> Tensor:
        """
        Args:
            sample (tensor (*, 1, S))
        Return:
            tensor (*, F, T): logmels
        """
        features = self.features(samples)
        # (*, 1, F, T)
        out = torch.squeeze(features, features.dim()-3)
        # (*, F, T)
        return out
    

class ClapPreprocess(BasePreprocess):
    """
    Original preprocess from CLAP
    https://github.com/microsoft/CLAP/blob/e8a6467b87cd85716e20c6a008126150d9740be0/msclap/CLAPWrapper.py#L251
    """
    target_sr = 44100
    audio_duration = 7  # sec
    # Starg = 44100 * 7 = 308700
    resample = True

    def __init__(self, device: str = "cpu"):
        self.use_cuda = device != "cpu"
        if self.use_cuda:
            raise NotImplementedError("This preprocessor doesn't work on GPU")

    def read_audio(self, audio_path):
        r"""Loads audio file or array and returns a torch tensor"""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        audio_time_series, sample_rate = self._resample(audio_time_series, sample_rate)

    def _resample(self, sample:Tensor, src_sr:int) -> Tensor:
        if self.resample and self.target_sr != src_sr:
            resampler = T.Resample(src_sr, self.target_sr)
            sample = resampler(sample)
            return sample, self.target_sr
        return sample, src_sr

    def _load_audio_into_tensor(self, sample:Tensor, sr:int=SR):
        """
        Load audio and return fixed 7sec sample (1, S)
        SR for CLAP = 44100
        S = 44.1 * 7 = 308700
        NOTE: if audio less than 7 sec, it padded (repeat)
              otherwise NotImplementedError
        Args:
            sample (Tensor): (1, S) sample with SR less than
                             S must be less than SR * 7.0
        Returns:
            Tensor: (target_sr * audio_dur, )
        """
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        # audio_time_series, sample_rate = self.read_audio(audio_path, resample=resample)

        # resample
        audio_time_series, sample_rate = self._resample(sample, sr)

        audio_time_series = audio_time_series.reshape(-1)

        # audio_time_series is shorter than predefined audio duration,
        # so audio_time_series is extended
        if self.audio_duration * self.target_sr >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil((self.audio_duration*sample_rate) /
                                        audio_time_series.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0:self.audio_duration*sample_rate]
        else:
            # because using random model stop being determenistic
            raise NotImplementedError("Audio more than 7 sec")
            # audio_time_series is longer than predefined audio duration,
            # so audio_time_series is trimmed
            start_index = random.randrange(
                audio_time_series.shape[0] - self.audio_duration*sample_rate)
            audio_time_series = audio_time_series[start_index:start_index +
                                                  self.audio_duration*sample_rate]
        return torch.FloatTensor(audio_time_series)

    def __call__(self, samples:Tensor) -> Tensor:
        """
        Starg = target_sr * audio_duration = 308700
        Args:
            samples (B, 1, S): samples with SR less than
                               S must be less than SR * 7.0
        Returns:
            (B, 1, Starg)
        """
        batch = []
        for sample in samples:
            audio_tensor = self._load_audio_into_tensor(sample, SR)
            audio_tensor = audio_tensor.reshape(
                1, -1).cuda() if self.use_cuda and torch.cuda.is_available() else audio_tensor.reshape(1, -1)
            batch.append(audio_tensor)

        # collate
        # https://github.com/microsoft/CLAP/blob/e8a6467b87cd85716e20c6a008126150d9740be0/msclap/CLAPWrapper.py#L170
        out = None
        elem = batch[0]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)


class MobnetPreprocess(BasePreprocess):
    clip_length = 160000
    def __init__(self, 
                 sr:int = SR, 
                 device:str = "cpu",
                 gain_augment:int=0):
        """
        gain_augment (int): non-zero only for train
        """
        super().__init__(sr, device)
        self.gain_augment = gain_augment

    def _pad_or_truncate(self, x):
        """Pad all audio to specific length."""
        if len(x) <= self.clip_length:
            return np.concatenate((x, np.zeros(self.clip_length - len(x), dtype=np.float32)), axis=0)
        else:
            return x[0: self.clip_length]

    @staticmethod
    def _pydub_augment(waveform, gain_augment):
        """
        gain_augment (int): must be non-zero
        """
        gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
        amp = 10 ** (gain / 20)
        waveform = waveform * amp
        return waveform

    def __call__(self, samples:Tensor) -> Tensor:
        """
        Features extraction
        Args:
            samples (tensor (*, 1, S)): (*, 1, 160'000)
        Return:
            tensor (*, 1, S)
        """
        if self.gain_augment:
            samples = self._pydub_augment(samples, self.gain_augment)
        # sample = self._pad_or_truncate(sample)
        return samples


# Define your own preprocess algorithm
# by inherit from BasePreprocess


def init_preprocessor(preprocess_cfg:Union[dict, None],
                      device:str="cpu"
                      ) -> BasePreprocess:
    """
    preprocess_cfg (dict, None): 
        {
            "use": "{class_name_i}",
            "{class_name_1}": kwargs (dict),
            "{class_name_2}": kwargs (dict),
            ...
        }
        NOTE: if None, BasePreprocess is used
    """
    if preprocess_cfg is None:
        return BasePreprocess()
    preprocess_cfg = dict(preprocess_cfg)  # copy
    name = preprocess_cfg["use"]
    if not name:
        return BasePreprocess()
    try:
        params = preprocess_cfg[name]
        # define class
        preprocessor = globals()[name]
    except KeyError:
        raise ValueError(f"Invalid preprocess name '{name}'")
    try:
        # init
        preprocessor = preprocessor(**params, device=device)
    except TypeError:
        raise ValueError(f"Invalid preprocess params {params}")
    return preprocessor



if __name__ == '__main__':
    config = utils.config_from_yaml('config.yaml')
    device = "cpu"

    preprocessor = init_preprocessor(config["preprocess"], 
                                     device=device)

    # (BS, 1, S)
    samples = torch.rand((50, 1, 80000)).to(device=device)
    inp = preprocessor(samples)
    print(inp.shape)
    # (BS, F, T)
    