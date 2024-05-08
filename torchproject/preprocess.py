"""
Preprocess raw audio sample
"""
from typing import Union, Tuple
import torch
import torchaudio
from torch import Tensor, nn
from torchaudio.transforms import MelSpectrogram
from . import utils

SR = 8000  # sample rate

def load_audio(audio_path:str) -> Tuple[Tensor, int]:
    sample, sr = torchaudio.load(audio_path)
    return sample, sr

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
        Args:
            samples (tensor (*, 1, S)):
        Return:
            tensor (*, 1, S): source samples
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
        self.features = MelSpectrogram(sample_rate=sr,
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


# Define your own preprocess algorithm
# by inherit from BasePreprocess


def init_preprocessor(preprocess_cfg:Union[dict, None],
                      device:str="cpu"
                      ) -> BasePreprocess:
    """
    preprocess_cfg (dict, None): 
        {
            "{class_name}": kwargs (dict)
        }
        NOTE: if None, BasePreprocess is used
    """
    if preprocess_cfg is None:
        return BasePreprocess()
    preprocess_cfg = dict(preprocess_cfg)  # copy
    name = next(iter(preprocess_cfg))
    params = preprocess_cfg[name]
    try:
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
    device = "cuda"

    preprocessor = init_preprocessor(config["preprocess"], 
                                     device=device)

    # (BS, 1, S)
    samples = torch.rand((3, 1, 48000)).to(device=device)
    inp = preprocessor(samples)
    print(inp.shape)
    # (BS, F, T)
    