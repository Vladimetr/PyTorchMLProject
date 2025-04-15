from typing import Tuple
import sys
import torch
from torch import nn
from torch import Tensor
from .base import BaseModel
sys.path.append("/app/references/EfficientAT")
from models.dymn.model import get_model as get_dymn
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH



class DynMobnet(BaseModel):
    """ Dynamic MobileNet"""
    def __init__(self, 
                 n_classes:int=2, 
                 preprocess_cfg:dict=None, 
                 device:str="cpu",
                 pretrained:str=None,
                 temp:float=1.0,
                 training:bool=False):
        super().__init__(n_classes, preprocess_cfg, device)
        width = NAME_TO_WIDTH(pretrained)
        self.model = get_dymn(width_mult=width, 
                              pretrained_name=pretrained,
                              pretrain_final_temp=temp,
                              num_classes=n_classes).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        # original params for pretrained models
        self.mel = AugmentMelSTFT(n_mels=128,
                        sr=32000,
                        win_length=800,
                        hopsize=320,
                        n_fft=1024,
                        freqm=0,
                        timem=0,
                        fmin=0,
                        fmax=None,
                        fmin_aug_range=10,
                        fmax_aug_range=2000
                        ).to(device)
        self.train() if training else self.eval()
        
    def _mel_forward(self, x):
        """
        inp: (B, 1, S): for ex. (50, 1, 160000)
        out: (B, 1, n_mels, T): for ex. (50, 1, 128, 500)
        """
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x
            
    def to(self, device: str) -> None:
        self.mel.to(device)
        self.model.to(device)
        self.softmax.to(device)

    def before_mixup(self, x: Tensor):
        return self._mel_forward(x)

    def after_mixup(self, x: Tensor) -> Tuple[Tensor]:
        logits, embeds = self.model(x)
        # (B, C), (B, 960)
        probs = self.softmax(logits)  # (B, C)
        return logits, probs

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        inp: (B, 1, S): sample
        out:
            (B, 1, C): logits
            (B, 1, C): probs
            (B, F): embeddings

        """
        x = self._mel_forward(x)
        logits, embeds = self.model(x)
        # (B, C), (B, 960)
        probs = self.softmax(logits)  # (B, C)
        return logits, probs



if __name__ == "__main__":
    import librosa
    import torchaudio
    from torch import autocast
    import os.path as osp

    audio_path = "/app/data/examples/airplane_1.wav"
    resample_rate = 32000
    device = torch.device('cuda') if False and torch.cuda.is_available() else torch.device('cpu')

    model = DynMobnet(n_classes=100, device=device, training=False, pretrained="dymn10_as")
    
    (sample, _) = librosa.core.load(audio_path, sr=resample_rate, mono=True)
    sample = torch.from_numpy(sample[None, :]).to(device)

    try:
        sample, sr = torchaudio.load(audio_path, normalize=True)
    except TypeError:
        sample, sr = torchaudio.load(audio_path, normalization=True)
    if resample_rate != sr:
        resample = torchaudio.transforms.Resample(sr, resample_rate)
        sample = resample(sample)
        sr = resample_rate
    sample = sample.to(device)

    # (1, S)

    sample = torch.unsqueeze(sample, 0).to(device)
    # (B, 1, S)
    print(sample.shape, sample.dtype, sample.device)

    with torch.no_grad():
        logits, probs, features = model(sample)

    # save feature vec
    print("Features shape", features.shape)
    bname = osp.splitext(osp.basename(audio_path))[0]
    feat_path = osp.join("/app/references/EfficientAT/etalons_devcont", bname + ".pt")
    # torch.save(features, feat_path)
    etalon = torch.load(feat_path).to(device)
    print("CLOSE:", torch.isclose(etalon.to(torch.float16), features.to(torch.float16), rtol=1e-3, atol=1e-3)[0, :10])

