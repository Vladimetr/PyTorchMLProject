"""
https://github.com/microsoft/CLAP
Accuracy 0.937 on ESC50
I checked myself on 16kHz
"""
from typing import Union, List
import sys
import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseModel
sys.path.append("/app/references/CLAP")
from msclap import CLAP
from ..utils import read_classes

"""
2023 version
audioenc_name='HTSAT', 
batch_size=1024, 
d_proj=1024, 
demo=False, 
duration=7, 
fmax=8000, 
fmin=50, freeze_text_encoder_weights=True, 
hop_size=320, 
mel_bins=64, 
n_fft=1024, 
num_classes=527, 
out_emb=768, 
sampling_rate=44100, 
temperature=0.003, 
text_len=77, 
text_model='gpt2', 
transformer_embed_dim=768, 
window_size=1024
"""

class CLAPmultimodal(BaseModel):
    """
    Model with audio encoder & text encoder
    probs = softmax(similarity_scores)
    between text embeds and audio embeds
    """
    def __init__(self,
                 classes:Union[List[str], str],
                 n_classes:int=None,
                 preprocess_cfg:dict=None,
                 version:str="2023",
                 prompt:str="this is the sound of ",
                 training:bool=False,
                 device:str="cpu"
                 ):
        super().__init__(
            n_classes=n_classes, 
            preprocess_cfg=preprocess_cfg,
            device=device,
            training=training
        )
        use_cuda = device != "cpu"
        self.model = CLAP(version=version, use_cuda=use_cuda)
        classes = read_classes(classes)
        assert len(classes) == n_classes
        self._init_cls_embeds(classes, prompt)

    def _init_cls_embeds(self, classes:List[str], prompt:str=""):
        print("Preparing text embeds...", flush=True)
        texts = [prompt + cls_ for cls_ in classes]
        with torch.no_grad() as _:
            self.cls_embeds = self.model.get_text_embeddings(texts)
        # (C, F)

    def load(self, weights:str):
        """
        Weights are already loaded and fixed
        """
        pass

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        B - batch size
        C - n classes
        S - 44100Hz * 7sec = 308700
        Args:
            x (B, 1, S)
        Returns:
            tuple
              (B, C): output logits
              (B, C): output probs (output of softmax)
        """
        with torch.no_grad():
            audio_embeds = self.model._get_audio_embeddings(x)
            # similarity scores
            sims = self.model.compute_similarity(audio_embeds, self.cls_embeds)
            preds = F.softmax(sims, dim=1).cpu()
            # (C, ) float
        return sims, preds
    

class CLAPencoder(BaseModel):
    """
    Classifier with pre-trained CLAP audio encoder
    HTSAT_Swin_Transformer
    """
    enc_dim = 1024  # encoder vector size
    def __init__(self,
                 n_classes:int,
                 preprocess_cfg:dict=None,
                 version:str="2023",
                 hid_dims:List[int]=[],
                 dropout:float=0.0,
                 freeze_encoder=True,
                 training:bool=False,
                 device:str="cpu",
                 ):
        super().__init__(
            n_classes=n_classes, 
            preprocess_cfg=preprocess_cfg,
            device=device
        )
        use_cuda = device != "cpu"
        self.model = CLAP(version=version, use_cuda=use_cuda)
        self.preprocessor = None
        self.freeze_encoder = freeze_encoder
        if not freeze_encoder:
            raise NotImplementedError("freeze_encoder=False")
        last_dim = self.enc_dim

        # hidden layers after encoder
        self.hid_layers = []  # hidden FC layers
        for hid_dim in hid_dims:
            hid = nn.Linear(last_dim, hid_dim).to(device)
            act = nn.ReLU().to(device)
            dropout = nn.Dropout(dropout).to(device)
            self.hid_layers += [hid, act, dropout]
            last_dim = hid_dim

        # head layer - classifier
        self.head = nn.Linear(last_dim, n_classes).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        self.train() if training else self.eval()

    def to(self, device) -> None:
        self.model.to(device)
        # hidden layers
        layers = self.hid_layers + [self.head, self.softmax]
        for layer in layers:
            layer.to(device)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        B - batch size
        C - n classes
        S - 44100Hz * 7sec = 308700
        Args:
            x (B, 1, S)
        Returns:
            tuple
              (B, C): output logits
              (B, C): output probs (output of softmax)
        """
        # freezed encoder
        if self.freeze_encoder:
            with torch.no_grad():
                audio_embeds = self.model._get_audio_embeddings(x)
        else:
            audio_embeds = self.model._get_audio_embeddings(x)
        # (B, F)

        # trainable layers
        x = audio_embeds
        for layer in self.hid_layers:
            x = layer(x)
        logits = self.head(x)
        probs = self.softmax(logits)
        return logits, probs
    


if __name__ == "__main__":
    from time import time
    model = CLAPencoder(n_classes=100, device="cpu", training=True)
    print(model.get_num_params())
    exit()


    inp = torch.rand((5, 1, 44100), dtype=torch.float32).to("cuda:0")

    start = time()
    for i in range(100):
        a = model(inp)
        print(i, ")", a[0].shape, a[1].shape)
    print(time() - start)
    