"""
Validation and preprocessing of raw audio
"""
import os
import os.path as osp
import ffmpeg
import torch
from torch import Tensor
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram
import torch.nn.functional as F
from . import utils

class FileError(Exception):
    pass
class AudioError(Exception):
    """ These errors can be fixed using ffmpeg """
    pass
class TooLongAudioError(Exception):
    pass
class TooShortAudioError(Exception):
    pass

# acceptable audio formats
AUDIO_FORMATS = ['wav', 'mp3', 'mpeg', 'flac']


class BaseFeatures(nn.Module):
    def __init__(self, sr:int=8000):
        super().__init__()
        self.sr = sr

    def __call__(self, sample:Tensor) -> Tensor:
        """
        Features extraction
        Args:
            sample (tensor (1, S)):
        Return:
            tensor (1, S): source sample
        """
        return sample

    def split(self, features:Tensor, chunk:float, pad:bool=True
              ) -> Tensor:
        """
        Split features on parts (chunks)
        T - total feature time (sample size here)
        Tch - feature time for chunk
        N - num of chunks
        Args:
            features (tensor (1, T)):
            chunk (float): chunk size (sec)
            pad (bool): whether to pad to last chunk or drop
        Returns:
            tensor (N, *, Tch): splitted features
        """
        features_shape = features.shape
        s_len = features_shape[-1]
        chunk_len = int(chunk * self.sr)
        # add padding
        if s_len % chunk_len > 0 and pad:
            pad = chunk_len - (s_len % chunk_len)
            features = F.pad(features, (0, pad), mode='constant', value=0)
            assert features.shape[1] % chunk_len == 0

        chunks = torch.split(features, chunk_len, dim=-1)
        last_chunk = chunks[-1]
            
        # drop last short chunk
        if last_chunk.shape[-1] != chunk_len:
            chunks = chunks[ :-1]
        try:
            chunks = torch.stack(chunks, dim=0)
        except RuntimeError:
            return torch.empty(0, *features_shape[ :-1], chunk_len)
        return chunks

    def collate(self, features_ls, max_dur:float=None) -> Tensor:
        """
        Collate B-list of features to tensor(B, *)
        All features will be padded to given max_dur
        or max time of given features
        Args:
            features_ls (list[tensor]): B-list of features (*, T)
            max_dur (float, None): max duration of features (sec).
                If None, max duration of features
        Raise:
            ValueError: "Features list is empty"
            ValueError: "Time is more than max time"
        Returns:
            tensor (B, *, Tmax): collated features 
        """
        return torch.zeros(15, 1, 40000)


class LogmelFeatures(BaseFeatures):
    def __init__(self, 
                 sr:int=8000,
                 wnd_step:float=0.008,
                 wnd_len:float=0.010,
                 nfilt:int=40,
                 nfft:int=512):
        super().__init__()
        self.features = MelSpectrogram(sample_rate=sr,
                                       n_fft=nfft,
                                       f_max=sr // 2,
                                       win_length=int(wnd_len * sr),
                                       hop_length=int(wnd_step * sr),
                                       n_mels=nfilt)
        self.wnd_step = wnd_step

    def __call__(self, sample:Tensor) -> Tensor:
        """
        Args:
            sample (tensor (*, 1, S))
        Return:
            tensor (*, F, T): logmels
        """
        features = self.features(sample)
        # (*, 1, F, T)
        return torch.squeeze(features, features.dim()-3)

    def split(self, features: Tensor, chunk: float, pad: int = None) -> Tensor:
        """
        TODO: check this
        """
        # (F, T)
        shape = features.shape
        total_time = shape[-1]
        chunk = round(chunk / self.wnd_step) + 1
        chunk_step = chunk - 1

        chunks = []
        for t in range(0, total_time - chunk + 1, chunk_step):
            f = features[..., t : t + chunk]
            # (F, Tfr)
            chunks.append(f)

        remain_t = total_time - (t + chunk_step + 1)
        assert remain_t >= 0

        # add remain part
        if remain_t and pad:
            pad_f = pad * torch.ones(*shape[ :-1], chunk, 
                                     dtype=features.dtype,
                                     device=features.device)
            # (F, Tfr)
            pad_f[..., :remain_t] = features[..., -remain_t: ]
            chunks.append(pad_f)

        # N-list to tensor (N, *)
        try:
            chunks = torch.stack(chunks, 0)
        except Exception:
            raise ValueError("No chunks in features {}"\
                             .format(features.size()))

        return chunks



def init_features_extractor(name:str, params:dict) -> BaseFeatures:
    if name is None:
        return BaseFeatures()

    extractors = {
        "logmel": LogmelFeatures,
    
    }
    try:
        extractor = extractors[name]
    except KeyError:
        raise ValueError(f"Invalid features '{name}'")

    return extractor(**params)


class AudioPreprocess():
    """
    Example of audio pipeline:
        preprocess = AudioPreprocess()
        sample, dur = preprocess.from_audio_path(audio_path)
        features = preprocess.extract_features(sample)
    """
    def __init__(self, 
                 sr:int=8000,
                 codec:str='pcm_s16le',
                 min_dur:float=0.2,
                 max_dur:float=300,
                 format:str='wav',
                 features:dict=None
                 ):
        self.sr = sr
        self.codec = codec
        self.format = format
        self.min_dur = min_dur   # (sec)
        self.max_dur = max_dur   # (sec)
        if features:
            features_name = list(features.keys())[0]
            features_params = features[features_name]
        else:
            features_name, features_params = None, dict()

        self.features_extractor = init_features_extractor(features_name, 
                                                          features_params
        )
        # audio convert using FFMPEG
        self.ffmpeg_convert_params = {
            'ar': str(self.sr),
            'acodec': self.codec,
            'f': self.format,
            'loglevel': 'error',
        }

    def extract_features(self, sample:Tensor) -> Tensor:
        return self.features_extractor(sample)

    @staticmethod
    def media_info(audio_path):
        """
        Get audio parameters (like ffprobe)
        Args:
            audio_path (str)
        Returns:
            dict: parameters
        """
        return ffmpeg.probe(audio_path)

    @staticmethod
    def get_sample(audio_path):
        """
        C - number of channels
        S - sample length
        Args:
            audio_path (str): path/to/audio.wav
        Returns:
            tuple
                tensor (C, S): sample (float)
                sr (int): sample rate
        """
        try:
            sample, sr = torchaudio.load(audio_path, normalization=True)
        except TypeError:
            # for new versions of torchaudio
            sample, sr = torchaudio.load(audio_path, normalize=True)
        return sample, sr

    def save_audio(self, sample, wav_path, sr=8000):
        torchaudio.save(wav_path, sample, sr)
        
    def validate_audio_file(self, audio_path):
        """
        Args:
            audio_path (str)
        Raises:
            TypeError: "audio path must be 'str'"
            FileNotFoundError: "No file exists '<audio_path>'"
            FileError: "Invalid audio format <ext>"
            FileError: "Empty file <audio_path>"
        """
        # Check type
        if not isinstance(audio_path, str):
            raise TypeError("audio path must be 'str'" )
        
        # Check audio path exists
        if not osp.exists(audio_path):
            raise FileNotFoundError(f"File not found '{audio_path}'")
        
        # Check empty file
        if not osp.getsize(audio_path):
            raise FileError(f"File '{audio_path}' is empty")

        # Check audio format (file extension)
        ext = osp.splitext(audio_path)[1]  # '.wav'
        if not ext[1: ] in AUDIO_FORMATS:
            raise FileError(f"Invalid audio format '{ext}'")

    def validate_audio(self, audio_path):
        """
        Perfomance time ~ 48ms for 14sec audio
        Args:
            audio_path (str)
        Raises:
            AudioError: "Invalid codec <codec>."
            AudioError: "Invalid sample rate <sr>."
            AudioError: "Invalid format <format>."
        NOTE: These errors can be solved be FFMPEG convertation
            see convert_audio() method
        """
        media_info = self.media_info(audio_path)

        codec = media_info['streams'][0]['codec_name']
        sr = int(media_info['streams'][0]['sample_rate'])
        format_ = media_info['format'][0]['format_name']

        if codec != self.codec:
            raise AudioError("Invalid codec '{}'.".format(codec))

        if sr != self.sr:
            raise AudioError("Invalid sample rate '{}'".format(sr))

        if format_ != self.format:
            raise AudioError("Invalid format '{}'".format(format_))

    def validate_sample(self, sample, dur=None):
        """
        C - number of channels
        S - sample length
        Args:
            sample (tensor (C, S))
            dur (float or None): duration (sec) - (optionally)
        Raise:
            TooShortAudioError:
            TooLongAudioError:
        """
        if not dur:
            dur = sample.size()[1] / self.sr

        if dur < self.min_dur:
            msg = "Audio is too short: {:.2f} sec.".format(dur)
            raise TooShortAudioError(msg)

        if self.max_dur and dur > self.max_dur:
            msg = "Audio is too long: {:.2f} sec.".format(dur)
            raise TooLongAudioError(msg)

    def convert(self, audio_path_in, audio_path_out=None):
        """
        Using python lib FFMPEG convert audio
        Args:
            audio_path_in (str)
            audio_path_out (str or None): if None, 
                it will be like '<audio_path_in>_preprocessed.wav'
        Raises:
            ValueError: "Output audio <audio_path_out> 
                        must have '<self.format>' format."
        Return:
            str: audio_path_out
        """
        # check or define output audio path
        if audio_path_out and not audio_path_out.endswith(self.format):
            msg = f"Output audio '{audio_path_out}' must have "\
                  f"'{self.format}' format"
            raise ValueError(msg)
        
        if not audio_path_out:
            postfix =  '_preprocessed.' + self.format
            audio_path_out = osp.splitext(audio_path_in)[0] + postfix

        # convert
        if osp.exists(audio_path_out):
            os.remove(audio_path_out)  # it's like over-write
        stream = ffmpeg.input(audio_path_in)
        output = ffmpeg.output(stream, audio_path_out, **self.ffmpeg_convert_params)
        ffmpeg.run(output)

        return audio_path_out

    def from_audio_path(self, audio_path, rm_converted=False):
        """
        C - number of channels
        S - sample length
        Args:
            audio_path (str):
            rm_converted (bool): whether to remove convert output audio
        Raises:
            TypeError: "audio path must be 'str'"
            FileNotFoundError: "No file exists '<audio_path>'"
            FileError: "Invalid audio format <ext>"
            FileError: "Empty file <audio_path>"
            TooShortAudioError:
            TooLongAudioError:
        Return:
            tuple (
                tensor (C, S): sample
                dur (float): audio duration
           )
        """
        converted_audio = None
        self.validate_audio_file(audio_path)
        try:
            # it's pretty long ~ 90 ms
            self.validate_audio(audio_path)
        except:
            audio_path = self.convert(audio_path)
            converted_audio = audio_path
        # load audio
        sample, sr = self.get_sample(audio_path)
        dur = sample.shape[1] / sr
        # check sample
        self.validate_sample(sample, dur)
        # remove converted output audio
        if rm_converted and converted_audio:
            os.remove(converted_audio)
        return sample, dur



if __name__ == '__main__':
    config = utils.config_from_yaml('config.yaml')
    preprocess_params = config["preprocess"]

    audio_path = 'data/audios/example.wav'
    preprocess = AudioPreprocess(**preprocess_params)

    sample, dur = preprocess.from_audio_path(audio_path)
    features = preprocess.extract_features(sample)
