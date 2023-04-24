"""
при использовании eval с флагом -p
Предобработать сырую запись с помощью
preprocess.sh
"""
import os
from librosa import load
from pydub.utils import mediainfo

params = {'sample_rate' : '8000',
          'channels' : '1',
          'codec_name' : 'pcm_s16le',
          'format_name' : 'wav'}

bash_script = 'preprocess.sh'

def check_sound(audio_path):
    """
    всевозможные проверки записи
    :param audio_path:
    :return: message: 'OK', 'Not Found', 'Empty File',
        'not sound format', 'Invalid param', 'Sound Error', 'too Short'
    + после этой функции нужно проверять на мин. длительность
    """
    # ------
    if not os.path.exists(audio_path):
        return 'Not Found'
    # ------
    file_name = os.path.split(audio_path)[1]                # ('test_sounds', 'test.mp3')
    file_name, file_ext = os.path.splitext(file_name)       # ('test' , '.mp3')
    if not file_ext in ['.wav', '.mp3', '.mpeg', '.flac']:
        return "not Sound Format: '{}'".format(file_ext)
    # ------
    if os.path.getsize(audio_path) == 0:
        return 'Empty File'
    # ------
    media_info = mediainfo(audio_path)
    # ------
    for key, value in params.items():
        if not media_info[key] == value:
            return 'Invalid {}'.format(key)
    # ------
    return 'OK'


def convert_audio(audio_path, wav_path=None):
    """
    normalization, freq, re-sampling
    :param audio_path: it has already been checked!
    :param wav_path: if None, preprocessed_audio.wav
    """
    if not wav_path:
        wav_path = 'preprocessed_audio.wav'
    command = 'sh ' + bash_script + ' ' + audio_path + ' ' + wav_path
    os.system(command)


if __name__ == '__main__':
    audio_path = 'test.wav'
    message = check_sound(audio_path)
    print(message)

    if 'Invalid' in message:
        print('preprocessing...')
        convert_audio(audio_path)