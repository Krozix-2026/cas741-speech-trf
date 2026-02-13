# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path

import librosa
import numpy as np
import scipy.io.wavfile

from clips import read_clip_file


root = Path('~').expanduser() / 'Data' / 'Appleseed'
src = root / 'audio'
dst = root / 'stimuli'


def save_clips(clips, name, sr):
    x = np.concatenate(clips, 0)
    # presentation can't handle 32 bit wav files
    x *= 32767
    np.round(x, out=x)
    x = x.astype(np.int16)
    scipy.io.wavfile.write(dst / f'{name}.wav', sr, x)


if __name__ == '__main__':
    current_file = sr = None
    clips = []
    for tag, value in read_clip_file():
        if tag == 'file':
            print(f'file: {value}')
            current_file = str(src / f'{value}.MP3')
        elif tag == 'stimulus':
            i = value.split(' - ', 1)[0]
            save_clips(clips, f"segment {i}", sr)
            clips = []
        elif tag == 'clip':
            start, stop = value
            offset = start.total_seconds()
            duration = stop.total_seconds() - offset
            audio, sr = librosa.load(current_file, None, False, offset, duration)
            clips.append(audio)
