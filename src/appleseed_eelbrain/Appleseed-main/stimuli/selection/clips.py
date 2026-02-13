# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path
from datetime import datetime, timedelta


path = Path('clips.txt')
time_fmt = '%M:%S.%f'


def read_time(string):
    t = datetime.strptime(string, time_fmt)
    return timedelta(minutes=t.minute, seconds=t.second, microseconds=t.microsecond)


def read_clip_file(istart=0):
    stimulus_i = istart
    for line in path.open('r'):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        tag, value = line.split(maxsplit=1)
        if tag == 'file':
            pass
        elif tag == 'stimulus':
            value = f'{stimulus_i} - {value}'
            stimulus_i += 1
        elif tag == 'clip':
            start, stop = map(read_time, value.split('-'))
            value = (start, stop)
        else:
            raise ValueError(f"Unknown tag {tag!r} in line: {line!r}")
        yield tag, value
