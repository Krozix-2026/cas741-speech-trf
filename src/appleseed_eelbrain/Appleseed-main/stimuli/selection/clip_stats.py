# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from datetime import timedelta

from clips import read_clip_file


def fmt(t):
    s = t.total_seconds()
    m = s // 60
    s -= m * 60
    return f'{m:02.0f}:{s:02.0f}'


if __name__ == '__main__':
    total = timedelta(0)
    stimulus = timedelta(0)
    for tag, value in read_clip_file():
        if tag == 'file':
            print(f'(file {value})')
        elif tag == 'stimulus':
            print(f'Stimulus {value} ({fmt(stimulus)})')
            stimulus = timedelta(0)
        elif tag == 'clip':
            start, stop = value
            dt = stop - start
            print(f'{fmt(start)} - {fmt(stop)}: {fmt(dt)}')
            total += dt
            stimulus += dt
    print('-----------------------')
    print(f'Total: {fmt(total)}')
