import glob
import os.path
import soundfile
import sys


args = sys.argv
if (len(args) > 1):
    path = args[1]

if not os.path.exists(path):
    raise FileNotFoundError(path)

if os.path.isfile(path):
    assert path.endswith('.scp')
    with open(path, 'r') as f:
        data = f.readlines()
    files = [line.split(' ')[-1].strip() for line in data]
else:
    files = glob.glob(os.path.join(path, '*.wav'))

num = len(files)
seconds = 0
for i, f in enumerate(files):
    t = soundfile.info(f).duration
    seconds += t
    print("[%d/%d] %d s" % (i + 1, num, t), end="\r", flush=True)

m, s = divmod(seconds, 60)
h, m = divmod(m, 60)

print("Duration of wavs in [%s] is %02d:%02d:%02d" % (path, h, m, s))

