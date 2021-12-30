import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("jsonstr", type=str, help="a raw json string or a json file")
parser.add_argument("--keys", type=str, nargs="+", default="wer", help="a list of keys to plot")
parser.add_argument("--out", type=str, default="all_snapshots_results.png")
args = parser.parse_args()


styles = [
    "seaborn-white",
    "dark_background",
    "bmh",
    "ggplot",
    "fivethirtyeight",
]
plt.style.use(styles[2])

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12


if Path(args.jsonstr).exists():
    with Path(args.jsonstr).open("r") as f:
        rets = json.load(f)
else:
    rets = json.loads(args.jsonstr)

linestyles_str = [
    ":",    # dotted
    "-",    # solid
    "--",   # dashed
    "-.",   # dashdot
]
marker_str = [
    "o",    # circle
    "v",    # triangle_down
    "s",    # square
    "^",    # triangle_up
    "<",    # triangle_left
    ">",    # triangle_right
    ",",    # pixel
    "1",    # tri_down
    "2",    # tri_up
    "3",    # tri_left
    "4",    # tri_right
    "8",    # octagon
    "p",    # pentagon
    "P",    # plus (filled)
    "*",    # star
    "h",    # hexagon1
    "H",    # hexagon2
    "+",    # plus
    "x",    # x
    "X",    # x (filled)
    "D",    # diamond
    "d",    # thin diamond
    "|",    # vline
    "_",    # hline
    ".",    # point
]

dsets = rets.keys()
keys = list(rets.values())[0].keys()
num_keys = len(keys)
fig = plt.figure(figsize=(6.5, 4 * num_keys))
# set the spacing between axes
plt.subplots_adjust(hspace=0.5)
for i, key in enumerate(keys):
    ax = fig.add_subplot(num_keys, 1, i + 1)
    ax.set_title(key.upper())
    for j, dset in enumerate(dsets):
        values = rets[dset][key]
        linestyle=linestyles_str[j % len(linestyles_str)]
        marker = marker_str[j % len(marker_str)]
        ax.plot(range(1, len(values) + 1), values, linewidth=2, linestyle=linestyle, marker=marker, label=dset)
    ax.set_xlabel("epoch")
    ax.set_ylabel(key.upper())
    ax.legend(loc='best', numpoints=1, fancybox=True)

fig.savefig(args.out)
print("Image generated: %s" % args.out)
