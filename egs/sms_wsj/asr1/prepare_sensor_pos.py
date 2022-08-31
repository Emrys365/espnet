import argparse
import json
import re
from pathlib import Path


def compact_jsonstring(string, ptn, max_line_length=120):
    def my_replace(match, max_length=max_line_length):
        s = " ".join(match.group().split())
        # match = match.group()
        return s if len(s) <= max_length else match.group()
    return re.sub(ptn, my_replace, string, flags=re.MULTILINE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonstr", type=str, help="Path to sms_wsj.json that is generated during data simulation")
    parser.add_argument("--outdir", type=str, default="data")
    args = parser.parse_args()

    with Path(args.jsonstr).open("r") as f:
        data = json.load(f)["datasets"]

    for dset, info in data.items():
        sensor_pos_json = {}
        for uttid, meta in info.items():
            sensor_pos = meta["sensor_position"]
            idx, u1, u2 = uttid.split("_")
            uid = u1[:3] + "_" + u2[:3] + "_" + uttid
            # transpose the 2D list
            sensor_pos_json[uid] = list(map(list, zip(*sensor_pos)))
        path = Path(args.outdir) / f"{dset}/sensor_pos.json"
        with open(path, "w") as f:
            s = json.dumps(sensor_pos_json, indent=4)
            s = compact_jsonstring(s, r"(?<=^\s{8}\[)\n((?:\s{12}.*\n)+)\s{8}(?=\])")
            f.write(s)
