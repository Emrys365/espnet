import argparse
import numpy as np
from pb_bss_eval.evaluation.module_srmr import srmr
import soundfile as sf
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scpfile", type=str, help="Path to the scp file containing paths to audio files")
    parser.add_argument("outfile", type=str, help="Path to the output file for writing results")
    parser.add_argument("--ref_channel", type=int, default=None, help="use the reference channel if audios are multi-channel")
    args = parser.parse_args()

    if args.ref_channel is not None:
        print(f"Warning: using ref_channel={args.ref_channel} for multi-channel input audios", flush=True)
    scores = []
    with open(args.outfile, "w") as out:
        with open(args.scpfile, "r") as f:
            data = f.readlines()
        for line in tqdm(data):
            if len(line.strip()) == 0:
                continue
            uttid, path = line.strip().split(maxsplit=1)
            wav, fr = sf.read(path)
            if wav.ndim > 1:
                wav = wav[:, args.ref_channel]
            score = srmr(wav, fr)
            scores.append(score)
            out.write(f"{uttid} {score}")
    print(f"===== Average SRMR score: {np.nanmean(scores)} =====", flush=True)
