import jiwer        # pip install jiwer
import numpy as np
from pathlib import Path
from pystoi import stoi
import soundfile as sf
import torch
import transformers # pip install transformers
from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer
import warnings

'''
Functions to compute the metrics for task 1 of the L3DAS21 challenge.
Both functions require numpy matrices as input and can compute only 1 batch at time.
'''


#TASK 1 METRICS
warnings.filterwarnings("ignore", category=FutureWarning)
transformers.logging.set_verbosity_error()
# run manually in advance to save the downloaded models in your specified directory:
# wer_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
# wer_model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h")
# wer_tokenizer.save_pretrained('facebook/wav2vec2-base-960h/tokenizer')
# wer_model.save_pretrained('facebook/wav2vec2-base-960h/model')
wer_tokenizer = Wav2Vec2Tokenizer.from_pretrained("./facebook/wav2vec2-base-960h/tokenizer")
wer_model = Wav2Vec2ForMaskedLM.from_pretrained("./facebook/wav2vec2-base-960h/model")


def wer(clean_speech, denoised_speech):
    """
    computes the word error rate(WER) score for 1 single data point
    """
    def _transcription(clean_speech, denoised_speech):

        # transcribe clean audio
        input_values = wer_tokenizer(clean_speech, return_tensors="pt").input_values
        logits = wer_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript_clean = wer_tokenizer.batch_decode(predicted_ids)[0]

        # transcribe
        input_values = wer_tokenizer(denoised_speech, return_tensors="pt").input_values
        logits = wer_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript_estimate = wer_tokenizer.batch_decode(predicted_ids)[0]

        return [transcript_clean, transcript_estimate]

    transcript = _transcription(clean_speech, denoised_speech)
    try:   # if no words are predicted
        wer_val = jiwer.wer(transcript[0], transcript[1])
    except ValueError:
        wer_val = None

    return wer_val

def task1_metric(clean_speech, denoised_speech, sr=16000):
    '''
    Compute evaluation metric for task 1 as (stoi+(1-word error rate)/2)
    This function computes such measure for 1 single datapoint
    '''
    WER = wer(clean_speech, denoised_speech)
    if WER is not None:  # if there is no speech in the segment
        STOI = stoi(clean_speech, denoised_speech, sr, extended=False)
        WER = np.clip(WER, 0., 1.)
        STOI = np.clip(STOI, 0., 1.)
        metric = (STOI + (1. - WER)) / 2.
    else:
        metric = None
        STOI = None
    return metric, WER, STOI


if __name__ == "__main__":
    import argparse
    from distutils.util import strtobool

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_scp", type=str, required=True, help="Paths to the .scp file containing reference signals")
    parser.add_argument("--enh_scp", type=str, required=True, help="Paths to the .scp file containing enhanced signals")
    parser.add_argument("--outdir", type=str, default="-", help="Path to the directory for storing output files")
    parser.add_argument("--out_suffix", type=str, default=".scp", help="suffix of the output files")
    parser.add_argument("--verbose", type=strtobool, default=False)
    args = parser.parse_args()

    ref = {}
    with open(args.ref_scp, "r") as f:
        for line in f:
            uttid, wav = line.strip().split(maxsplit=1)
            ref[uttid] = wav

    enh = {}
    with open(args.enh_scp, "r") as f:
        for line in f:
            uttid, wav = line.strip().split(maxsplit=1)
            enh[uttid] = wav

    metrics = {}
    stois = {}
    wers = {}
    assert ref.keys() == enh.keys()
    for uttid, ref_wavpath in ref.items():
        enh_wav, fr = sf.read(enh[uttid])
        ref_wav, _ = sf.read(ref_wavpath)
        metric, wer_score, stoi_score = task1_metric(ref_wav, enh_wav, sr=fr)
        metrics[uttid] = metric
        wers[uttid] = wer_score
        stois[uttid] = stoi_score
        if args.verbose:
            print(f"[{uttid}] metric={metric} wer={wer_score} stoi={stoi_score}")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f"metric{args.out_suffix}").open("w") as f:
        for uttid, metric in metrics.items():
            f.write(f"{uttid} {metric}\n")
    with (outdir / f"stoi{args.out_suffix}").open("w") as f:
        for uttid, stoi_score in stois.items():
            f.write(f"{uttid} {stoi_score}\n")
    with (outdir / f"wer{args.out_suffix}").open("w") as f:
        for uttid, wer_score in wers.items():
            f.write(f"{uttid} {wer_score}\n")
