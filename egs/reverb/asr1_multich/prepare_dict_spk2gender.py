import json
from pathlib import Path
import re


if __name__ == "__main__":
    import sys
    args = sys.argv

    # first, fill out gender info for MC-WSJ-AV Dev and Eval sets
    speakers = {
        # MC-WSJ-AV Dev (5)
        "t6c": "F", "t7c": "F", "t8c": "M", "t9c": "M", "t10": "M",
        # MC-WSJ-AV Eval (10)
        "t21": "M", "t22": "M", "t23": "F", "t24": "F", "t25": "M",
        "t36": "F", "t37": "M", "t38": "M", "t39": "F", "t40": "F",
    }
    gender_map = {"male": "M", "female": "F"}
    if len(args) != 2:
        print("Usage: %s <path-to-wsjcam0>'" % args[0])
    else:
        rootdir = Path(args[1]).resolve()
        ifo_files = rootdir.rglob("*.ifo")
        for ifo in ifo_files:
            # .ifo example format:
            #
            #     ** Please fill in the appropriate fields **
            #
            # Talker Name		:
            #
            # Talker Sex		: male
            #
            # Talker Dialect		: Southern
            #
            # Talker Age		: 30
            #
            # Talker Contact Address	:
            #
            # Microphone Used (.lwv)	: Sennheiser_HMD414
            #
            # Microphone Used (.rwv)	: Canford_C100PB
            #
            #
            # Recorded by jfjf100 on 30-Nov-1993 15:11:30.00

            spk = ifo.parent.stem
            if spk in speakers:
                continue
            with ifo.open("r") as f:
                for line in f:
                    if line.strip().startswith("Talker Sex"):
                        gender = re.search(r'(?<=Talker Sex)\s*:\s*(\w+)', line).group(1)
                        break
            speakers[spk] = gender_map[gender.lower()]

        s = json.dumps(speakers, indent=4, ensure_ascii=False, sort_keys=True).encode('utf-8')
