import os
import sys
import soundfile as sf
import numpy as np
import argparse

from overlap_save import OLS

def main(args):
    sig_dir = args.sig_dir
    rir_dir = args.rir_dir
    if not os.path.isfile(sig_dir):
        raise FileNotFoundError(f'The signal file is not exist, please confirm the signal dir:{sig_dir}')
    if not os.path.isfile(rir_dir):
        raise FileNotFoundError(f'The signal file is not exist, please confirm the signal dir:{rir_dir}')
    sig, fs_sig = sf.read(sig_dir)
    rir, fs_rir = sf.read(rir_dir)
    assert(fs_sig == fs_rir)

    rir_len = int(1.0 * fs_rir)
    rir = rir[0: rir_len]

    frame_size = 128
    frame_num = int(len(sig) / frame_size)
    ols =  OLS(128, len(rir), rir)

    out_sig = np.zeros_like(sig)
    for i in range(frame_num):
        frame = sig[i * frame_size : (i + 1) * frame_size]
        out_frame = ols.process(frame)
        out_sig[i * frame_size : (i + 1) * frame_size] = out_frame
    
    out_dir = os.path.dirname(sig_dir)
    otu_name = 'output.wav'
    out_dir = os.path.join(out_dir, otu_name)
    sf.write(out_dir, out_sig, fs_sig)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sig_dir', type=str, default=f'data/speech.wav', help="the path of input signal")
    parser.add_argument('--rir_dir', type=str, default=f'data/rir.wav', help="the path of room impluse respone")
    args = parser.parse_args()
    main(args)