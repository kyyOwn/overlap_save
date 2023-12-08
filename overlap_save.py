'''
@FilePath: overlap_save.py
@Author: Kangyuyong
@Date: 2023-03-30
@LastEditTime: 2023-12-08
@Description: 
@Email: yuyongkang1024@gmail.com.
'''

import os
import numpy as np


class OLS():
    def __init__(self, frame_size, rir_len, rir):
        self.frame_size = frame_size
        self.rir_len = rir_len
        self.fft_size = int(2**(np.ceil(np.log2(frame_size + rir_len))))
        self.rir = rir
        self.input_buf = np.zeros(self.fft_size)
        self.output_buf = np.zeros(self.fft_size)
        self.rir_spec = np.fft.rfft(self.rir, self.fft_size)
        return
    
    def process(self, in_sig):
        self.input_buf[0:self.frame_size] = in_sig
        self.input_buf = np.roll(self.input_buf, -self.frame_size)
        in_spec = np.fft.rfft(self.input_buf, self.fft_size)
        out_spec = in_spec * self.rir_spec
        self.output_buf = np.fft.irfft(out_spec)
        out_sig = self.output_buf[-self.frame_size:]
        return out_sig