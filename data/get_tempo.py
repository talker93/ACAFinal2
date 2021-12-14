#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import madmom
import numpy as np
import scipy.stats

beats = madmom.features.beats.RNNBeatProcessor()("test.wav")
when_beats = madmom.features.beats.BeatTrackingProcessor(fps=100)(beats)
m_res = scipy.stats.linregress(np.arange(len(when_beats)),when_beats)

first_beat = m_res.intercept 
beat_step = m_res.slope

print("bpm = ", round(60/beat_step))