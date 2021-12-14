import numpy as np
import jams
from scipy.io import wavfile
import librosa
import librosa.display

def preprocess_audio(data):
	data = data.astype(float)
	
	data = librosa.util.normalize(data)
	
	data = librosa.resample(data, sr_original, 22050)
	sr_curr = 22050
	
	data = np.abs(librosa.cqt(data,
	       hop_length=512, 
	       sr=sr_curr, 
	       n_bins=192, 
	       bins_per_octave=24))

	return data

file_audio = "C:/Users/Administrator/Downloads/TabCNN-master/TabCNN-master/data/GuitarSet/audio/audio_mic/00_Jazz3-137-Eb_comp_mic.wav"
file_anno = "C:/Users/Administrator/Downloads/TabCNN-master/TabCNN-master/data/GuitarSet/annotation/00_Jazz3-137-Eb_comp.jams"
jam = jams.load(file_anno)
sr_original, data = wavfile.read(file_audio)
sr_curr = sr_original
audio_cqt = np.swapaxes(preprocess_audio(data),0,1)
# output = {}
# output['repr'] = np.swapaxes(preprocess_audio(data),0,1)


string_midi_pitches = [40, 45, 50, 55, 59, 64]
highest_fret = 19
num_classes = highest_fret + 2


preproc_mode = "c"
downsample = True
normalize = True
sr_downs = 22050

cqt_n_bins = 192
cqt_bins_per_octave = 24

n_fft =2048
hop_length = 512

save_path = "spec_repr/" + "c" + "/"

frame_indices = range(len(audio_cqt))
times = librosa.frames_to_time(frame_indices, sr = 22050, hop_length=512)
# print(output)
# frame_indices = range(len(output["repr"]))

labels = []
for string_num in range(6):
	anno = jam.annotations["note_midi"][string_num]
	string_label_samples = anno.to_samples(times)
	for i in frame_indices:
		if string_label_samples[i] == []:
			string_label_samples[i] = -1
		else:
			string_label_samples[i] = int(round(string_label_samples[i][0]) - string_midi_pitches[string_num])
	labels.append([string_label_samples])

labels = np.array(labels)
labels = np.squeeze(labels)
labels = np.swapaxes(labels, 0, 1)




# # slice data based on tempo detection
# tempo = 120
# beat_per_sec = tempo / 60
# beat_num =  int(round(beat_per_sec *(data.size / 44100)))
# quaver_num = beat_num * 8
     
# quaver_position = np.zeros(quaver_num)
# for i in range(quaver_num):
#     quaver_position[i] = i * (1/ beat_per_sec /8)
# quaver_frames = librosa.time_to_frames(quaver_position, 22050, 512)

# # only the front half of quarters are unmuted
# # padding method
# j = 0
# for i in quaver_frames:
#     if (j%2) != 0:
#         audio_cqt[i:i+2] = 0
#         labels[i:i+2] = 0
#     j = j+1
# # drop method
# audio_cqt_quarter = np.zeros((quaver_num, 192))
# labels_quarter = np.zeros((quaver_num, 6))
# j = 0
# for i in quaver_frames:
#     if(j%2) == 0:
#         audio_cqt_quarter[j:j+2] = audio_cqt[i:i+2]
#         labels_quarter[j:j+2] = labels[i:i+2]
#     j = j + 1
    
    
    
# slice data based on onset detection
y, sr = librosa.load('C:/Users/Administrator/Downloads/TabCNN-master/TabCNN-master/data/GuitarSet/audio/audio_mic/00_Jazz3-137-Eb_comp_mic.wav')
librosa.onset.onset_detect(y=y, sr=sr, units='time')

o_env = librosa.onset.onset_strength(y, sr=sr)
times_env = librosa.times_like(o_env, sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

# import matplotlib.pyplot as plt
# D = np.abs(librosa.stft(y))
# fig, ax = plt.subplots(nrows=2, sharex=True)
# librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis = 'time', y_axis = 'log', ax=ax[0])
# ax[0].set(title='Power spectrogram')
# ax[0].label_outer()
# ax[1].plot(times, o_env, label='Onset strength')
# ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Onset')
# ax[1].legend()

audio_cqt_onset = np.zeros((onset_frames.shape[0] * 9, 192))
labels_onset = np.zeros((onset_frames.shape[0] * 9, 6))
for i, onset in enumerate(onset_frames):
    print(i, onset)
    audio_cqt_onset[i*9:i*9+9] = audio_cqt[onset-3:onset+6]
    labels_onset[i*9:i*9+9] = labels[onset-3:onset+6]


    


