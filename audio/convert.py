import os
import librosa.display
import matplotlib.pyplot  as plt
import numpy as np

def wav2spec_file(wvpath, pngpath, size):
    datadir = os.listdir(wvpath)
    for file in datadir:
        if file[-3:] == 'wav' and not os.path.isfile(os.path.join(pngpath, file[:-3]+'png')):
            tempath = os.path.join(wvpath, file)
            y, sr = librosa.load(tempath)
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

            # Convert to log scale (dB). We'll use the peak power as reference.
            log_S = librosa.amplitude_to_db(S, ref=np.max)

            # Make a new figure
            plt.figure(figsize=(size/100, size/100))
            librosa.display.specshow(log_S, sr=sr)
            plt.savefig(os.path.join(pngpath, file[:-3]+'png'), dpi=100)
            plt.close()
            
wav2spec_file('wav', 'spec256', 256)