import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, mir_eval, urllib
import librosa.display
import math


filename = 'viblib/v-09-10-4-23.wav'

y, fs = librosa.load(filename,sr=None)


# gets mfccs -----------------------------------
mfccs = librosa.feature.mfcc(y=y,)


# get zcr --------------------------------------

onset_frames = librosa.onset.onset_detect(y, sr=fs, delta=0.04, wait=4)
onset_times = librosa.frames_to_time(onset_frames, sr=fs)
onset_samples = librosa.frames_to_samples(onset_frames)


def extract_features(x, fs):
    zcr = librosa.zero_crossings(x).sum()
    energy = scipy.linalg.norm(x)
    return [zcr, energy]

frame_sz = math.floor(fs*0.090) # frame size?

features = numpy.array([extract_features(y[i:i+frame_sz], fs) for i in onset_samples])

print(features)

min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
features_scaled = min_max_scaler.fit_transform(features)


print (features_scaled.shape)
print (features_scaled.min(axis=0))
print (features_scaled.max(axis=0))

plt.scatter(features_scaled[:,0], features_scaled[:,1])
plt.xlabel('Zero Crossing Rate (scaled)')
plt.ylabel('Spectral Centroid (scaled)')

plt.show()


# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfccs, x_axis='time')
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# plt.show()


# librosa.display.waveplot(x, fs)
# plt.show()