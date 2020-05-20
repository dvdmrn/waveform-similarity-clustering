import matplotlib.pyplot as plt
import librosa
import librosa.display
from dtw import dtw
from numpy.linalg import norm
import numpy as np
from os import listdir, path
from tqdm import tqdm

signalFolder = "viblib"


files = listdir(signalFolder)

distanceMatrix = np.zeros([len(files),len(files)])

for i in tqdm(range(0,len(files))):
	f1 = signalFolder+"/"+files[i]
	y1, sr1 = librosa.load(f1)
	mfcc1 = librosa.feature.mfcc(y1, sr1)

	for j in range(0,len(files)):
		f2 = signalFolder+"/"+files[j]

		y2, sr2 = librosa.load(f2)
		mfcc2 = librosa.feature.mfcc(y2, sr2)
		dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))

		distanceMatrix[i][j] = dist


print(distanceMatrix)
plt.imshow(distanceMatrix, cmap='plasma', interpolation='nearest')
plt.show()

np.save("distance_matrix",distanceMatrix)

exit()
# def writeFile(toWrite):
# 	with open("distance_matrix.txt","w+") as dm:
# 		dm.write(toWrite)

# writeFile(distanceMatrix)



exit()





file1 = 'viblib/v-09-12-8-30.wav'
file2 = 'viblib/v-09-09-8-24.wav'

y2, sr2 = librosa.load(file2)


plt.subplot(1, 2, 1)
mfcc1 = librosa.feature.mfcc(y1, sr1)
librosa.display.specshow(mfcc1)

plt.subplot(1, 2, 2)
mfcc2 = librosa.feature.mfcc(y2, sr2)
librosa.display.specshow(mfcc2)

# plt.show()


dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print('Normalized distance between the two sounds:', dist)