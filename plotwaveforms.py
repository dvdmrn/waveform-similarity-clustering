import matplotlib.pyplot as plt
import librosa 
import librosa.display
import os 
import json
import math

vibfolder = "viblib"

with open("clusters.json","r") as f:
	jdict = json.load(f)

toLoad = jdict["5"]
nSubplots = len(toLoad)

rows = nSubplots
col=1
if nSubplots > 10:
	rows =  math.ceil(nSubplots/2)
	col = 2

if nSubplots > 20:
	rows = math.ceil(nSubplots / 3)
	col = 3


for i in range(0,len(toLoad)):

	plt.subplot(rows,col,i+1)
	y, sr = librosa.load(os.path.join(vibfolder,toLoad[i]))
	librosa.display.waveplot(y, sr=sr)
	plt.title(toLoad[i])

plt.show()


