import librosa
import pandas

import glob
import numpy

import os

genres = ["blues","country","hiphop","metal","reggae","classical","disco","jazz","pop","rock"]
column_names = ["filename","spectral_centroids","spectral_rolloff","bandwidth","zero_crossing","chromagram","rms","mfcc0","mfcc1","mfcc2","mfcc3","mfcc4","mfcc5","mfcc6","mfcc7","mfcc8","mfcc9","mfcc10","mfcc11","mfcc12","mfcc13","mfcc14","mfcc15","mfcc16","mfcc17","mfcc18","mfcc19","genre"]

dataset = pandas.DataFrame(columns=column_names)

old_data = []
if os.path.exists("dataset.csv"):
	old_data = pandas.read_csv("dataset.csv")["filename"].values
else:
	dataset.to_csv("dataset.csv", index=None)


for genre in genres:
	print(genre)
	for filename in glob.glob("./genres/" + genre + "/*.wav"):
		if (filename in old_data):
			print(filename, " exists.")
		else:
			print(filename, " processing.")
			x, sr = librosa.load(filename, sr=44100)
			chromagram = librosa.feature.chroma_stft(x, sr=sr)
			spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
			spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
			bandwidth = librosa.feature.spectral_bandwidth(x, sr=sr)
			zero_crossing = librosa.feature.zero_crossing_rate(x)
			mfcc = mfcc = librosa.feature.mfcc(x, sr=sr)
			rms = librosa.feature.rms(x)

			dataset = pandas.DataFrame(columns=column_names)
			dataset = dataset.append({
				"filename": filename,
				"spectral_centroids": numpy.mean(spectral_centroids),
				"spectral_rolloff": numpy.mean(spectral_rolloff),
				"bandwidth": numpy.mean(bandwidth),
				"zero_crossing": numpy.mean(zero_crossing),
				"chromagram": numpy.mean(chromagram),
				"rms": numpy.mean(rms),
				"mfcc0":numpy.mean(mfcc[0]),
				"mfcc1":numpy.mean(mfcc[1]),
				"mfcc2":numpy.mean(mfcc[2]),
				"mfcc3":numpy.mean(mfcc[3]),
				"mfcc4":numpy.mean(mfcc[4]),
				"mfcc5":numpy.mean(mfcc[5]),
				"mfcc6":numpy.mean(mfcc[6]),
				"mfcc7":numpy.mean(mfcc[7]),
				"mfcc8":numpy.mean(mfcc[8]),
				"mfcc9":numpy.mean(mfcc[9]),
				"mfcc10":numpy.mean(mfcc[10]),
				"mfcc11":numpy.mean(mfcc[11]),
				"mfcc12":numpy.mean(mfcc[12]),
				"mfcc13":numpy.mean(mfcc[13]),
				"mfcc14":numpy.mean(mfcc[14]),
				"mfcc15":numpy.mean(mfcc[15]),
				"mfcc16":numpy.mean(mfcc[16]),
				"mfcc17":numpy.mean(mfcc[17]),
				"mfcc18":numpy.mean(mfcc[18]),
				"mfcc19":numpy.mean(mfcc[19, ]),
				'genre': genre
				}, ignore_index=True)
			dataset.to_csv("dataset.csv", index=None, header=False, mode="a")










