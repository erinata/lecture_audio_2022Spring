import librosa
import librosa.display

import matplotlib.pyplot as pyplot



genre = "pop"
num = "00008"


audio_file = "./genres/" + genre + "/" + genre + "." + num + ".wav"
x, sr = librosa.load(audio_file, sr=44100)

# print(x)
# print(len(x))


wave_plot = pyplot.figure(figsize=(13,5))
librosa.display.waveplot(x, sr)
wave_plot.savefig("waveplot_" + genre + num + ".png")
pyplot.close()


stft_data = librosa.stft(x)
stft_data_db = librosa.amplitude_to_db(abs(stft_data))
spectrogram = pyplot.figure(figsize=(13,5))
librosa.display.specshow(stft_data_db, sr=sr, x_axis="time", y_axis="hz")
pyplot.colorbar()
spectrogram.savefig("spectrogram_" + genre + num + ".png")
pyplot.close()

hop_length=44100
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
chroma_plot = pyplot.figure(figsize=(13,5))
librosa.display.specshow(chromagram, x_axis="time", y_axis="chroma", hop_length=hop_length, cmap='coolwarm')
chroma_plot.savefig("chroma_plot_" + genre + num + ".png")
pyplot.close()











