import os
import numpy as np
import soundfile as sf

from pydub import AudioSegment
from pydub.playback import play
from sklearn.mixture import GaussianMixture
from python_speech_features import mfcc

import matplotlib.pyplot as plt

n_components = 12
covariance_type = 'diag'
tol = 1e-3
max_iter = 100

validation_length = 100


# def flattenToMono(data):
#     output = np.copy(data)
#     if len(data.shape) == 2:
#         output = output[:, 0].flatten()
#     else:
#         output = output.flatten()
#     return output


# def loadFiles(path):
#     outputData, outputSampleRates = list(), list()
#     for (directoryPath, directoryName, fileName) in os.walk(path):
#         for file in fileName:
#             if file.endswith('.wav'):
#                 tempPath = os.getcwd() + '/' + os.path.join(directoryPath,
#                                                             file)
#                 data, samplerate = sf.read(tempPath)
#                 outputData.append(flattenToMono(data))
#                 outputSampleRates.append(samplerate)
#     return outputData, outputSampleRates


def main():
    path =  './data'
    recordings, samplerates = loadFiles(path)

    # Classification
    classificators = list()
    for recordingIndex in range(len(recordings)):
        classificators.append(
            GaussianMixture(n_components=n_components,
                            covariance_type=covariance_type,
                            tol=tol, max_iter=max_iter))

    features = list()
    # Features
    for recordingIndex in range(len(recordings)):
        features.append(mfcc(recordings[recordingIndex],
                             samplerate=samplerates[recordingIndex],
                             winfunc=lambda x: np.hamming(x)))

    # Trainning
    for featuresIndex in range(len(features)):
        classificators[featuresIndex].fit(
            features[featuresIndex])

    # Validation Test
    score = 0
    for testNumber in range(validation_length):
        currentRecordingNumber = testNumber % 2
        recording = recordings[currentRecordingNumber]
        currentMfcc = mfcc(recording,
                           samplerate=samplerates[currentRecordingNumber],
                           winfunc=lambda x: np.hamming(x))

        loglikelihoods = list()
        for classificatorIndex in range(len(classificators)):
            loglikelihoods.append(
                classificators[classificatorIndex].score(currentMfcc))

        max_index = loglikelihoods.index(max(loglikelihoods))
        if max_index == currentRecordingNumber:
            score += 1

    # print score
    print("The overall score: " +
          str(float(score) / validation_length * 100))


def loadPydubFile():
    path = './data/comstar/XC384314.mp3'
    data = AudioSegment.from_mp3(path)
    play(data)
    # plt.plot(data)
    # plt.show()



if __name__ == "__main__":
    # main()
    loadPydubFile()
