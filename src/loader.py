import os
import numpy as np
import soundfile as sf

def flattenToMono(data):
    output = np.copy(data)
    if len(data.shape) == 2:
        output = output[:, 0].flatten()
    else:
        output = output.flatten()
    return output


def loadFiles(path):
    outputData, outputSampleRates = list(), list()
    for (directoryPath, directoryName, fileName) in os.walk(path):
        for file in fileName:
            if file.endswith('.wav'):
                tempPath = os.getcwd() + '/' + os.path.join(directoryPath,
                                                            file)
                data, samplerate = sf.read(tempPath)
                outputData.append(flattenToMono(data))
                outputSampleRates.append(samplerate)
    return outputData, outputSampleRates