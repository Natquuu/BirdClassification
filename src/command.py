import os
import pydub as pd

class BirdRecordingFactory:
   def __init__(self, path):
      self.path = path
      self.commandMap = dict(: list())

   """ Acquire all paths to mp3 files and add them to the |commandMap|
   dictionary based on the directory where recording is located, which
   represents classification of the bird. """
   def acquireBirdRecordingsPaths(self):
      outputData, outputSampleRates = list(), list()
      for (directoryPath, directoryName, fileName) in os.walk(path):
         self.commandMap[directoryName] = list()
         for file in fileName:
            if file.endswith('.mp3'):
               tempPath = os.getcwd() + '/' + os.path.join(directoryPath,
                                                                file)
               self.commandMap[directoryName].append(tempPath)

   def getAllClassificationsAvailable(self):
      return self.commandMap.keys()

