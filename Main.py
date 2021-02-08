import numpy as np
import python_speech_features as speech
import signal_parametrization
import time
import sklearn.mixture._gaussian_mixture

print("start")
wektoro = np.random.rand(50000)
t = time.time()
framed = signal_parametrization.framing(
    wektoro, 16000, 0.025, 0.01, np.hamming)
t1 = time.time() - t
t = time.time()
framed2 = speech.sigproc.framesig(wektoro, 400, 160, np.hamming)
t2 = time.time() - t

mfcc = signal_parametrization.mfcc(wektoro, rate=41000, winlen=0.01,
                                   winstep=0.005,
                                   winfunc=lambda x: np.ones((x,)),
                                   ndft=512, lowfreq=0,
                                   highfreq=20500, nmfilter=41,
                                   ncep=13, Energy=True)
mfcc2 = speech.mfcc(wektoro, samplerate=41000, winlen=0.01, winstep=0.005,
                    winfunc=lambda x: np.ones((x,)), nfft=512, lowfreq=0,
                    highfreq=20500, nfilt=41, numcep=13, appendEnergy=True,
                    preemph=0, ceplifter=0)

gmm = sklearn.mixture._gaussian_mixture.GaussianMixture(n_components=100,
                                                        init_params='random')
gmm.fit(mfcc)
print("end")
