import numpy as np
import math as m
import logging
import scipy.fftpack as sc


def framing(signal, rate, winlen, winstep, winfunc=lambda x: np.ones((x,))):
    """function framing the signal
    signal: 1D array
    rate: signal rate in Hz
    winlen: length of window in seconds
    winstep: step between the windows in seconds
    winfunc: function of window, default no window is apllied
    """
    length = int(m.ceil(rate*winlen))
    step = int(m.ceil(rate*winstep))
    start = 0
    end = length
    numofframes = int(m.ceil((1+(len(signal)-length)/step)))
    framed = np.zeros((numofframes, length))
    row = 0
    if end >= len(signal):
        zero = np.zeros(int(length - len(signal[start:])))
        framed[row] = np.concatenate((signal[start:], zero), axis=None)
    else:
        while row < numofframes:
            if end <= len(signal):
                framed[row] = signal[start:end]
            else:
                zero = np.zeros(int(length-len(signal[start:])))
                framed[row] = np.concatenate((signal[start:], zero), axis=None)
            start += step
            end += step
            row += 1
    win = np.tile(winfunc(length), (numofframes, 1))
    return framed*win


def magspec(framed, ndft):
    """
    function return the power spectrum of each frame in framed.
    When frames is an NxK array, functions returns Nx(ndft/2+1) array
    ndft: length of DFT,It can't be grater than length of frame
    framed: an array of frames from signal
    ndft: the length of DFT
    """

    if np.shape(framed)[1] > ndft:
        logging.warning(
            "frame length (%d) is greater than DFT size (%d)",
            np.shape(framed)[1], ndft)
        logging.warning("frame will be truncated. Increase ndft to avoid.'")

    magspect = 1/ndft * np.square(np.absolute(np.fft.rfft(framed, ndft)))
    return magspect


def HztoMel(hz):
    """Function make a frequency conversion from Hz to Mel"""
    return 1127 * np.log(1 + hz/700)


def MeltoHz(mel):
    """Function make a frequency conversion from Mel to Hz"""
    return 700*(np.exp(mel/1127) - 1)


def melfreqwrap(magspectrum, rate, ndft,  lowfreq, highfreq, nmfilter):
    """
    Function creates a filter bank for MFCC features and
    returnes the dot of magspectrum and mel filter bank
    magspectrum: array of the power spectrum
    rate: signal rate in Hz
    ndft: length of DFT,It can't be grater than length of frame
    lowfreq: lowest frequency of Mel Bank Filter in Hz
    highfreq: highest frequency of Mel Bank Filter, in Hz, the highest is
    rate/2.
    nmfilter: Number of filters in Mel Bank Filter
    """

    highmelfreq = HztoMel(highfreq)
    lowmelfreq = HztoMel(lowfreq)
    melfreq = np.linspace(lowmelfreq, highmelfreq, nmfilter + 2)
    freq = MeltoHz(melfreq)
    fftbin = np.floor(((ndft+1)*freq)/rate)
    trgfbank = np.zeros([nmfilter, int(ndft / 2 + 1)])
    for n in range(0, nmfilter):
        for k in range(int(fftbin[n]), int(fftbin[n+1])):
            trgfbank[n, k] = (k-fftbin[n])/(fftbin[n+1]-fftbin[n])
        for k in range(int(fftbin[n+1]), int(fftbin[n+2])):
            trgfbank[n, k] = (fftbin[n+2]-k)/(fftbin[n+2]-fftbin[n+1])
    return np.dot(magspectrum, trgfbank.T)


def centerfreq(f, j, a, b, c):
    """
    Function computing the center frequency  filter expressed in linear-scale
    :param f: the upper or lower frequency of filter bank
    :param j: of upper frequency j = 1, for lower frequency j = 0
    :param a: numerical value.
        For ERB in polynominal format
        a = 6.23 * 10**(-6) , ERB = a* f^2 + b*f + c
    :param b: numerical value.
        For ERB in polynominal format
        b = 93.39 * 10**(-3), ERB = a* f^2 + b*f + c
    :param c: numerical value.
        For ERB in polynominal format
        c = 28.52, ERB = a* f^2 + b*f + c
    :return:
    """
    a__ = (-1) ** j * (0.5/(700+f))
    b__ = (-1) ** j * (700/(700+f))
    c__ = (-1) ** (j+1) * (0.5*f) * (1+(700/(700+f)))
    b_ = (b - b__) / (a - a__)
    c_ = (c - c__) / (a - a__)
    return 0.5 * (-b_ + m.sqrt(b_ ** 2 - 4 * c_))


def hmelfreqwrap(magspectrum, rate, ndft,  lowfreq, highfreq, nmfilter):
    """
    function based on:
    Speech emotion recognition using cepstral features extracted with
    novel triangular filter banks based on bark and ERB frequency
    scales
    Nagarajan Sugan, Nettimi Satya Sai Srinivas , Lakshmi Sutha Kumar,
    Malaya Kumar Nath, Aniruddha Kanhe
    Function creates a filter bank for HFCC features and returnes the
    dot of magspectrum and mel filter bank
    magspectrum: array of the power spectrum
    rate:   signal rate in Hz
    ndft:   length of DFT,It can't be grater than length of frame
    lowfreq:    lowest frequency of Mel Bank Filter in Hz
    highfreq:   highest frequency of Mel Bank Filter, in Hz, the highest is
                rate/2
    nmfilter:   Number of filters in Mel Bank Filter
    """
    a = 6.23 * 10**(-6)
    b = 93.39 * 10**(-3)
    c = 28.52
    lowfreq = centerfreq(lowfreq, 0, a, b, c)
    highfreq = centerfreq(highfreq, 1, a, b, c)
    mlowfreq = HztoMel(lowfreq)
    mhighfreq = HztoMel(highfreq)
    melfreq = np.zeros(nmfilter + 2)
    melfreq[0] = mlowfreq
    melfreq[-1] = mhighfreq
    for i in range(1, nmfilter):
        melfreq[i] = mlowfreq + i * ((mhighfreq - mlowfreq)/(nmfilter+1))
    melfreq = MeltoHz(melfreq)
    flowlim = np.zeros(nmfilter + 2)
    fuplim = np.zeros(nmfilter + 2)
    for i in range(0, nmfilter+1):
        ERB = a*melfreq[i] ** 2 + b*melfreq[i] + c
        flowlim[i] = -(700 + ERB) + m.sqrt((700 + ERB) ** 2
                                           + melfreq[i]*(melfreq[i] + 1400))
        fuplim[i] = flowlim[i] + 2 * ERB
    freq = np.concatenate((melfreq, flowlim, fuplim), axis=0)
    freq = np.sort(freq, axis=None)
    fftbin = np.floor(((ndft + 1) * freq) / rate)
    trgfbank = np.zeros([nmfilter, int(ndft / 2 + 1)])
    for n in range(0, nmfilter):
        for k in range(int(fftbin[n]), int(fftbin[n + 1])):
            trgfbank[n, k] = (k - fftbin[n]) / (fftbin[n + 1] - fftbin[n])
        for k in range(int(fftbin[n + 1]), int(fftbin[n + 2])):
            trgfbank[n, k] = (fftbin[n + 2] - k) / (
                fftbin[n + 2] - fftbin[n + 1])
    return np.dot(magspectrum, trgfbank.T)


def mfcc(signal, rate=41000, winlen=0.01, winstep=0.005,
         winfunc=np.hamming, ndft=512, lowfreq=0,
         highfreq=20500, nmfilter=41, ncep=13,
         Energy=True, pre_emphasis=0.95, cep_lifter=22):
    """
    Function compute the MFCC from an audio signal

    signal: 1D array of audio signal
    rate: signal sample rate in Hz, default 41 kHz
    winlen:length of window in seconds, default 10 ms
    winstep: step between the windows in seconds, default 5 ms
    winfunc: function of window, default, Hamming window
    ndft: length of DFT, default 512
    lowfreq:    lowest frequency of Mel Bank Filter in Hz, default is 0
    highfreq:   highest frequency of Mel Bank Filter, the highest
                is rate/2 in Hz, default is 20,5 kHz
    nmfilter: Number of filters in Mel Bank Filter, default is 41
    ncep: Number of coefficients, default is 13
    Energy: if this is true, the zeroth cepstral coefficient
            the log of the total frame energy, default is True
    pre_emphasis:  coefficient of pre emphasis filter,
                    0 is no filter, default is 0.95
    cep_lifter: apply sinusoidal liftering, 0 no liftering, default is 22
    """
    emphasized_signal = np.append(signal[0], signal[1:] -
                                  pre_emphasis * signal[:-1])
    framed = framing(emphasized_signal, rate, winlen, winstep, winfunc)
    magspectrum = magspec(framed, ndft)
    assert highfreq <= rate / 2, "highfreq is greater than samplerate/2"
    melspectrum = melfreqwrap(magspectrum, rate, ndft, lowfreq,
                              highfreq, nmfilter)
    melspectrum = np.where(melspectrum == 0, np.finfo(float).epsneg,
                           melspectrum)
    # melspectrum can not be zero, problems with log
    melspectrum = np.log(melspectrum)
    mfcc_ = sc.dct(melspectrum, type=2, axis=1, norm='ortho')[:, :ncep]
    if Energy:
        energy = np.sum(magspectrum, 1)
        energy = np.where(energy == 0, np.finfo(float).epsneg, energy)
        # energy can not be zero, problems with log
        mfcc_[:, 0] = np.log(energy)
        # first cepstral coefficient is repleced by log of frame energy
    if cep_lifter > 0:   # cep_lifter <= 0, do nothing
        (nframes, ncoeff) = mfcc_.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc_ *= lift
    return mfcc_


def hfcc(signal, rate=41000, winlen=0.01, winstep=0.005,
         winfunc=np.hamming, ndft=512, lowfreq=0,
         highfreq=20500, nmfilter=41, ncep=13, Energy=True,
         pre_emphasis=0.95, cep_lifter=22):
    """
    Function compute the HFCC from an audio signal

    signal: 1D array of audio signal
    rate: signal sample rate in Hz, default 41 kHz
    winlen:length of window in seconds, default 10 ms
    winstep: step between the windows in seconds, default 5 ms
    winfunc: function of window, default, Hamming window
    ndft: length of DFT, default 512
    lowfreq:    lowest frequency of Mel Bank Filter in Hz, default is 0
    highfreq:   highest frequency of Mel Bank Filter,
                the highest is rate/2 in Hz, default is 20,5 kHz
    nmfilter:   Number of filters in Mel Bank Filter, default is 41
    ncep:   Number of coefficients, default is 13
    Energy: if this is true, the zeroth cepstral coefficient the log
            of the total frame energy, default is True
    pre_emphasis:   coefficient of pre emphasis filter,
                    0 is no filter, default is 0.95
    cep_lifter: apply sinusoidal liftering,
                0 no liftering, default is 22
    """
    emphasized_signal = np.append(signal[0], signal[1:] -
                                  pre_emphasis * signal[:-1])
    framed = framing(emphasized_signal, rate, winlen, winstep, winfunc)
    magspectrum = magspec(framed, ndft)
    assert highfreq <= rate / 2, "highfreq is greater than samplerate/2"
    hmelspectrum = hmelfreqwrap(magspectrum, rate, ndft,
                                lowfreq, highfreq, nmfilter)
    hmelspectrum = np.where(hmelspectrum == 0,
                            np.finfo(float).epsneg, hmelspectrum)
    # melspectrum can not be zero, problems with log
    # melspectrum = np.log(hmelspectrum) # ! NEVER USED?
    hfcc_ = sc.dct(hmelspectrum, type=2, axis=1,
                   norm='ortho')[:, :ncep]
    if Energy:
        energy = np.sum(magspectrum, 1)
        energy = np.where(energy == 0, np.finfo(float).epsneg, energy)
        # energy can not be zero, problems with log
        hfcc_[:, 0] = np.log(energy)
        # first cepstral coefficient is repleced by log of frame energy

    if cep_lifter > 0:  # cep_lifter <= 0, do nothing
        (nframes, ncoeff) = hfcc_.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        hfcc_ *= lift
    return hfcc_
