#!/usr/bin/env python
# coding: utf-8

# # Project 1

# Example using the sklearn library:

# In[1]:


# Main functions
get_ipython().run_line_magic('matplotlib', 'widget')

from scipy.io import wavfile
import array
import scipy
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Latex, Audio
from numpy import sqrt
from sklearn.decomposition import FastICA

plt.style.use('default')
plt.interactive(True)

REFERENCE_MIX = "./data0proj.wav"

SIGNAL0_FILE = "./sample-01.wav"
SIGNAL1_FILE = "./sample-03.wav"
SIGNAL2_FILE = "./sample-02.wav"
SIGNAL3_FILE = "./sample-04.wav"

default_mix = [[1, 1], [0.5, 2]]
default_noise_level = 0.05


def ReadWavFile(path):
    samplerate, signal_raw = wavfile.read(path)
    samples = np.array([float(s)/2**15 for s in signal_raw])
    n_samples = len(samples)
    times = np.linspace(0, (1/samplerate)*n_samples, n_samples)
    return samples, times, samplerate
def PlotAudio(samples, samplerate, title=""):
    times = np.linspace(0, (1/samplerate)*len(samples), len(samples))
    fig = plt.figure()
    plt.plot(times, samples)
    if title != "":
        plt.title(title)
    plt.show()
    display(Audio(data=samples, rate=samplerate))
def AddNoise(data, level, f = np.random.normal):
    level = np.max(data) * level
    s = data + (f(size=data.shape) * level)
    return s * abs(1/np.max(s)) # normalized

# quality - mean squared error
def DecompositionQuality(original, decomposed):
    diff = original-decomposed
    mean = np.mean(diff)
    score = np.mean(np.array([x**2 for x in diff]))
    return score, mean, diff # quality, mean, diff.

def PlotDecompositionQuality(original, decomposed, label):
    quality, mean, diff = DecompositionQuality(original, decomposed)
    var = np.var(diff)
    plt.figure()
    plt.plot(diff, label="difference" )
    plt.axhline(y=var, color="g", linestyle=":", label="Variance: " + str(round(var, 4)))
    plt.title("Quality of " + label + " decomposition : " + str(quality))
    plt.legend()
    plt.show()
    return quality, mean, diff

# Assumption: the original signal will clearly have a corresponding decomposed signal. This might be unwise.
def AssignDecomposed(originals, decomposed):
    assert len(originals) == len(decomposed)
    #return decomposed
    values = []
    i = 0
    for o in originals:
        ii = 0
        for d in decomposed:
            q, _, _ = DecompositionQuality(o, d)
            values.append((q, i, ii))
            ii += 1
        i += 1
    taken = []
    values = sorted(values, key=lambda v: v[0])
    slots = [None] * len(originals)
    nt = 0
    for it in values:
        if nt == len(originals):
            break
        if slots[it[1]] is not None:
            continue
        if it[2] in taken:
            continue

        slots[it[1]] = decomposed[it[2]]
        taken.append(it[2])
        nt += 1
    assert nt == len(originals)
    print(values)
    print(slots)
    return slots


# In[2]:


# Reaading and generating data

signal0, times, signal0_samplerate = ReadWavFile(SIGNAL0_FILE)
signal1, times, signal1_samplerate = ReadWavFile(SIGNAL1_FILE)
signal2, times, signal2_samplerate = ReadWavFile(SIGNAL2_FILE)
signal3, times, signal3_samplerate = ReadWavFile(SIGNAL3_FILE)

assert signal0_samplerate == signal1_samplerate
assert signal2_samplerate == signal3_samplerate
assert signal1_samplerate == signal2_samplerate

signal0 = signal0 / np.linalg.norm(signal0)
signal1 = signal1 / np.linalg.norm(signal1)
signal2 = signal2 / np.linalg.norm(signal2)
signal3 = signal3 / np.linalg.norm(signal3)

samplerate = signal0_samplerate

display(Markdown("## Signal 0:"))
PlotAudio(signal0, samplerate, "Signal 0")

display(Markdown("## Signal 1:"))
PlotAudio(signal1, samplerate, "Signal 1")

# add noise
S = np.c_[AddNoise(signal0, default_noise_level), AddNoise(signal1, default_noise_level)]

S /= S.std(axis=0)  # Standardize data

A = np.array(default_mix)  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

mic0 = X[:,0]
mic1 = X[:,1]

PlotAudio(mic0, samplerate, "Microphone 0 reading")
PlotAudio(mic1, samplerate, "Microphone 1 reading")


# In[3]:


# Reference test

ica = FastICA(n_components=2, whiten="arbitrary-variance")
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

assigned = AssignDecomposed([signal0, signal1], [S_[:, 0] / np.linalg.norm(S_[:, 0]), S_[:, 1] / np.linalg.norm(S_[:, 1])])
reconstructed0 = assigned[0]
reconstructed1 = assigned[1]
assigned = None

PlotAudio(reconstructed0, samplerate, "Signal 0, reconstructed with sklearn")

PlotAudio(reconstructed1, samplerate, "Signal 1, reconstructed with sklearn")

display(Markdown(f"Reference quality for mixing matrix {default_mix}, noise={default_noise_level}"))

qref0, _, _ = PlotDecompositionQuality(signal0, reconstructed0, "signal 0")
qref1, _, _ = PlotDecompositionQuality(signal1, reconstructed1, "signal 1")

# conserve RAM
reconstructed0 = None
reconstructed1 = None


# # Implementation, basic test

# ## Whitening:
# $X$ - input
# 
# $X_w$ - Whitened output
# 
# $C$ - covariance matrix of $X$
# 
# $E$ - eigen vectors of $C$
# 
# $D$ - eigen values of $C$
# 
# $X_w = D^{-1/2}E^TX$

# In[4]:


from numpy import nan
import sys
import time
epsilon = sys.float_info.epsilon
def g(data):
    return np.tanh(data)
def g_prime(data):
    return 1 - np.tanh(data)**2
def g_both(data):
    g_ret = np.tanh(data)
    g_prime_ret = 1 - g_ret**2
    return g_ret, g_prime_ret

def FastICA(readings: np.ndarray, maxiter=1000, tolerance = 1e-7, showtime = True):
    start = time.time()
    n = readings.shape[0]
    nsamples = readings.shape[1]
    signals = []

    # center
    for i in range(0, n):
        signal : np.ndarray = readings[i, :]
        signal -= np.mean(signal) # zero the mean
        signals.append(signal)
    S = np.array(signals)

    # whiten the input data
    cov = np.cov(S)
    evals, evecs = np.linalg.eigh(cov)
    inv = np.diag(1.0 / np.sqrt(evals + epsilon))
    #whitened input
    Sw = inv @ evecs.T @ S
    iters = 0

    onescol = np.array([[1]] * nsamples)
    Wmats = []
    for p in range(0, n):
        Wp = np.random.uniform(size=(n, 1))
        i = 0
        while i < maxiter:
            WX = Wp.T @ Sw
            gW, gW_prime = g_both(WX)
            L = (1/nsamples)*(Sw @ gW.T)
            R = (1/nsamples) * (gW_prime @ onescol) @ Wp.T
            Wnew = L - R.T
            for w in Wmats:
                w = np.array([w]).T
                Wnew -= (Wnew.T @ w) * w
            Wnew = Wnew / np.linalg.norm(Wnew)
            dif = 1 - abs((Wp.T @ Wnew).item())
            if np.isnan(dif):
                Wp = np.random.uniform(size=(n, 1))
                print("Invalid iteration, got nan dif!")
            else:
                Wp = Wnew
            print(f"it: {i} dif: {dif}")
            if dif < tolerance:
                break
            i += 1
        if i == maxiter:
            print("Maximum number of iterations exceeded!")
        Wmats.append(np.array([Wp[i][0] for i in range(0, n)]))
        iters += i
    Wmats = np.array(Wmats)
    res = Wmats @ Sw
    # normalize the samples
    for i in range(0, res.shape[0]):
        res[i] = res[i, :] / np.linalg.norm(res[i, :])

    end = time.time()
    elapsed = end-start
    if showtime:
        display(Markdown(f"### Decomposed {n} inputs into {n} outputs in {iters} iterations, elapsing {elapsed}s"))

    return res, iters


# # Test I: two signals

# In[5]:


res, n_iters = FastICA(np.array([mic0, mic1]))

assigned = AssignDecomposed([signal0, signal1], [res[0, :], res[1, :]])
r0 = assigned[0]
r1 = assigned[1]
assigned = None

PlotAudio(r0, samplerate, "Reconstructed signal 0")

PlotAudio(r1, samplerate, "Reconstructed signal 1")

display(Markdown(f"Quality for mixing matrix {default_mix}, noise={default_noise_level}"))


q0, _, _ = PlotDecompositionQuality(signal0, r0, "signal 0")
display(Markdown(f"Reference quality for signal 0 = {qref0}"))
qdif0 = q0 - qref0
display(Markdown(f"Achieved quality for signal 0 = {q0} ({"+" if qdif0 > 0 else ""}{qdif0})"))

q1, _, _ = PlotDecompositionQuality(signal1, r1, "signal 1")
display(Markdown(f"Reference quality for signal 1 = {qref1}"))
qdif1 = q1 - qref1
display(Markdown(f"Achieved quality for signal 1 = {q1} ({"+" if qdif1 > 0 else ""}{qdif1})"))

r0 = None
r1 = None


# # Test II - 3 signals

# In[6]:


display(Markdown("## Signal 2:"))
PlotAudio(signal2, samplerate, "Signal 2")


default_mix = [[1, 1, 0.5], [0.5, 2, 1], [0.25, 1.5, 1.1]]
default_noise_level = 0.025

# add noise
S = np.c_[AddNoise(signal0, default_noise_level), AddNoise(signal1, default_noise_level), AddNoise(signal2, default_noise_level)]

S /= S.std(axis=0)  # Standardize data

A = np.array(default_mix)  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

mic0 = X[:,0]
mic1 = X[:,1]
mic2 = X[:,2]


display(Markdown("Mic 0:"))
display(Audio(data=mic0, rate=samplerate))

display(Markdown("Mic 1:"))
display(Audio(data=mic1, rate=samplerate))

display(Markdown("Mic 2:"))
display(Audio(data=mic2, rate=samplerate))

res, n_iters = FastICA(np.array([mic0, mic1, mic2]), tolerance=1e-8)

assigned = AssignDecomposed([signal0, signal1, signal2], [res[0, :], res[1, :], res[2, :]])
r0 = assigned[0]
r1 = assigned[1]
r2 = assigned[2]
assigned = None

PlotAudio(r0, samplerate, "Reconstructed signal 0")
PlotAudio(r1, samplerate, "Reconstructed signal 1")
PlotAudio(r2, samplerate, "Reconstructed signal 2")


# # Test III - 4 signals

# In[7]:


display(Markdown("## Signal 3:"))
PlotAudio(signal3, samplerate, "Signal 3")


default_mix = [[0.1, 1, 1, 0.5], [3, 0.5, 2, 1], [1.1, 0.25, 1.5, 1.1], [0.9, 1.8, 0.13, 0.3]]
default_noise_level = 0.0125

# add noise
S = np.c_[AddNoise(signal0, default_noise_level), AddNoise(signal1, default_noise_level), AddNoise(signal2, default_noise_level), AddNoise(signal3, default_noise_level)]

S /= S.std(axis=0)  # Standardize data

A = np.array(default_mix)  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

mic0 = X[:,0]
mic1 = X[:,1]
mic2 = X[:,2]
mic3 = X[:,3]

S = None


display(Markdown("Mic 0:"))
display(Audio(data=mic0, rate=samplerate))

display(Markdown("Mic 1:"))
display(Audio(data=mic1, rate=samplerate))

display(Markdown("Mic 2:"))
display(Audio(data=mic2, rate=samplerate))

res, n_iters = FastICA(np.array([mic0, mic1, mic2, mic3]), tolerance=1e-9)

assigned = AssignDecomposed([signal0, signal1, signal2, signal3], [res[0, :], res[1, :], res[2, :], res[3, :]])
r0 = assigned[0]
r1 = assigned[1]
r2 = assigned[2]
r3 = assigned[3]
assigned = None
res = None

PlotAudio(r0, samplerate, "Reconstructed signal 0")
PlotAudio(r1, samplerate, "Reconstructed signal 1")
PlotAudio(r2, samplerate, "Reconstructed signal 2")
PlotAudio(r3, samplerate, "Reconstructed signal 2")


# In[7]:




