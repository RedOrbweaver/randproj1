# Helper functions
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
plt.rcParams['figure.figsize'] = [10, 5]

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