# Helper functions
from scipy.io import wavfile
import array
import scipy
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Latex, Audio
from numpy import sqrt
from sklearn.decomposition import FastICA
import scipy.signal as sps
import scipy.stats as stat

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

def DecompositionQuality(original, decomposed):
    score = np.corrcoef(original, decomposed, "valid")[1]
    return abs(score[0]), score[0] # score, correlation
def Normalize(dt):
    return np.array(dt) / np.sum(dt)

def GetFullCorrelation(s0, s1):
    fc = sps.correlate(s0, s1, "full")
    return np.array(fc * np.max(fc))

def PlotDecompositionQuality(original, decomposed, label, playable=False, samplerate=0, to_compare=None, compare_label="", playable_compare = False):
    score, correlation = DecompositionQuality(original, decomposed)
    full_correlation = GetFullCorrelation(original, decomposed)
    
    comp = to_compare is not None

    comp_score, comp_correlation = (None, None) if not comp else DecompositionQuality(original, to_compare)
    comp_full_correlation = None if not comp else GetFullCorrelation(original, decomposed)

    fig, ax = plt.subplots(3) if not comp else plt.subplots(3, 2)
    plt.tight_layout()

    display(Markdown("### Absolute correlation coefficient: " + str(score) + ("" if not comp else " vs " + str(comp_score) + " for " + compare_label)))

    def PlotAxis(ax, original, decomposed, full_correlation, score, label, original_label):
        var = np.var(full_correlation)
        mean = np.mean(full_correlation)
        ax[0].plot(decomposed, label = "Decomposed " + label)
        ax[0].set_title("Decomposed " + label)
        ax[0].legend()

        ax[1].plot(original, label="Original " + original_label)
        ax[1].set_title("Original " + original_label)
        ax[1].legend()

        ax[2].plot(full_correlation, label = "Discreate linear cross-correlation of " + label)
        ax[2].set_title("Correlation, avg=" + str(score))
        ax[2].axhline(y=sqrt(var), color="g", linestyle=":", label="Standard devation: ≈" + str(round(var, 5)))
        ax[2].axhline(y=-sqrt(var), color="g", linestyle=":")
        ax[2].axhline(y=mean, color="r", linestyle="-.", label="Mean: ≈" + str(round(mean, 5)))
        ax[2].legend()

    if comp:
        PlotAxis([ax[0][0], ax[1][0], ax[2][0]], original, decomposed, full_correlation, score, label, label)
        PlotAxis([ax[0][1], ax[1][1], ax[2][1]], original, to_compare, GetFullCorrelation(original, to_compare), np.corrcoef(original, to_compare), compare_label, label)
    else:
        PlotAxis(ax, original, decomposed, full_correlation, score, label, label)

    plt.show()
    if playable:
        assert samplerate != 0
        display(Markdown("Output:"))
        display(Audio(data = decomposed, rate=samplerate, ))
    if playable_compare:
        assert samplerate != 0
        display(Markdown("Sklearn output:"))
        display(Audio(data = decomposed, rate=samplerate, ))
    #print(full_correlation)
    return score, correlation, full_correlation

def AssignDecomposed(originals, decomposed):
    assert len(originals) == len(decomposed)
    #return decomposed
    values = []
    i = 0
    for o in originals:
        ii = 0
        for d in decomposed:
            q, _ = DecompositionQuality(o, d)
            values.append((q, i, ii))
            ii += 1
        i += 1
    taken = []
    values = sorted(values, key=lambda v: -v[0])
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
    #print(values)
    #print(slots)
    return slots
def MatrixToLatex(mat:np.array, command="bmatrix")->str:
    ret = ""
    if len(mat.shape) == 1:
        for i in range(0, mat.shape[0]):
            ret += str(mat[i])
            if i != mat.shape[0] - 1:
                ret += "&"
    else:
        for i in range(0, mat.shape[0]):
            for ii in range(0, mat.shape[1]):
                ret += str(mat[i, ii])
                if ii != mat.shape[1] - 1:
                    ret += "&"
            if i != mat.shape[1] - 1:
                ret += "\\\\"
    return "\\begin{" + command + "} " + ret + "\\end{" + command + "}"