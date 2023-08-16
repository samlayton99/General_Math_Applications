# Wavelets Project Description

Wavelets, a pivotal mathematical technique for analyzing signals and images, provide a fresh lens to understand and process signals and images. Within this Jupyter Notebook, I offer an introduction to wavelets along with a Python embodiment of the discrete wavelet transform. Furthermore, I shed light on the application of wavelets in signal denoising and image compression realms.

## Installation

To set up your environment, ensure you install the following packages:

```python
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
%matplotlib inline
plt.rcParams['figure.figsize'] = [15,8]
import pywt
from scipy import linalg as la
import bitstring as bs
```

Use pip for installation:

```bash
pip install scipy matplotlib numpy imageio pywt bitstring
```

## Usage

Open `wavelets.ipynb` in Jupyter Notebook and proceed to run the cells in order. Every section is replete with explanations of the concepts and their Python implementations.

## Mathematical Background

Wavelets are mathematical tools used for the analysis of signals and images. Unlike Fourier transforms which depict a signal using sine and cosine functions of various frequencies, wavelets represent signals via wavelets of distinct scales and positions. This sparse representation is key to their wide applicability.

## Classes and Functions

### WaveletTransform

This class is designed to execute the discrete wavelet transform on data arrays (either 1D or 2D). Key methods include:

- `forward`: Implements the forward wavelet transform.
- `inverse`: Executes the inverse wavelet transform.
- `plot`: Provides a visualization of the wavelet transform.

### Signal

A class symbolizing a signal as a sample sequence. Noteworthy methods encompass:

- `__init__`: Initializes the signal.
- `add_noise`: Infuses Gaussian noise into the signal.
- `denoise`: Employs wavelet thresholding for noise removal.
- `plot`: Visualizes the signal and its wavelet transformation.

### Image

Representing an image via a 2D pixel array, this class offers:

- `__init__`: Image initialization.
- `compress`: Utilizes wavelet thresholding for image compression.
- `plot`: Displays the image and its wavelet transformation.

### Functions

- `plot_signal`: Depicts a signal using Matplotlib.
- `plot_image`: Showcases an image via Matplotlib.
- `plot_wavelet_transform`: Visualizes a signal or image's wavelet transform.

## Project Flow

1. **WaveletTransform**: The journey starts with this class, highlighting the discrete wavelet transform process on data arrays.
2. **Signal**: This class introduces signals, from their generation to their visualization post wavelet transformation.
3. **Image**: Images, from creation to compression, are explored next.
4. **Applications**: The final leg demonstrates wavelets in action - signal denoising and image compression, leveraging the aforementioned classes.
5. **Generalized Functions**: This portion shows the generalized use of wavelets on 1-Dimensional functions