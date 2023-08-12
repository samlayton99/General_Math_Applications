from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
plt.rcParams['figure.figsize'] = [15,8]
import pywt
from scipy import linalg as la




class WSQ:
    """Perform image compression using the Wavelet Scalar Quantization
    algorithm. This class is a structure for performing the algorithm. To
    actually perform the compression and decompression, use the _compress
    and _decompress methods respectively. Note that all class attributes 
    are set to None in __init__, but their values are initialized in the 
    compress() method.
    
    Attributes:
        _pixels (int): Number of pixels in source image.
        _s (float): Scale parameter for image preprocessing.
        _m (float): Shift parameter for image preprocessing.
        _Q ((16, ), ndarray): Quantization parameters q for each subband.
        _Z ((16, ), ndarray): Quantization parameters z for each subband.
        _bitstrings (list): List of 3 BitArrays, giving bit encodings for 
            each group.
        _tvals (tuple): Tuple of 3 lists of bools, indicating which 
            subbands in each groups were encoded.
        _shapes (tuple): Tuple of 3 lists of tuples, giving shapes of each 
            subband in each group.
        _huff_maps (list): List of 3 dictionaries, mapping Huffman index to 
            bit pattern.
    """
    
    def __init__(self):
        self._pixels = None
        self._s = None
        self._m = None
        self._Q = None
        self._Z = None
        self._bitstrings = None
        self._tvals = None
        self._shapes= None
        self._huff_maps = None
        self._infoloss = None
    
    def decompose(self, img):
        """Decompose an image into the WSQ subband pattern using the
        Coiflet1 wavelet.
        
        Parameters:
            ((m,n) ndarray): NumPy array holding the image to be 
                decomposed.
            
        Returns:
            (list): List of 16 NumPy arrays containing the WSQ 
                subbands in order.
        """
        # Use the PyWavelets library to perform the decomposition and initialize the subbands list.
        coeffs = pywt.wavedec2(img, 'coif1', level=5)
        subbands = [coeffs[0]]

        # Loop through the coefficients and add them to the subbands list.
        for i in range(1, len(coeffs)):
            for j in range(len(coeffs[i])):
                subbands.append(coeffs[i][j])

        # Return the subbands list.
        return subbands
    
    def recreate(self, subbands):
        """Recreate an image from the 16 WSQ subbands.
        
        Parameters:
            subbands (list): List of 16 NumPy arrays containing the WSQ 
                subbands in order.
            
        Returns:
            ((m,n) ndarray): NumPy array, the image recreated from the 
                WSQ subbands.
        """
        # Adjust the subbands list to match the format required by the PyWavelets library.
        adjusted = [subbands[0]]
        for i in range((len(subbands) - 1)// 3):
            adjusted.append((subbands[3*i + 1], subbands[3*i+2], subbands[3*i+3]))

        # Use the PyWavelets library to perform the reconstruction.
        return pywt.waverec2(adjusted, 'coif1')

    def quantize(self, coeffs, Q, Z):
        """A uniform quantizer. Map wavelet coefficients to 
        integer values using the quantization parameters Q and Z.
        
        Parameters:
            coeffs ((m,n) ndarray): Contains the floating-point values to 
                be quantized.
            Q (float): The step size of the quantization.
            Z (float): The null-zone width (of the center quantization bin).
            
        Returns
            ((m,n) ndarray): NumPy array of the quantized values.
        """
        # Make sure the coefficients are a NumPy array.
        coeffs = np.array(coeffs)

        # If Q is 0, return an array of zeros.
        if Q == 0:
            return np.zeros(coeffs.shape).astype(int)

        # Make a mask of all the coefficients that greater than Z/2 and if they are less than -Z/2.
        greaterMask = coeffs > Z/2.0
        zeroMask = np.abs(coeffs) <= Z/2.0
        lessMask = coeffs < -Z/2.0

        # Get the quantized values and return them in an array.
        coeffs[greaterMask] = np.floor((coeffs[greaterMask] - Z/2)/Q).astype(int) + 1
        coeffs[zeroMask] = 0
        coeffs[lessMask] = np.ceil((coeffs[lessMask] + Z/2)/Q).astype(int) - 1
        return coeffs.astype(int)

    def dequantize(self, coeffs, Q, Z, C=0.44):
        """Approximately reverse the quantization effect carried out in quantize().
        
        Parameters:
            coeffs ((m,n) ndarray): Array of quantized coefficients.
            Q (float): The step size of the quantization.
            Z (float): The null-zone width (of the center quantization bin).
            C (float): Centering parameter, defaults to .44.
            
        Returns:
            ((m,n) ndarray): Array of dequantized coefficients.
        """
        
        # first, make sure the coefficients are a NumPy array.
        coeffs = np.array(coeffs).astype(float)

        # If Q is 0, return an array of zeros.
        if Q == 0:
            return np.zeros(coeffs.shape).astype(float)

        # Make a positive mask and a negative mask.
        posMask = coeffs > 0
        negMask = coeffs < 0
        zeroMask = coeffs == 0

        # Get the dequantized values and return them in an array
        coeffs[posMask] = (coeffs[posMask].astype(float) - C) * Q + Z/2
        coeffs[negMask] = (coeffs[negMask].astype(float) + C) * Q - Z/2
        coeffs[zeroMask] = 0.0
        return coeffs


#noisy_darkhair = imread('noisy_darkhair.png')

# Perform the WSQ compression and decompression.
wsq = WSQ()

a = wsq.quantize([[1,2,3],[4,5,6],[7,8,9]], 1, 2)
print(a)
print(wsq.dequantize(a, 1, 2,1))

