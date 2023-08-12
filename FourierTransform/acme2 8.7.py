import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft

# Exercise 33
# Make a class that accepts a function, a sample number 'n', a period 'T', and a sampling nyquist frequency 'v'
class PeroidicSampling:
    def __init__(self, f = lambda x: np.sin(np.pi / 6 * x) , n = 5, v = 1000, T = 3,):
        # Set the function, sample number, period, and nyquist frequency
        self.function = f
        self.n = n
        self.T = T
    
        # Set the nyquist frequency if n < 2v + 1, otherwise set it to v
        if n < 2 * v + 1:
            self.v = int(np.floor((n - 1) / 2))
        else:
            self.v = v

        # Sample the function at n points from 0 to T and store the values in an array
        self.x = np.linspace(0, (n-1)*T/n, n)
        self.sample = self.function(self.x)

        # Initilize the array of fourier coefficients and calulate the transform
        self.coefficients = np.zeros(2*self.v+1, dtype=complex)
        transform = fft(self.sample)/n
        self.coefficients[self.v] = transform[0]

        # Sort the transform into the array of coefficients
        for i in range(self.v):
            self.coefficients[self.v + i + 1] = transform[i + 1]
            self.coefficients[self.v - i - 1] = transform[self.n -i - 1]

    # Define a method that returns the coefficients
    def coeff(self):
        return self.coefficients

    # Return a function that is the Fourier series approximation of the original function
    def fourierfunction(self):

        # Define the function to return
        def helpfunction(t):

            # Initialize the sum and loop through the coefficients
            sum = 0
            for i in range(2*self.v+1):

                # Add the coefficient times the appropriate exponential and add to the sum
                k = i - self.v
                c = self.coefficients[i]
                sum += c * np.exp((np.pi * 2 / self.T) * 1j * k * t)
            
            # If the input is not an integer, loop through and check for imaginary parts
            if type(t) != int:
                real = True

                # Loop through the array and check for imaginary parts
                for i in range(len(t)):
                    if sum[i].imag < 1e-10:
                        continue

                    # If there is an imaginary part, set real to false and break
                    else:
                        real = False
                        break

                # If there are no imaginary parts, return the real part
                if real:
                    return np.real(sum)

            # if the imaginary sum is less than 10^-10, ignore the imaginary part
            if sum.imag < 1e-10:
                return np.real(sum)
            
            # return the sum
            return sum
        
        # return the function
        return helpfunction

    # Define a method that plots the original function and the Fourier series approximation
    def plot(self, min = 0, max = 2* np.pi, n = 1000):
        # Get the Fourier series approximation
        f = self.fourierfunction()
        x = np.linspace(min, max, n)

        # Plot the original function and the Fourier series approximation
        plt.plot(x, self.function(x), label='Original Function')
        plt.plot(x, f(x), label='Fourier Series Approximation')

        # Plot the points of the original function
        plt.scatter(self.x, self.sample, label='Sampled Points')
        plt.tight_layout()


# Define our function and initialize our class
function1 = lambda x: np.sin(np.pi * 6 * x)
fourier = PeroidicSampling(function1, 19)

# Return the coefficients
print(fourier.coeff())

# Define a callable approximation function g(t)
g = fourier.fourierfunction()
print("f(2) = " + str(function1(2)))
print("g(2) = " + str(g(2)))



# Exercise 34
# Define our function and initialize our n values in an array
f = lambda x: 1 - 3 * np.sin(np.pi * 12 * x + 7) + 5 * np.sin(np.pi * 2 * x - 1) + 5 * np.sin(np.pi * 4 * x - 3)
nval = [3,7,11,13]

# Loop through the different n values and plot the Fourier series approximation
for n in nval:
    # Initialize our class and get the right subplot
    plt.subplot(2,2,nval.index(n)+1)
    fourier = PeroidicSampling(f, n, 1000, 1)
    
    # Plot the Fourier series approximation and the scatter
    fourier.plot(0, 1, 1000)

plt.show()
