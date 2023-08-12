# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
<Sam>
<Section 002>
<1/5/23>
"""
import numpy as np
from scipy.interpolate import BarycentricInterpolator
from matplotlib import pyplot as plt
from scipy import linalg as la


# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    # Copy the x values of the interpolating points and loop through them.
    denom = np.zeros(len(xint))
    for i in range(len(xint)):

        # Calculate the denominator of the Lagrange basis polynomial.
        addValue = 1
        for j in range(len(xint)):

            # If i and j are not the same index, multiply the difference by the previous value. Add to the denominator array.
            if i != j:
                addValue = addValue * (xint[i] - xint[j])
        denom[i] = addValue
    
    # Make the return array and loop through the points to be evaluated. Identify the test value.
    returnMatrix = np.zeros((len(xint),len(points)))
    for k in range(len(points)):
        testVal = points[k]

        # Loop through the x values of the interpolating points and calculate the numerator.
        for i in range(len(xint)):
            returnVal = 1
            for j in range(len(xint)):

                # If i and j are not the same index, multiply the difference by the previous value. Add to the return array. And divide by the denominator.
                if i != j:
                    returnVal = returnVal * (testVal - xint[j])
            returnMatrix[i][k] = returnVal / denom[i]
    
    # Multiply the return array by the y values of the interpolating points and return the array.
    return yint @ returnMatrix


# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        # Store the x and y values of the interpolating points and the number of points and C.
        self.x = np.array(xint)
        self.y = np.array(yint)
        self.n = len(xint)
        self.C = (np.max(self.x) - np.min(self.x)) / 4

        # Initialize the weights attribute, shuffle the x values, and loop through the x values.
        self.w = np.ones(self.n)
        shuffle = np.random.permutation(self.n - 1)
        for i in range(self.n):

            # Calculate scaled weights and store them in the weights attribute.
            temp = (self.x[i] - self.x[np.arange(self.n) != i]) / self.C
            temp = temp[shuffle]
            self.w[i] /= np.prod(temp)

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        # make the points array a numpy array and loop through it. Identify the test value.
        points = np.array(points, dtype=float)
        for i in range(len(points)):
            testVal = points[i]
            
            # if the test value is in the x values of the interpolating points, store the corresponding y value in the return array.
            if points[i] in self.x:
                points[i] = self.y[self.x == points[i]]
                continue

            # Calculate the numerator and denominator of the Barycentric Lagrange polynomial.
            numerator = np.sum(self.w * self.y / (testVal - self.x))
            denominator = np.sum(self.w / (testVal - self.x))

            # Calculate the value of the Barycentric Lagrange polynomial and store it in the return array.
            points[i] = numerator / denominator
        return points

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        # Store the x and y values of the interpolating points and the number of points and C.
        self.x = np.append(self.x, xint)
        self.y = np.append(self.y, yint)
        self.n += len(xint)
        self.C = (np.max(self.x) - np.min(self.x)) / 4

        # Initialize the weights attribute, shuffle the x values, and loop through the x values.
        self.w = np.ones(self.n)
        shuffle = np.random.permutation(self.n - 1)
        for i in range(self.n):

            # Calculate scaled weights and store them in the weights attribute.
            temp = (self.x[i] - self.x[np.arange(self.n) != i]) / self.C
            temp = temp[shuffle]
            self.w[i] /= np.prod(temp)

# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    # initialize the lambda function, x values, and the n values.
    f = lambda x: 1 / (1 + 25 * x**2)
    nVals = np.array([2**i for i in range(2, 9)])
    x = np.linspace(-1, 1, 400)

    # initialize the normal and chebyshev error arrays.
    normalError = []
    chebyshevError = []

    # loop through the n values and make the barycentric interpolator for each n value.
    for n in nVals:
        xint = np.linspace(-1, 1, n)
        yint = f(xint)
        bary = BarycentricInterpolator(xint, yint)

        # calculate the error for the equally spaced points and append it to the normal error array.
        error = la.norm(f(x) - bary(x), ord = np.inf)
        normalError.append(error)
        
        #Interpolate Runge’s function with n + 1 Chebyshev extremal points, also via SciPy, and compute the absolute error.
        xint = (np.cos(np.arange(n + 1) * np.pi / n) * (np.max(xint) - np.min(xint)) + np.max(xint) + np.min(xint)) / 2
        yint = f(xint)
        bary = BarycentricInterpolator(xint, yint)
        
        # calculate the error for the chebyshev points and append it to the chebyshev error array.
        errorCheb = la.norm(f(x) - bary(x), ord = np.inf)
        chebyshevError.append(errorCheb)
    
    # plot the error for the normal and chebyshev points in a log-log plot. Give it a title and label the axes.
    plt.title("Error of Interpolation")
    plt.loglog(nVals, normalError, label = "Normal")
    plt.loglog(nVals, chebyshevError, label = "Chebyshev")

    # label the axes.
    plt.xlabel("n")
    plt.ylabel("Error")

    # make it neat, give a legend, and show the plot.
    plt.tight_layout()
    plt.legend()
    plt.show()


# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    # Take Chebychev points and calculate the Chebyshev coefficients.
    samples = f(np.cos(np.arange(2 * n)* np.pi / n))
    coeffs = np.real(np.fft.fft(samples))[:n+1] / n

    # Scale the coefficients by 1/2 for the first and last coefficients. Return the coefficients.
    coeffs[0] /= 2
    coeffs[n] /= 2
    return coeffs


# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    # Load the air quality data and store the x and y values.
    data = np.load("airdata.npy")
    x = np.arange(len(data))
    y = data

    # Make the function to find the near chebyshev points.
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    a, b = 0, 366 - 1/24
    points = fx(a, b, n)

    # Make a domain and interpolate the data
    domain = np.linspace(0, b, 8784)
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)
    poly = Barycentric(domain[temp2], data[temp2])
    
    # set the figure size, plot the data, and give it a title
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, y, label = "Data")

    # label the axis and give it a title
    plt.title("Air Quality Data")
    plt.xlabel("Time")
    plt.ylabel("Air Quality")

    # plot the interpolating polynomial and give it a title
    plt.subplot(1, 2, 2)
    plt.title("Interpolating Polynomial")
    plt.plot(x, poly(domain), label = "Interpolating Polynomial", color = "orange")

    # label the axis, make it neat, and show it
    plt.xlabel("Time")
    plt.ylabel("Air Quality")
    plt.tight_layout()
    plt.show()