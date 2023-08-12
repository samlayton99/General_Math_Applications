# quassian_quadrature.py
"""Volume 2: Gaussian Quadrature.
<Sam Layton>
<001>
<1/26/23>
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import norm
from matplotlib import pyplot as plt

class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        # Store the number of points and weights.
        self.n = n

        # If it is a legendre polynomial, store the name and the inverse weight
        if polytype == "legendre":
            self.polytype = polytype
            self.invw = lambda x: x - x + 1

        # If it is a chebyshev polynomial, store the name and the inverse weight
        elif polytype == "chebyshev":
            self.polytype = polytype
            self.invw = lambda x: np.sqrt(1 - x**2)

        # If it is neither, raise a value error
        else:
            raise ValueError("polytype must be 'legendre' or 'chebyshev'")

        # Store the points and weights based on n
        self.points, self.weights = self.points_weights(n)


    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        # Initialize J, the Jacobi matrix
        J = np.zeros((n,n))

        # If it is a legendre polynomial, calculate the Jacobi matrix
        if self.polytype == "legendre":
            for i in range(1,n):
                J[i,i-1] = np.sqrt(i**2 / (4*i**2 - 1))
                J[i-1,i] = np.sqrt(i**2 / (4*i**2 - 1))
        
        # If it is a chebyshev polynomial, calculate the Jacobi matrix
        else:
            for i in range(1,n):
                J[i,i-1] = np.sqrt(1/4)
                J[i-1,i] = np.sqrt(1/4)

            # set the right first entries for the chebyshev polynomial
            J[0,1], J[1,0] = np.sqrt(1/2), np.sqrt(1/2)

        # Calculate the eigenvalues and eigenvectors of the Jacobi matrix
        eigvals, eigvecs = la.eig(J)
        points = np.real(eigvals)

        # If it is a legendre polynomial, set scale to 2
        if self.polytype == "legendre":
            scale = 2

        # If it is a chebyshev polynomial, set scale to pi
        else:
            scale = np.pi

        # calculate the weights and return the points and weights
        weights = np.real(eigvecs[0])**2*scale
        return points, weights


    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        # Get the points and weights from function, and calculate the approximation
        return (self.invw(self.points) * f(self.points)) @ self.weights


    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        # Define our scale and our transformation
        scale = (b-a)/2
        h = lambda x: f(scale*x + (b+a)/2)

        # Return the approximation
        return self.basic(h) * scale


    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        # Define our scale and our transformation
        scale1 = (b1-a1)/2
        scale2 = (b2-a2)/2
        h = lambda x, y: f(scale1*x + (b1+a1)/2, scale2*y + (b2+a2)/2)

        # Define our weight matrix and initialize our G matrix 
        W = np.outer(self.weights, self.weights)
        G = np.zeros((self.n, self.n))

        # Loop through the matrix and calculate the G matrix
        for i in range(self.n):
            for j in range(self.n):
                G[i,j] = self.invw(self.points[i]) * self.invw(self.points[j]) * h(self.points[i], self.points[j])

        # Calculate the approximation matrix and return the scaled sum
        approxMat = G * W
        return scale1 * scale2 * np.sum(approxMat)


# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    # Calculate the "exact" value of the integral
    F = norm.cdf(2) - norm.cdf(-3)

    # Initialize the arrays for the number of points and the error arrays
    nVals = np.arange(5, 55, 5)
    legendreErrors = []
    chebyshevErrors = []

    # Define the normal pdf
    f = lambda x: (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)

    # Loop through the number of points
    for n in nVals:
        # Initialize the GuassianQuadrature class
        legendre = GaussianQuadrature(n, "legendre")
        chebyshev = GaussianQuadrature(n, "chebyshev")

        # Calculate error of the legendre and chebyshev approximations
        legendreError = np.abs(legendre.integrate(f, -3, 2) - F)
        chebyshevError = np.abs(chebyshev.integrate(f, -3, 2) - F)

        # Append the errors to the error arrays
        legendreErrors.append(legendreError)
        chebyshevErrors.append(chebyshevError)
    
    # Plot the errors and give it a title
    plt.title("Error of Legendre and Chebyshev Approximations")
    plt.plot(nVals, legendreErrors, label="Legendre Error")
    plt.plot(nVals, chebyshevErrors, label="Chebyshev Error")

    # Import quaud, find the error of it, and plot it
    from scipy.integrate import quad
    quadError = np.abs(quad(f, -3, 2)[0] - F)
    plt.axhline(quadError, label="Scipy Error", color="red")

    # Give it a legend, plot it on a log scale, and show it
    plt.legend()
    plt.yscale("log")
    plt.show()