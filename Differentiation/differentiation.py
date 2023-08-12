# differentiation.py
"""Volume 1: Differentiation.
<Sam Layton>
<Section 001>
<1/24/23>
"""

import numpy as np
import sympy as sy
from scipy import linalg as la
from matplotlib import pyplot as plt
from autograd import numpy as anp
from autograd import elementwise_grad
from autograd import grad
import time

# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    # Create the symbolic variable x, then create the symbolic expression and take the derivative.
    x = sy.Symbol('x')
    expr = (sy.sin(x) + 1)**sy.sin(sy.cos(x))
    expr = sy.diff(expr)

    # Convert the SymPy expression to a Python function and return it.
    f = sy.lambdify(x, expr, 'numpy')
    return f

def plotderiv():
    # create the symbolic variable x, then create the symbolic expression and lambdify it
    x = sy.Symbol('x')
    expr = (sy.sin(x) + 1)**sy.sin(sy.cos(x))
    f = sy.lambdify(x, expr, 'numpy')

    # take the derivative of the expression and lambdify it, get the domain
    expr = sy.diff(expr)
    df = sy.lambdify(x, expr, 'numpy')
    domain = np.linspace(-np.pi, np.pi, 1000)

    # plot the function and its derivative. Give it a title
    plt.plot(domain, f(domain), label='f(x)')
    plt.plot(domain, df(domain), label="f'(x)")
    plt.title("f(x) and f'(x)")

    # Make it neat, give it a legend and show it
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()


# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    # make the x vector a numpy array and add h to it
    x = np.array(x)
    nudge = x + h
    # calculate the derivative and return it
    deriv = (f(nudge) - f(x)) / h
    return deriv

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    # make the x vector a numpy array and add h to it
    x = np.array(x)
    nudge1 = x + h
    nudge2 = x + 2*h

    # calculate the derivative and return it
    deriv = (-3 * f(x) + 4*f(nudge1) - f(nudge2)) / (2*h)
    return deriv

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    # make the x vector a numpy array and subtract h from it
    x = np.array(x)
    nudge = x - h

    # calculate the derivative and return it
    deriv = (f(x) - f(nudge)) / h
    return deriv

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    # make the x vector a numpy array and subtract h from it
    x = np.array(x)
    nudge1 = x - h
    nudge2 = x - 2*h

    # calculate the derivative and return it
    deriv = (3 * f(x) - 4*f(nudge1) + f(nudge2)) / (2*h)
    return deriv

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    # make the x vector a numpy array and add h to it
    x = np.array(x)
    nudge1 = x + h
    nudge2 = x - h

    # calculate the derivative and return it
    deriv = (f(nudge1) - f(nudge2)) / (2*h)
    return deriv

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    # make the x vector a numpy array and add h to it
    x = np.array(x)
    nudge1 = x + h
    nudge2 = x - h

    # Make additional nudges
    nudge3 = x + 2*h
    nudge4 = x - 2*h

    # calculate the derivative and return it
    deriv = (f(nudge4) - 8*f(nudge2) + 8*f(nudge1) - f(nudge3)) / (12*h)
    return deriv

def plotdiff():
    # create the symbolic variable x, then create the symbolic expression and lambdify it
    x = sy.Symbol('x')
    expr = (sy.sin(x) + 1)**sy.sin(sy.cos(x))
    f = sy.lambdify(x, expr, 'numpy')

    # take the derivative of the expression and lambdify it
    expr = sy.diff(expr)
    df = sy.lambdify(x, expr, 'numpy')

    # make the domain of x to plot
    domain = np.linspace(-np.pi, np.pi, 1000)

    # Plot f, its sympy derivative, and all of its other approximations
    plt.title("f(x) and its derivative types")
    plt.plot(domain, f(domain), label="f(x)")
    plt.plot(domain, df(domain), label="Actual derivative")

    # Plot the approximation types for all of the functions
    plt.plot(domain, fdq1(f, domain), label="fdq1")
    plt.plot(domain, fdq2(f, domain), label="fdq2")
    plt.plot(domain, bdq1(f, domain), label="bdq1")

    # Continue and plot the rest of the approximation types
    plt.plot(domain, bdq2(f, domain), label="bdq2")
    plt.plot(domain, cdq2(f, domain), label="cdq2")
    plt.plot(domain, cdq4(f, domain), label="cdq4")

    # Set the limits of the graph and show it. You can see they are all really similar
    plt.ylim(-1.5, 3)
    plt.legend()
    plt.show()


# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    # from problem 1, we can get the exact value of f'(x0), and get the h values
    df = prob1()
    exact = df(x0)
    h = np.logspace(-8, 0, 9)
    
    # get the function f
    f = lambda x: (np.sin(x) + 1)**(np.sin(np.cos(x)))

    # Get the approximations for each of the functions in lambda form
    f1 = lambda x, h: fdq1(f, x, h)
    f2 = lambda x, h: fdq2(f, x, h)
    b1 = lambda x, h: bdq1(f, x, h)
    b2 = lambda x, h: bdq2(f, x, h)
    c2 = lambda x, h: cdq2(f, x, h)
    c4 = lambda x, h: cdq4(f, x, h)

    # put these functions into lists with their names
    approxFuncts = [f1, f2, b1, b2, c2, c4]
    functNames = ["fdq1", "fdq2", "bdq1", "bdq2", "cdq2", "cdq4"]

    # Loop through the h values and calculate the approximations, reset the error
    for j in range(len(approxFuncts)):
        error = []
        for i in range(len(h)):

            # append the error to the list
            error.append(abs(approxFuncts[j](x0, h[i]) - exact))

        # plot the error
        plt.loglog(h, error, label=functNames[j],  marker='o')

    # Set the title, labels, and show the plot
    plt.title("Absolute Error of Approximations")
    plt.xlabel("h")
    plt.ylabel("Absolute Error")

    # Make it neat, add a legend, and show the plot
    plt.tight_layout()
    plt.legend()
    plt.show()


# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """

    # Load the data, convert alpha and beta to radians, and get alpha and beta
    data = np.load("plane.npy")
    alpha = np.array(np.deg2rad(data[:, 1]))
    beta = np.array(np.deg2rad(data[:, 2]))

    # Get the x and y coordinates
    x = 500 * (np.tan(beta) / (np.tan(beta) - np.tan(alpha)))
    y = 500 * (np.tan(beta) * np.tan(alpha)) / (np.tan(beta) - np.tan(alpha))
    
    # Define a function that takes the element wise difference of the array
    def deriv(x):
        # Make it an np array and get the centered difference
        x = np.array(x)
        centered = (x[2:] - x[:-2]) / 2

        # Get the forward and backward difference and concatenate them
        front = (x[1] - x[0])
        back = (x[-1] - x[-2])
        return np.concatenate(([front], centered, [back]))
    
    # Get the x and y derivatives
    x_deriv = deriv(x)
    y_deriv = deriv(y)

    print(x_deriv)
    # Get the speed and return it
    speed = np.sqrt(x_deriv**2 + y_deriv**2)
    return speed


# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """

    # Get the length of x and initialize the identity matrix
    n = len(x)
    m = len(f(x))
    I = np.eye(n)

    # Initialize the Jacobian matrix and make x an np array
    J = np.zeros((m, n))
    x = np.array(x)

    # Loop n times and get the partial nudge up and nudge down
    for i in range(n):
        nudgeUp = x + h * I[i]
        nudgeDown = x - h * I[i]

        # Get the partial derivative and store it in the Jacobian matrix
        J[:, i] = (f(nudgeUp) - f(nudgeDown)) / (2 * h)
    return J


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (jax.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    # Turn it into an anp array
    x = anp.array(x)
    
    # Base cases
    if n == 0:
        return anp.ones_like(x)
    elif n == 1:
        return x

    # Recursively compute the nth Chebyshev polynomial
    else:
        return 2 * x * cheb_poly(x, n - 1) - cheb_poly(x, n - 2)


def prob6():
    """Use JAX and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    # Get the domain and the derivative
    domain = anp.linspace(-1, 1, 500)
    deriv = elementwise_grad(cheb_poly)

    # Plot the Chebyshev polynomials and their derivatives in respective subplots
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.title("n = " + str(i))
        plt.plot(domain, deriv(domain, i))

    # give it a title and show the plot
    plt.tight_layout()
    plt.show()


# Problem 7
def prob7(N=200):
    """
    Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the "exact" value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            JAX (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and JAX.
    For SymPy, assume an absolute error of 1e-18.
    """
    # Create our function using anp and initialize the times and errors arrays
    f = lambda x: (anp.sin(x) + 1) ** anp.sin(anp.cos(x))
    times = np.zeros((N, 3))
    errors = np.zeros((N, 2))

    # Repeat the following experiment N times
    for i in range(N):
        
        # Get the random number x0 
        x0 = anp.random.random() * 40 - 20

        # Get the exact value of f'(x0) and the time it takes
        start = time.perf_counter()
        exact = prob1()(x0)
        end = time.perf_counter()
        exact_time = end - start

        # Get the approximation using cdq4 and the time it takes
        start = time.perf_counter()
        approx1 = cdq4(f, x0)
        end = time.perf_counter()
        cdq4_time = end - start

        # Get the approximation using JAX and the time it takes
        start = time.perf_counter()
        approx2 = grad(f)(x0)
        end = time.perf_counter()
        jax_time = end - start

        # Get the error for each approximation
        cdq4_error = abs(exact - approx1)
        jax_error = abs(exact - approx2)

        # Store the times and errors in their respective arrays
        times[i] = [cdq4_time, jax_time, exact_time]
        errors[i] = [cdq4_error, jax_error]

    # Plot the times and errors
    plt.plot(times[:, 0], errors[:, 0], "o", label="Difference Quotient", alpha = 0.3)
    plt.plot(times[:, 1], errors[:, 1], "o", label="Autograd", alpha = 0.3)
    plt.plot(times[:, 2], np.ones(N) * 1e-18, "o", label="SymPy", alpha = 0.3)
    
    # Give it a title and label the axes
    plt.title("Computation Time vs. Absolute Error")
    plt.xlabel("Computation Time")
    plt.ylabel("Absolute Error")

    # Set the x and y scales to log, give it a legend, and show the plot
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(["Difference Quotient", "Autograd", "SymPy"])
    plt.show()