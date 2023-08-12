# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
<Name>
<Class>
<Date>
"""
import numpy as np
from scipy.stats.mvn import mvnun
from matplotlib import pyplot as plt


# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    # Start a count and loop through N random points.
    count = 0
    for i in range(N):

        # Generate a random point in the n-dimensional unit cube.
        p = np.linalg.norm(np.random.uniform(-1,1,n))

        # If the point is inside the unit ball, add 1 to the count.
        if p <= 1:
            count += 1

    # return the ratio of points inside 
    return 2**n * count/N


# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    return (b - a) * np.mean(f(np.random.uniform(b, a, N)))


# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    # Get the volume and shift the random points
    vol = np.prod(np.abs(np.array(maxs) - np.array(mins)))
    draws = np.random.uniform(0, 1, (N,len(mins)))
    shifted = (draws * (np.array(maxs) - np.array(mins)) + np.array(mins))

    # Return the mean of the function at the shifted points times the volume
    return vol * np.mean([f(shifted[i]) for i in range(len(shifted))])


# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    # Define our n and the domain
    n = 4
    mins = [-3/2, 0, 0, 0]
    maxs = [3/4, 1, 1/2, 1]

    # Define our function and get exact value using mvnun()
    f = lambda x: np.exp(-x.T@x/2) / (2*np.pi)**(n/2)
    exact = mvnun(mins, maxs, np.zeros(n), np.eye(n))[0]
    
    # get our domain for graphing and initialize our error array
    N = np.logspace(1, 5, 20, dtype=int)
    error = np.zeros(len(N))

    # loop through N and get the corresponding error
    for k in range(len(N)):
        t = mc_integrate(f, mins, maxs, N[k])
        error[k] = np.abs((mc_integrate(f, mins, maxs, N[k]) - exact)) / np.abs(exact)

    # plot the error and the line 1/sqrt(N) and give it a legend
    plt.loglog(N, error, label='Error')
    plt.loglog(N, 1/np.sqrt(N), label='1/sqrt(N)')
    plt.legend()

    # Give it a title, labels, and show it
    plt.title('Error vs. Sample Size')
    plt.xlabel('Sample Size')
    plt.ylabel('Relative Error')
    plt.show()