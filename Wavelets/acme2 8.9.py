import numpy as np
from matplotlib import pyplot as plt

#######################################################################################
# Make a function that takes in a function f and integer n, return an array that samples f at points k/2^n for k=0,1,...,2^n - 1
def sample(f, n):
    # return the sampled function
    return np.array([f(k/2**n) for k in range(2**n)])

# Write a method that accepts an array and a value x in [0,1) and returns the sum of the nk wavelets at x
def wavelet_sum(a, x):
    # return the index of the array that is closest to x
    return a[int(x*len(a))]

# make a function that plots the wavelet sum of f for n = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
def plot_wavelet_sum(f, jval):
    # make a linspace and loop through all values of n
    x = np.linspace(0, 1, 1000, endpoint = False)
    i = 1
    for n in jval:
        # Change the subplot and give it a title
        plt.subplot(1, len(jval), i)
        plt.title('n = ' + str(n))
        i += 1

        # Plot the wavelet sum by sampling f and looping through all values of x
        y = [wavelet_sum(sample(f, n), x_i) for x_i in x]
        plt.plot(x, y, label = 'n = ' + str(n))

    # Make the plot look nice
    plt.tight_layout()
    plt.show()
#######################################################################################

#8.45 part i) and ii)
# Split the array into its approximation and detail components
def splitdetail(a,j):
    # get length n and return components
    n = 2**j
    return a[:n], a[n:]

# Define a function f
def f(x):
    # Define it as in the exercise
    return 100 * x **2 * (1-x) * np.abs(np.sin(10*x / 3))

# make a masking function
def phimask(a):
    # return the mask
    mask1 = 0<= np.array(a)
    mask2 = np.array(a) < 1
    return mask1 * mask2


# Define the FWT function that takes in an array and an integer j
def ifwt(a, j = 0):
    # make a np array of a and get the length of a
    a = np.array(a)
    m = int(np.log2(len(a)))
    L = []

    # Loop through all values of m until it is equal to j
    while m > j:
        # Add the detail component to the list L, adjust a, and decrease m
        L = np.concatenate((np.array(L), (.5 * (a[::2] - a[1::2]))[::-1]))
        a = .5 * (a[::2] + a[1::2])
        m -= 1

    # Return the b vector
    return np.insert(L[::-1], 0, a)

# Define a function that plots the approximation and detail components of the FWT
def plot_splitdetail(f,n, jval):
    # Get your n and a and b vectors
    a = sample(f, n)
    b = ifwt(a)

    # Split the detail and approximation components and give x a linspace
    approx, detail = splitdetail(b, jval)
    x = np.linspace(0, 1, 1000, endpoint = False)

    # make the haar son and daughter functions
    phi = lambda x,j,k: phimask((2**j)*x - k) * 1
    psi = lambda x,j,k: phi(x,j+1,2*k) - phi(x,j+1,2*k+1)

    # Initialize the sum and the counter
    counter = 1
    approx_sum = approx[0] * phi(x, 0, 0)

    # Make the approximation component
    for j in range(jval):
        for k in range(2**j):
            # if the approximation component is not 0, add it to the sum
            if approx[counter] != 0:
                approx_sum += approx[counter] * psi(x, j, k)
            counter += 1

    # Initialize the detail sum and reset the counter
    counter = 0
    detail_sum = np.zeros(len(x))

    # Make the detail component if it is not 0
    if len(detail) != 0:
        for j in range(jval, n):
            for k in range(2**j):
                # if the detail coefficients are not 0, add it to the sum
                if detail[counter] != 0:
                    detail_sum += detail[counter] * psi(x, j, k)
                counter += 1
            

    # Set the Figure size and plot the original
    plt.figure(figsize = (9,3))
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.plot(x, f(x),'k')

    # Plot the approximation components
    plt.subplot(1, 3, 2)
    plt.title('Approximation')
    plt.plot(x, approx_sum, 'r')

    # Plot the detail components
    plt.subplot(1, 3, 3)
    plt.title('Detail')
    plt.plot(x, detail_sum)
    plt.ylim(-1, 12)

    # Make the plot look nice
    plt.ylim(-1, 12)
    plt.tight_layout()
    plt.show()
    
plot_splitdetail(f, 6, 5)


# 8.46 part i)
def separatefunctions(f, l, jval):
    # Get your n and a and b vectors
    a = sample(f, l)
    b = ifwt(a)

    # Split the detail and approximation components and give x a linspace
    approx, detail = splitdetail(b, jval)

    # make the haar son and daughter functions
    phi = lambda x,j,k: phimask((2**j)*np.array(x) - k) * 1
    psi = lambda x,j,k: phi(x,j+1,2*k) - phi(x,j+1,2*k+1)

    # Make the approximation function
    def approxfunc(x):
        # Initialize the sum and the counter
        counter = 1
        approx_sum = approx[0] * phi(x, 0, 0)

        # Make the approximation function
        for j in range(jval):
            for k in range(2**j):
                # if the approximation component is not 0, add it to the sum
                if approx[counter] != 0:
                    approx_sum += approx[counter] * psi(x, j, k)
                counter += 1

        # Return the approximation sum
        return approx_sum


    # Make the detail function
    def detailfunc(x):
        # if x is an integer, make x start 
        if type(x) == float or type(x) == int:
            detail_sum = 0
        else:
            detail_sum = np.zeros(len(x))
        
        # Initialize the counter
        counter = 0

        # Make the detail component if it is not 0
        if len(detail) != 0:
            for j in range(jval, l):
                for k in range(2**j):
                    # if the detail coefficients are not 0, add it to the sum
                    if detail[counter] != 0:
                        detail_sum += detail[counter] * psi(x, j, k)
                    counter += 1
        
        # Return the detail sum
        return detail_sum
    
    # Return the approximation and detail functions
    return approxfunc, detailfunc


# 8.46 part ii)
# Define a function g
def g(x):
    # Define it as in the exercise
    return np.sin(2*np.pi*x - 5) / np.sqrt(np.abs(x-np.pi/20))

# plot the functions
def plotmany(f):
    # Make the x vector and set the figure size
    x = np.linspace(0, 1, 1000, endpoint = False)
    plt.figure(figsize = (10,7))

    # Loop through the j values and get the approximation and detail functions
    for j in range(10):
        approx ,detail = separatefunctions(f, 10, j)

        # Plot the approximation function and set the y limits
        plt.subplot(4, 5, j + 1 + 5*(j//5))
        plt.title('Approx: j = ' + str(j))
        plt.plot(x, approx(x), 'r')
        plt.ylim(-3, 7)
    
        # Plot the detail function and set the y limits
        plt.subplot(4, 5,  j + 6 + 5*(j//5))
        plt.title('Detail: j = ' + str(j))
        plt.plot(x, detail(x))
        plt.ylim(-5, 5)
    
    # Make the plot look nice
    plt.tight_layout()
    plt.show()

plotmany(g)


# 8.46 part iii)
"""They are different because the WFT is making a change of basis from the original 
basis to the haar basis. We then are discarding the detail coefficients only until 
it is in the same subspace. Thus the approximation function is technically different,
because when the detail is included, it represents the original function in a higher
dimensional space. The sampling in the lower dimensional space is thus not the same 
as our WFT sampling, yet our WFT sampling has a more accurate representation of the
original function, because noise is not included in the discarded detail 
coefficients."""


# 8.47 part i)
# Create a function to shift and scale any function on the interval [a,b] to the interval [0, 1]
def shiftscaleto1(f, a, b):
    # Define the function
    def g(x):
        return f((b-a)*x + a)
    
    # Return the function
    return g

# Create the reverse of the above function
def shiftscaleto2(f, a, b):
    # Define the function
    def g(x):
        return f((x-a)/(b-a))
    
    # Return the function
    return g

# Make a shifting function that does the WFT on any interval [a,b]
def extendedwft(f, a = -1, b = 1, l = 6,jval = 5,n = 1000):
    # Make our x vector on the interval [a,b] and shift and scale the function
    xab = np.linspace(a,b,n, endpoint = False)
    fshift = shiftscaleto1(f, a, b)
    
    # Get the approximation and detail functions and shift back to the interval [a,b]
    approx, detail = separatefunctions(fshift, l, jval)
    approx = shiftscaleto2(approx, a, b)
    detail = shiftscaleto2(detail, a, b)

    # Return the approximation and detail functions
    return approx, detail


# 8.47 part ii)
# Plot the functions in the same style as in 8.46 part ii) on the interval [-1,1]
def plotmany2(f,a = -1, b = 1):
    # Make the x vector and set the figure size
    x = np.linspace(a, b, 1000, endpoint = False)
    plt.figure(figsize = (10,7))

    # Loop through the different j values and get the approx and detail functions
    for j in range(10):
        approx ,detail = extendedwft(f, a, b, 10, j)

        # Plot the approximation function and set the y limits
        plt.subplot(4, 5, j + 1 + 5*(j//5))
        plt.title('Approx: j = ' + str(j))
        plt.plot(x, approx(x), 'r')
        plt.ylim(-3, 7)
    
        # Plot the detail function and set the y limits
        plt.subplot(4, 5,  j + 6 + 5*(j//5))
        plt.title('Detail: j = ' + str(j))
        plt.plot(x, detail(x))
        plt.ylim(-5, 5)
    
    # Set the tight layout and show the plot
    plt.tight_layout()
    plt.show()

plotmany2(g)