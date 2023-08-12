import numpy as np
from autograd import grad
from matplotlib import pyplot as plt
import sympy as sy
import time
from autograd import numpy as anp


# --------------------------------- Problem 16 ------------------------------------
# Problem 16 i)
def diff(f, x0, n, h = 2 * np.sqrt(np.finfo(np.float64).eps)):
    # Initialize the Identity matrix and make x an np array
    I = np.eye(n)
    x0 = np.array(x0)
    
    # Get the length of output by checking if it's a scalar, make the gradient vector
    output = f(x0)
    if np.isscalar(output):
        m = 1
        J = np.zeros(n)

        # Loop n times and get the partial nudge up and calculate the partial derivative
        for i in range(n):
            nudgeUp = x0 + h * I[i]
            J[i] = (f(nudgeUp) - f(x0)) / h

    # If it's not a scalar, get the length of the output and make the Jacobian matrix
    else:
        m = len(output)
        J = np.zeros((m, n))

        # Loop n times and get the partial nudge up and calculate the partial derivative
        for i in range(n):
            nudgeUp = x0 + h * I[i]
            J[:,i] = (f(nudgeUp) - f(x0)) / h
    
    # Return the Jacobian matrix
    return J

# Problem 16 ii)
def derivative(x0,y0):
    # Define the sympy variables and function
    x = sy.symbols('x')
    y = sy.symbols('y')
    f1 = x / (x + 2 + y**2)
    f2 = y / (y**2 + x**2)

    # Take its total derivative and lambdify it 
    j11 = sy.lambdify((x,y),f1.diff(x), 'numpy')
    j12 = sy.lambdify((x,y),f1.diff(y), 'numpy')
    j21 = sy.lambdify((x,y),f2.diff(x), 'numpy')
    j22 = sy.lambdify((x,y),f2.diff(y), 'numpy')

    # Return the Jacobian matrix
    return np.array([[j11(x0,y0), j12(x0,y0)], [j21(x0,y0), j22(x0,y0)]])


def example():
    # Define our f function
    g = lambda x: np.array([x[0] / (x[0] + 2 + x[1]**2), x[1] / (x[1]**2 + x[0]**2)])

    # Print the Jacobian matrix using symbolic differentiation
    print("Symbolic differentiation:\n", derivative(2,3) )

    # Print the Jacobian matrix using numerical differentiation
    print("Numerical differentiation:\n", diff(g, [2,3], 2))

# problem 16 iii) and iv)
def example2():
    # Compute the actual Jacobian matrix, time it, and store the time
    start = time.time()
    J = derivative(2,3)
    end = time.time()
    ActualTime = end - start

    # Define our f function and initialize the time list
    f = lambda x: np.array([x[0] / (x[0] + 2 + x[1]**2), x[1] / (x[1]**2 + x[0]**2)])
    NumTime = []
    error = []

    # Loop through the range of h values
    for i in range(2,54):
        h = 2 ** (-i)

        # Compute the Jacobian matrix using numerical differentiation and time it
        start = time.time() 
        J2 = diff(f, [2,3], 2, h)
        end = time.time()
        NumTime.append(end - start)

        # Find the error between the two Jacobian matrices
        error.append(np.linalg.norm(J - J2, np.inf))

    # Get the optimal h value and average time and print it
    k = np.argmin(error) + 2
    print("The optimal h value is 2^(-", k, ") = ", 2.0 ** (-k))
    print("The average Numerical time is ", np.mean(NumTime), " seconds")
    print("The Actual time is ", ActualTime, " seconds")



# --------------------------------- Problem 17 ------------------------------------
# Problem 17 i)
def diff_centered(f, x0, n, h = 1.4 * np.power(np.finfo(np.float64).eps, 1/3)):
    I = np.eye(n)
    x = np.reshape(x0, (n,1))
    return (f(x + h * I) - f(x - h * I)) / (2 * h)


# Problem 17 ii) and iii)
def example_centered():
    # Compute the actual Jacobian matrix, time it, and store the time
    start = time.time()
    J = derivative(2,3)
    end = time.time()
    ActualTime = end - start

    # Define our f function and initialize the time list
    f = lambda x: np.array([x[0] / (x[0] + 2 + x[1]**2), x[1] / (x[1]**2 + x[0]**2)])
    NumTime = []
    error = []

    # Loop through the range of h values
    for i in range(2,54):
        h = 2 ** (-i)

        # Compute the Jacobian matrix using numerical differentiation and time it
        start = time.time() 
        J2 = diff_centered(f, [2,3], 2, h)
        end = time.time()
        NumTime.append(end - start)

        # Find the error between the two Jacobian matrices
        error.append(np.linalg.norm(J - J2, np.inf))

    # Get the optimal h value and average time and print it
    k = np.argmin(error) + 2
    print("The optimal h value is 2^(-", k, ") = ", 2.0 ** (-k))
    print("The average Numerical time is ", np.mean(NumTime), " seconds")
    print("The Actual time is ", ActualTime, " seconds")


# --------------------------------- Problem 18 ------------------------------------
# Problem 18
# Define all methods below
def symbolically(f,x,x0):
    # Return the symbolic derivative
    return f.diff(x).subs(x,x0)

def forward(f,x,h):
    # Return the forward difference
    return (f(x + h) - f(x))/h

def centered(f,x,h):
    # Return the centered difference
    return (f(x + h) - f(x - h))/(2*h)

def complex(f,x,h):
    # Return the complex difference
    return np.imag(f(x + 1j*h)) / h

def autogradmethod(f,x):
    # Return the autograd difference
    return grad(f)(x)

def compare():
    # Get your h values and function
    h = [2**(-i) for i in range(1,54)]
    f = lambda x: (np.sin(x)**3 + np.cos(x)) * np.exp(-x)
    method = [forward, centered, complex]

    # Get the symbolic derivative and evaluate at x = 1.5
    x = sy.Symbol('x')
    expr = (sy.sin(x)**3 + sy.cos(x)) * sy.exp(-x)

    # Calculate the symbolic derivative and time it 
    start = time.time()
    symboldiff = symbolically(expr,x,1.5)
    end = time.time()
    print("Symbolic time: ", end - start)

    # Initialize the error matrix and method names
    errorMat = np.zeros((3,53))
    methodName = ["Forward", "Centered", "Complex"]

    # Loop through the other methods for different h values
    for i in range(len(method)):
        avgTime = 0
        for j in range(len(h)):
            # Calculate the derivative and get its time
            start = time.time()
            diff = method[i](f,1.5,h[j])
            end = time.time()
            avgTime += end - start

            # Calculate the error and print it
            error = abs(symboldiff - diff)
            errorMat[i,j] = error
            
        avgTime /= len(h)
        # Print the average time
        print("Average time for ", methodName[i], " is ", avgTime, " seconds")
        plt.plot(h,errorMat[i,:],label = methodName[i])

    # calculate the derivative using grad and time it
    af = lambda x: (anp.sin(x)**3 + anp.cos(x)) * anp.exp(-x)
    start = time.time()
    autodiff = autogradmethod(af,1.5)
    end = time.time()

    # Print the autograd time and the symbolic and autograd derivatives
    print("Autograd time: ", end - start)
    print("Autograd derivative: ", autodiff)
    print("Symbolic derivative: ", symboldiff)

    # Plot the error for the symbolic and autograd methods
    plt.title("Error over h for different methods")
    plt.plot(h,np.ones(len(h)),label = "Symbolic")
    plt.plot(h,np.ones(len(h)),label = "Autograd")

    # Make the graph be a log-log plot and label the axes
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('h')
    plt.ylabel('Error')

    # Show the legend and plot
    plt.legend()
    plt.show()

# We see that Centered and Complex decrease in error the fastest. It has the least error at roughly h = 10^-6 
# Centered has an error of roughly 10^-12.0 at its lowest point. Machine error starts to take over after that
# and the error increases at a steady rate, the same as Forward
# The Forward method is slowest and has the greatest error at roughly h = 10^-7.5, with an error of roughly 10^-9.0
# The complex method is the fastest and has the least error. It bottoms out at roughly h = 10^-8, below machine error


# --------------------------------- TEST CASES ------------------------------------


#example()
#example2()
#example_centered()
#compare()