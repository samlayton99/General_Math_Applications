# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Name>
<Class>
<Date>
"""

import cvxpy as cp 
import numpy as np

# Problem 1
def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # Define the variable and the coefficients of the objective function.
    x = cp.Variable(3, nonneg = True)
    coeffs = np.array([2, 1, 3])
    objective = cp.Minimize(coeffs.T @ x)

    # Define the main constraints for the problem.
    A = np.array([[1, 2, 0], [0, 1, -4]])
    a = np.array([3, 1])
    B = np.array([[2, 10, 3]])
    b = np.array([12])

    # Define the positive constraints for the problem and put them into the constraints list
    P = np.eye(3)
    q = np.zeros(3)
    constraints = [A @ x <= a, B @ x >= b, P @ x >= q]

    # Solve the problem and return the solution and the optimal value.
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return x.value, prob.value


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # Define the variable and objective function.
    x = cp.Variable(A.shape[1])
    objective = cp.Minimize(cp.norm(x, 1))

    # Define the constraints for the problem and solve them
    constraints = [np.array(A) @ x == np.array(b)]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Return the solution and the optimal value.
    return x.value, prob.value


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # Define the variable and the coefficients of the objective function.
    p = cp.Variable(6, nonneg = True)
    coeffs = np.array([4, 7, 6, 8, 8, 9])
    objective = cp.Minimize(coeffs.T @ p)

    # Define the main constraints for the problem.
    A = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]])
    B = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
    a = np.array([7, 2, 4])
    b = np.array([5, 8])

    # Define the positive constraints for the problem and put them into the constraints list
    P = np.eye(6)
    q = np.zeros(6)
    constraints = [A @ p == a, B @ p == b, P @ p >= q]

    # Solve the problem and return the solution and the optimal value.
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return p.value, prob.value


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # Define the variable and the coefficients of the objective function.
    x = cp.Variable(3)
    Q = np.array([[3/2,1,1/2], [1,2,1], [1/2,1,3/2]])
    objective = cp.Minimize(cp.quad_form(x,Q) + np.array([3, 0, 1]) @ x)

    # Solve the problem and return the solution and the optimal value.
    prob = cp.Problem(objective)
    prob.solve()
    return x.value, prob.value


# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # Define the variable and objective function.
    x = cp.Variable(A.shape[1], nonneg = True)
    objective = cp.Minimize(cp.norm(A @ x - b, 2))

    # Define the constraints for the problem and solve them
    constraints = [cp.sum(x) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Return the solution and the optimal value.
    return x.value, prob.value


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    # Get the data and scale it by serving size
    data = np.load('food.npy', allow_pickle=True)
    scale = np.diag(data[:, 1].copy())
    info = (scale @ data[:, 2:].copy()).T
    price = data[:, 0].copy()
    
    # Define the variable and the coefficients of the objective function.
    x = cp.Variable(info.shape[1], nonneg = True)
    objective = cp.Minimize(price.T @ x)

    # Define the main constraints for the problem.
    A = info[:3, :].copy()
    B = info[3:, :].copy()
    a = np.array([2000, 65, 50])
    b = np.array([1000, 25, 46])

    # Define the positive constraints for the problem and put them into the constraints list
    P = np.eye(info.shape[1])
    q = np.zeros(info.shape[1])
    constraints = [A @ x <= a, B @ x >= b, P @ x >= q]

    # Solve the problem and return the solution and the optimal value.
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return x.value, prob.value