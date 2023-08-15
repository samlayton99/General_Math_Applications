# Convex Optimization Project Description

In this project, I leverage CVXPY, a comprehensive Python package tailored for convex optimization, to address a range of optimization challenges, including linear and quadratic programming. As convex optimization garners significance in domains such as engineering, economics, and machine learning, this endeavor highlights the versatility and depth of the CVXPY library, particularly in navigating complex problems such as optimizing healthy eating on a budget.

## Mathematical Background and Overview of Functions

Convex optimization focuses on minimizing a convex function over a convex set, proving its worth across several fields. The CVXPY library, a Python-embedded modeling language, allows users to express such problems in a syntax mirroring mathematical notation, facilitating the computation of optimal solutions.

1. **Convex Optimization Problem**
    - `prob1()`: 
        - Problem: Minimize a given convex function.
        - Constraints: \(x + 2y \leq 3\)
        - Outputs: The optimizer `x` and the optimal value.

2. **L1 Minimization**
    - `l1Min(A, b)`:
        - Problem: Minimize the L1 norm.
        - Parameters: Matrices `A` and vector `b`.
        - Outputs: The optimizer `x` and the optimal value.

3. **Transportation Problem**
    - `prob3()`: 
        - Problem: Minimize transportation costs.
        - Modification: Convert the last equality constraint into inequality constraints.
        - Outputs: The optimizer `p` and the optimal value.

4. **Function Minimization**
    - `prob4()`: 
        - Problem: Minimize a specific convex function over a given domain.
        - Outputs: The optimizer `x` and the optimal value.

5. **Optimization with Parameters**
    - `prob5(A, b)`:
        - Problem: Solve a particular convex optimization problem.
        - Parameters: Matrices `A` and vector `b`.
        - Outputs: The optimizer `x` and the optimal value.

6. **College Student Food Problem**
    - `prob6()`: 
        - Problem: Determine the optimal food mix based on nutritional and cost constraints.
        - Data: Utilizes input from the `food.npy` file.
        - Outputs: The optimizer `x` and the optimal value.

## How to Use

1. Import the requisite functions from this module.
2. Define or input the necessary parameters for the desired function.
3. Execute the function and subsequently interpret the outcomes.

## Dependencies

- CVXPY

## Conclusion

This set of functions vividly illuminates the versatility and prowess of CVXPY in solving convex optimization challenges. Leveraging CVXPY, users can express complex optimization problems in an intuitive syntax and rapidly derive optimal solutions.
