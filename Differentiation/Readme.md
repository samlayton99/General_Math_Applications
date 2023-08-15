# Differentiation Description

Derivatives play a key role in understanding how functions change at specific points. Depending on the context, they can be calculated symbolically, numerically, or with differentiation software. In this project, I explore these methods, including symbolic derivatives, difference quotients, and Jacobian matrix calculations, highlighting their advantages and limitations. Such understanding is crucial in numerical analysis, especially when analytical solutions are out of reach.

# Mathematical Background and Overview of Functions

Derivative approximations are at the core of many numerical methods. These methods allow us to gauge the behavior of a function based on its rate of change. Several techniques and mathematical constructs play pivotal roles in this domain.

1. **Derivative**:
    - The derivative of a function provides insight into how the function changes as its input changes. For any function \(f(x)\), its derivative \(f'(x)\) represents the instantaneous rate of change of the function.
    
2. **Difference Quotients**:
    - Difference quotients are numerical approximations to derivatives. They can be thought of as the average rate of change of a function over a small interval. Various types of difference quotients include:
        - **First Order Forward Difference Quotient**:
            \[ f'(x) \approx \frac{f(x+h) - f(x)}{h} \]
        - **Second Order Forward Difference Quotient**: 
            \[ f'(x) \approx \frac{-3f(x) + 4f(x+h) - f(x+2h)}{2h} \]
        - **First Order Backward Difference Quotient**: 
            \[ f'(x) \approx \frac{f(x) - f(x-h)}{h} \]
        - **Second Order Backward Difference Quotient**: 
            \[ f'(x) \approx \frac{3f(x) - 4f(x-h) + f(x-2h)}{2h} \]
        - **Second Order Centered Difference Quotient**: 
            \[ f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} \]
        - **Fourth Order Centered Difference Quotient**: 
            \[ f'(x) \approx \frac{f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)}{12h} \]
        
3. **Jacobian Matrix**:
    - For functions that map from one vector space to another, the Jacobian matrix is a crucial construct that contains all first-order partial derivatives of the function. It provides a linear approximation to the function near a given point.

## How to Use
1. Import the necessary functions from the provided module.
2. Define the function for which you wish to compute the derivative or any other derivative-related construct.
3. Use the relevant function to compute the desired output, passing in the appropriate parameters.

## Dependencies
- numpy
- sympy
- matplotlib

