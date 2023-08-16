# Interpolation Techniques Project Description

Interpolation techniques such as the Lagrange method and the Barycentric method are fundamental in numerical analysis. This project aims to understand and implement these techniques with a core focus on interpolating given data points to a polynomial of the least possible degree. Real-world applications such as interpolating air quality data and comparing different methods showcase the value and practicality of these interpolation methods.

## Mathematical Background and Overview of Functions

Interpolation is the process of determining a polynomial that fits a given set of data points. The polynomial should be of the least possible degree such that it exactly passes through all the provided data points. 

- **Lagrange Interpolation**: Given \( n \) data points, the Lagrange interpolation constructs an \( n-1 \) degree polynomial that passes through these points. The formula is:
\[ P(x) = \sum_{i=0}^{n-1} y_i \times L_i(x) \]
Where:
\[ L_i(x) = \prod_{0 \leq j \leq n, j \neq i} \frac{x - x_j}{x_i - x_j} \]

- **Barycentric Interpolation**: An efficient method to compute the Lagrange interpolating polynomial. Rather than calculating the Lagrange basis polynomials explicitly, it employs precomputed weights to facilitate quicker evaluations.

1. **Lagrange Method**
    - `lagrange(xint, yint, points)`: Returns the value of the Lagrange interpolating polynomial at the specified points.

2. **Barycentric Method**
    - `Barycentric`: A class designed for Barycentric Lagrange interpolation. This class handles the computation of Barycentric weights and evaluates the interpolating polynomial using these weights.

3. **Error Comparison of Interpolation Methods**
    - `prob5()`: Compares interpolation errors for Runge's function using various methods and represents the absolute error graphically.

4. **Chebyshev Polynomials and Interpolation**
    - `chebyshev_coeffs(f, n)`: Computes Chebyshev coefficients for an interpolating polynomial. Chebyshev polynomials are essential for optimal interpolation, particularly in regions where traditional methods might have considerable errors.

5. **Interpolation of Air Quality Data**
    - `prob7(n)`: Illustrates how the Barycentric Lagrange interpolation method can be applied to interpolate air quality data, presenting both the original dataset and the interpolated polynomial graphically.

## Project Flow

1. **Lagrange Interpolation**:
    - Implement the `lagrange` function to determine the polynomial that represents given data points and evaluate it at desired locations.

2. **Barycentric Interpolation**:
    - Use the `Barycentric` class to handle interpolation tasks efficiently. This class allows users to add more data points if required, without recomputing everything from the start.
    
3. **Error Evaluation**:
    - Utilize `prob5` to investigate the difference in results between various interpolation methods, offering visual insights into their performance.

4. **Efficient Interpolation with Chebyshev Polynomials**:
    - Employ `chebyshev_coeffs` to harness the power of Chebyshev polynomials for efficient interpolation.

5. **Real-World Data Application**:
    - With `prob7`, delve into real-world applications by interpolating air quality data, demonstrating the practical utility of these methods.

## How to Use

1. Import the necessary functions and classes from this module.
2. Provide the required data points, function, or any other relevant information.
3. Invoke the desired function or method and interpret the results visually or analytically.

## Dependencies

To use the functions and classes provided in this module, ensure that you have the following libraries installed and available:

- numpy: Used for numerical operations.
    ```python
    import numpy as np
    ```
- scipy: Contains the BarycentricInterpolator and linear algebra functionalities.
    ```python
    from scipy.interpolate import BarycentricInterpolator
    from scipy import linalg as la
    ```
- matplotlib: Used for visualization and plotting purposes.
    ```python
    from matplotlib import pyplot as plt
    ```

Ensure these libraries are correctly installed and imported before executing the interpolation functions and methods.



