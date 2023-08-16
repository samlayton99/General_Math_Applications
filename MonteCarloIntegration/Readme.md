# Monte Carlo Integration Project Description

This project leverages the power of Monte Carlo methods to estimate complex integrals, especially in higher dimensions. While traditional numerical integration techniques, such as Newton-Cotes formulas and Gaussian quadrature, are efficient for one-dimensional integrals, they falter in multi-dimensional scenarios. Monte Carlo integration, despite its slower convergence, excels in these high-dimensional contexts. My exploration not only addresses various integration challenges but also evaluates the precision of our techniques by studying relative errors in specific statistical applications.

## Mathematical Background and Overview of Functions

Monte Carlo integration leverages the Law of Large Numbers and random sampling to compute the value of integrals. It provides a reliable method for approximating integrals, especially in higher dimensions, where traditional methods might fall short.

1. **n-dimensional Unit Ball Volume Estimation**
    - `ball_volume(n, N=10000)`: This function approximates the volume of an n-dimensional unit ball. By understanding the unit ball as the set of points at a given distance (<=1) from the origin, we can imagine its 2D form as a circle and its 3D form as a sphere.

2. **1D Integration**
    - `mc_integrate1d(f, a, b, N=10000)`: Approximates the integral of a function `f` over a one-dimensional interval [a, b]. It does so by sampling random points within the domain and averaging the function's values at these points.

3. **Multi-dimensional Integration**
    - `mc_integrate(f, mins, maxs, N=10000)`: Expanding on the previous function, this generalizes the integration technique to n-dimensions. The function `f` here must be adapted to accept a list of points.

4. **Joint Distribution Integration and Error Analysis**
    - `prob4()`: This specialized function concerns the integration of a joint distribution of four standard normal random variables over a defined domain. Furthermore, it computes and visualizes the relative error of the Monte Carlo approximation against an exact benchmark value.

## How to Use

1. Begin by importing the necessary functions from the `montecarlo_integration.py` module.
2. Define or choose the function you wish to integrate, its derivative, your initial guesses, and other essential parameters.
3. Invoke the desired function and analyze the returned results.

## Dependencies

- **numpy**: Fundamental package for scientific computing in Python.
- **scipy**: Advanced tools for mathematics, science, and engineering.
- **matplotlib**: Library for creating static, animated, and interactive visualizations in Python.

## Mathematical Concepts

1. **n-dimensional Unit Ball Volume**: Describes the set of points in n-dimensions that are at a distance of less than or equal to 1 from the origin. In simpler terms, in 2D it's a circle, and in 3D, it's a sphere.

2. **Monte Carlo Integration**: Relying on the Law of Large Numbers, this technique uses random sampling to estimate the integral value.

3. **Multivariate Normal Distribution**: This is a higher-dimensional extension of the one-dimensional normal distribution. The `prob4()` function uses the joint distribution of `n` standard normal random variables.

4. **Relative Error**: This metric offers a scale-free measure of error, calculated as the absolute error divided by the exact value's magnitude.

## Project Flow

1. **Basic Volume Estimation**: We initiate with the Monte Carlo method, estimating the volume of an n-dimensional unit ball.
2. **1D Integration**: We then transition to approximating a 1D function's integral using the random sampling method.
3. **Integration in Higher Dimensions**: The `mc_integrate()` function extends the 1D integration to n-dimensions.
4. **Analysis of Error**: With the `prob4()` function, we investigate the Monte Carlo method's error for a joint distribution involving four standard normal random variables. This function offers a visualization of error contrasted against the number of random samples used.

To engage with this project actively, execute the provided functions with the parameters of your choice and delve into the results. Pay particular attention to the `prob4()` function if you're intrigued by error analysis.
