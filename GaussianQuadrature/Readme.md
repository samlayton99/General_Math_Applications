# Gaussian Quadrature Integration Project Description

Gaussian Quadrature is a prominent numerical technique aimed at approximating definite integrals. Unlike elementary integration methods like the trapezoidal or Simpson’s rule, Gaussian Quadrature leverages orthogonal polynomials and their zeros, known as nodes, together with particular weights for each node to achieve highly precise integration approximations. In this undertaking, the emphasis is on deploying Legendre and Chebyshev polynomials for this procedure.

## Mathematical Background and Overview of Functions

The backbone of Gaussian Quadrature is hinged on the utilization of orthogonal polynomials. These are polynomials that retain orthogonality over a defined interval with respect to a weighting function. Two of the most commonly employed orthogonal polynomials in Gaussian Quadrature are the Legendre and Chebyshev.

1. **Orthogonal Polynomials**
   - These are polynomials that remain orthogonal over a specified interval with respect to a weighting function.
   
2. **Legendre Polynomials**
   - They are orthogonal over the interval [-1, 1] without a weight function.
   
3. **Chebyshev Polynomials**
   - These are orthogonal over the interval [-1, 1] but with the weight function \( w(x) = \sqrt{1-x^2} \).
   
4. **Jacobi Matrix**
   - A symmetric matrix that is tied to orthogonal polynomials. The eigenvalues of this matrix yield the nodes, and the square of the primary element of its eigenvectors (scaled appropriately) gives the weights essential for the Gaussian Quadrature.

## Project Flow:

1. **Initialization**: Begin by initializing the Gaussian Quadrature with a chosen number of points (nodes) and the type of polynomial (either Legendre or Chebyshev).
   
2. **Node and Weight Calculation**: Use the Jacobi Matrix corresponding to the orthogonal polynomial to determine the nodes and weights.
   
3. **Standard Integration**: A basic method of integration is used for functions that are defined over the standard [-1,1] interval.
   
4. **Integration Transformation**: For functions defined over varying intervals, there's a transformation of the integration limits to [-1,1], post which the standard method is used.
   
5. **2D Integration**: For two-dimensional functions, a double summation technique is used over the product of weights and function values at the 2D nodes.
   
6. **Visualization and Comparison**: Utilize the `prob5` function to visually juxtapose the accuracy of Gaussian Quadrature integration employing both Legendre and Chebyshev polynomials against precise integral values for a certain function.

## How to Use

1. Import necessary classes or functions from the module.
   
2. Initialize Gaussian Quadrature with the number of nodes and polynomial type of choice.
   
3. Use appropriate methods to compute the integral of the desired function over the specified interval.
   
4. Call the `prob5` function for a comparative visualization of accuracy against exact integral values.

## Dependencies

The project makes use of the following libraries and their respective modules:

- **numpy**: Utilized as `np`
- **scipy**: 
  - `linalg` module, used as `la`
  - `stats` module, specifically the `norm` function
- **matplotlib**: `pyplot` module, used as `plt`

## Testing Code

To conduct a test, uncomment the `prob5` function located at the tail end of the script, allowing a comparison of Gaussian Quadrature's accuracy against exact integration values:

```python
prob5()
