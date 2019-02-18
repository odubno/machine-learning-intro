# Machine Learning

## Matrix Differentiation

* Defining what matrix differentiation is.
  * https://betterexplained.com/articles/vector-calculus-understanding-the-gradient/ 
* Refresher on linear algebra, drivatives and gradients.
  * https://atmos.washington.edu/~dennis/MatrixCalculus.pdf
* Khan tutorials on multivariable culculus. Jump straight to the Jacobian.  
  * https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives
* Step by step solutions of gradients.
  * http://www.cs.huji.ac.il/~csip/tirgul3_derivatives.pdf
  
### Determinant
  * Tells us the size by which the [unit vectors](https://en.wikipedia.org/wiki/Unit_vector) will grow by i.e. be transformed by.
  * The determinant tells us the factor by which areas are stretch out by (increased by).
```
   i j
  [1 0]
  [0 1]
```
```
  [3 1]
  [0 2]

^ has a derminant of 6 i.e. (3*2)-(1*0)
That means 

  [1 0]
  [0 1]

will grow 6 times it's size; the area will be 6 times the original size of the unit vectors.
```
 * If the determinant is zero, there is no inverse.
