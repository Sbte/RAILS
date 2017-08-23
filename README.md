# RAILS: Residual Approximation based Iterative Lyapunov Solver

RAILS is an iterative solver for generalized continuous time Lyapunov equations. These equations are of the form

*A\*X\*M'+M\*X\*A'+B\*B' = 0*,

where ' denotes the transpose, A and M are mxm, X is mxm and symmetric, and B is mxn where n << m. We solve for a low-rank approximation of *X* of the form *X=V\*T\*V'*. The method work by expanding the search space in every iteration by eigenvectors of the residual

*R = A\*V\*T\*V'\*M'+M\*V\*T\*V'\*A'+B\*B'*.

This method can be restarted, and therefore can limit the size of the search space, unlike many other method. It is especially well suited when one wants to perform multiple computations on similar systems since one can use the approximate solution from the previous computation as an initial guess (for instance in a continuation), or when n is relatively large since we can use any amount of eigenvectors of the residual to expand the space (as long as it's smaller or equal to n).

This repository contains both a C++ and a Matlab implementation of the algorithm.

## Installation

The code can be compiled using cmake. One can for instance run

```
mkdir build
cd build
cmake ../
make
```

This will build the code in the `build` directory. If Trilinos is present in your cmake path, it will also compile Trilinos wrappers so the code can be used in combination with Trilinos.

An extensive test suite is also present, which can be run using

```
make test
```

## Dependencies

* SLICOT
* Trilinos (optional)

## Reference

The algorithm was presented in

S\. Baars, J.P. Viebahn, T.E. Mulder, C. Kuehn, F.W. Wubs, H.A. Dijkstra, Continuation of probability density functions using a generalized Lyapunov approach, Journal of Computational Physics, Volume 336, 2017, Pages 627-643, ISSN 0021-9991, http://dx.doi.org/10.1016/j.jcp.2017.02.021

Please cite this paper when using the code in this repository.
