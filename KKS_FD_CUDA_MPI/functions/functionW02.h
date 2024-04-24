#ifndef FUNCTIONW_02_H_
#define FUNCTIONW_02_H_

#include <stdio.h>
#include <cmath>  // Replace CUDA-specific includes with standard C++ headers
//#include <cstdio>

#include "structures.h"
#include "utilityFunctions.h"

/*
 * Calculate g'(\phi)
 *
 * Arguments:
 *              1. double **phi -> all the phase volume-fraction values
 *              2. long phase -> differentiate wrt to this phase
 *              3. double *theta_i   -> coefficients for theta_i, one per phase
 *              4. double *theta_ij  -> coefficients for theta_ij, one per pair of phases
 *              5. double *theta_ijk -> coefficients for theta_ijk, one per triplet of phases
 *              6. long idx -> position of cell in 1D
 *              7. long NUMPHASES -> number of phases
 * Return:
 *              numerical evaluation of derivative of interpolation polynomial, as a double datatype
 */
#pragma acc routine seq
double calcDoubleWellDerivative(double **phi, long phase,
                                double *theta_i, double *theta_ij, double *theta_ijk,
                                long idx, long NUMPHASES);

#endif
