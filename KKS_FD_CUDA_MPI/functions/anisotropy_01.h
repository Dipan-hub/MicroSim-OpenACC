#ifndef ANISOTROPY_01_H_
#define ANISOTROPY_01_H_

#include <cmath> // Replacing CUDA headers with standard C++ header

#pragma acc routine seq
void anisotropy_01_dAdq(double *qab, double *dadq, long a, long b, double *dab, long NUMPHASES);

#pragma acc routine seq
double anisotropy_01_function_ac(double *qab, long a, long b, double *dab, long NUMPHASES);

#endif // ANISOTROPY_01_H_
