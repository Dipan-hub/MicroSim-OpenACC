// anisotropy_01.h
#ifndef ANISOTROPY_01_H
#define ANISOTROPY_01_H

#pragma acc routine seq
void anisotropy_01_dAdq(double *qab, double *dadq, long a, long b, double *dab, long NUMPHASES);

#pragma acc routine seq
double anisotropy_01_function_ac(double *qab, long a, long b, double *dab, long NUMPHASES);

#endif // ANISOTROPY_01_H
