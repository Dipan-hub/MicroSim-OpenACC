#ifndef MATRIX_CUH_
#define MATRIX_CUH_

#include <cmath>  // Replace CUDA-specific includes with standard C++ headers
#include <cstdio>

#include "structures.h"
#include "utilityFunctions.h"

// Ensure all functions are properly declared for use with OpenACC

// Matrix operations
void matinvnew(double **coeffmatrix, double **inv, long size);
void multiply(double **inv, double *y, double *prod, long size);
void multiply2d(double **m1, double **m2, double **prod, long size);
void vectorsum(double *y1, double *y2, double *sum, long size);
void substituteb(double **fac, double *y, double *vec, long size);
void substitutef(double **fac, double **y1, int index, double *vec, long size);
void pivot(double **coeffmatrix, double **factor, int k, int *tag, long size);
void colswap(double **m1, double **m2, int *tag, long size);
void rowswap(double **m1, double **m2, int *tag, long size);

// Host-side functions
int LUPDecompose(double **A, int N, double Tol, int *P);
void LUPInvert(double **A, int *P, int N, double **IA);
void matrixMultiply(double **A, double **B, double **C, int N);

// OpenACC annotations for functions that were originally device-side
#pragma acc routine seq
int LUPDecomposeC1(double A[][MAX_NUM_COMP], long N, double Tol, int *P);
int LUPDecomposeC2(double A[(MAX_NUM_COMP)*(MAX_NUM_COMP)], long N, double Tol, int *P);
void LUPSolveC1(double A[][MAX_NUM_COMP], int *P, double *b, long N, double *x);
void LUPSolveC2(double A[(MAX_NUM_COMP)*(MAX_NUM_COMP)], int *P, double *b, long N, double *x);
void LUPInvertC1(double A[][MAX_NUM_COMP], int *P, long N, double IA[][MAX_NUM_COMP]);
void LUPInvertC2(double A[(MAX_NUM_COMP)*(MAX_NUM_COMP)], int *P, long N, double IA[][MAX_NUM_COMP]);
int LUPDecomposePC1(double A[][MAX_NUM_PHASE_COMP], long N, double Tol, int *P);
int LUPDecomposePC2(double A[(MAX_NUM_PHASE_COMP)*(MAX_NUM_PHASE_COMP)], long N, double Tol, int *P);
void LUPSolvePC1(double A[][MAX_NUM_PHASE_COMP], int *P, double *b, long N, double *x);
void LUPSolvePC2(double A[(MAX_NUM_PHASE_COMP)*(MAX_NUM_PHASE_COMP)], int *P, double *b, long N, double *x);
void LUPInvertPC1(double A[][MAX_NUM_PHASE_COMP], int *P, long N, double IA[][MAX_NUM_PHASE_COMP]);
void LUPInvertPC2(double A[(MAX_NUM_PHASE_COMP)*(MAX_NUM_PHASE_COMP)], int *P, long N, double IA[][MAX_NUM_PHASE_COMP]);

#endif
