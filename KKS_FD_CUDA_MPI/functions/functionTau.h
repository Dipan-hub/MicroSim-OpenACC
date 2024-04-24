#ifndef FUNCTIONTAU_CUH_
#define FUNCTIONTAU_CUH_

#include "structures.h"  // Ensure this header is ready for OpenACC.
#include "matrix.h"  
#include "utilityFunctions.h"  // Ensure these utilities are compatible with OpenACC.

#pragma acc routine seq
double FunctionTau(double **phi, double *relaxCoeff, long idx, long NUMPHASES);

// Declaration of calculateTau function, ensure the implementation is compatible with OpenACC.
void calculateTau(domainInfo *simDomain, controls *simControls, simParameters *simParams);

#endif
