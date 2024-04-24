#include <cmath>
#include <openacc.h>

#include "utilityKernels.cuh"
#include <cub/device/device_reduce.cuh>

/*
 * This function will compute the change in every cell
 * as the absolute value of [B - A]
 *
 * The differences are stored in A.
 *
 * In practice, A is the old field, B is the new field.
 */
#pragma acc routine seq
void computeChange(double *A, double *B, long DIMENSION,
                   long sizeX, long sizeY, long sizeZ)
{
    long totalSize = sizeX * sizeY * sizeZ;
    #pragma acc parallel loop present(A[:totalSize], B[:totalSize])
    for (long i = 0; i < sizeX; i++) {
        for (long j = 0; j < (DIMENSION >= 2 ? sizeY : 1); j++) {
            for (long k = 0; k < (DIMENSION == 3 ? sizeZ : 1); k++) {
                long idx = (j + i * sizeY) * sizeZ + k;
                if (i < sizeX && ((j < sizeY && DIMENSION >= 2) || (DIMENSION == 1 && j == 0)) && ((k < sizeZ && DIMENSION == 3) || (DIMENSION < 3 && k == 0))) {
                    A[idx] = fabs(B[idx] - A[idx]);
                }
            }
        }
    }
}

/*
 * Reset each array element to zero for all given arrays
 */

#pragma acc routine seq
void resetArray(double **arr, long numArr,
                long DIMENSION,
                long sizeX, long sizeY, long sizeZ)
{
    long totalSize = sizeX * sizeY * sizeZ;
    #pragma acc parallel loop present(arr[:numArr][:totalSize])
    for (long iter = 0; iter < numArr; iter++) {
        for (long i = 0; i < sizeX; i++) {
            for (long j = 0; j < (DIMENSION >= 2 ? sizeY : 1); j++) {
                for (long k = 0; k < (DIMENSION == 3 ? sizeZ : 1); k++) {
                    long idx = (j + i * sizeY) * sizeZ + k;
                    if (i < sizeX && ((j < sizeY && DIMENSION >= 2) || (DIMENSION == 1 && j == 0)) && ((k < sizeZ && DIMENSION == 3) || (DIMENSION < 3 && k == 0))) {
                        arr[iter][idx] = 0.0;
                    }
                }
            }
        }
    }
}

void printStats(double **phi, double **comp,
                double **phiNew, double **compNew,
                double *maxerr, double *maxVal, double *minVal,
                domainInfo simDomain, subdomainInfo subdomain)
{
    long i, j = 0;
    double tempMax, tempMin;

    // Allocate memory for temporary results on the device
    #pragma acc enter data copyin(phi[:simDomain.numPhases][:subdomain.numCompCells], comp[:simDomain.numComponents-1][:subdomain.numCompCells], phiNew[:simDomain.numPhases][:subdomain.numCompCells], compNew[:simDomain.numComponents-1][:subdomain.numCompCells], maxerr[:simDomain.numPhases+simDomain.numComponents-1], maxVal[:simDomain.numPhases+simDomain.numComponents-1], minVal[:simDomain.numPhases+simDomain.numComponents-1])

    // Get all stats for compositions
    for (i = 0; i < simDomain.numComponents-1; i++) {
        #pragma acc parallel loop reduction(max:tempMax) reduction(min:tempMin) present(compNew[i][:subdomain.numCompCells], comp[i][:subdomain.numCompCells])
        for (long idx = 0; idx < subdomain.numCompCells; idx++) {
            tempMax = fmax(tempMax, compNew[i][idx]);
            tempMin = fmin(tempMin, compNew[i][idx]);
            comp[i][idx] = fabs(compNew[i][idx] - comp[i][idx]);  // Compute change
        }

        maxVal[j] = tempMax;
        minVal[j] = tempMin;

        #pragma acc parallel loop reduction(max:tempMax) present(comp[i][:subdomain.numCompCells])
        for (long idx = 0; idx < subdomain.numCompCells; idx++) {
            tempMax = fmax(tempMax, comp[i][idx]);
        }

        maxerr[j] = tempMax;
        j++;
    }

    // Get all stats for phi
    for (i = 0; i < simDomain.numPhases; i++) {
        #pragma acc parallel loop reduction(max:tempMax) reduction(min:tempMin) present(phiNew[i][:subdomain.numCompCells], phi[i][:subdomain.numCompCells])
        for (long idx = 0; idx < subdomain.numCompCells; idx++) {
            tempMax = fmax(tempMax, phiNew[i][idx]);
            tempMin = fmin(tempMin, phiNew[i][idx]);
            phi[i][idx] = fabs(phiNew[i][idx] - phi[i][idx]);  // Compute change
        }

        maxVal[j] = tempMax;
        minVal[j] = tempMin;

        #pragma acc parallel loop reduction(max:tempMax) present(phi[i][:subdomain.numCompCells])
        for (long idx = 0; idx < subdomain.numCompCells; idx++) {
            tempMax = fmax(tempMax, phi[i][idx]);
        }

        maxerr[j] = tempMax;
        j++;
    }

    #pragma acc exit data delete(phi, comp, phiNew, compNew, maxerr, maxVal, minVal)
}

void resetArray(double **arr, long numArr,
                long DIMENSION,
                long sizeX, long sizeY, long sizeZ)
{
    long totalSize = sizeX * sizeY * sizeZ;

    #pragma acc data present(arr[:numArr][:totalSize])
    {
        #pragma acc parallel loop collapse(3)
        for (long iter = 0; iter < numArr; iter++) {
            for (long i = 0; i < sizeX; i++) {
                for (long j = 0; j < (DIMENSION >= 2 ? sizeY : 1); j++) {
                    for (long k = 0; k < (DIMENSION == 3 ? sizeZ : 1); k++) {
                        long idx = (j + i * sizeY) * sizeZ + k;
                        arr[iter][idx] = 0.0;
                    }
                }
            }
        }
    }
}

