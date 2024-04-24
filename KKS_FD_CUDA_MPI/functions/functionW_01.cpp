#include "functionW_01.h"

#include <cmath>
#include <algorithm>


/*
 * calcDoubleObstacleDerivative
 *
 * Calculate g'(\phi)
 *
 * Arguments:
 *              1. double **phi -> all the phase volume-fraction values
 *              2. long phase -> differentiate wrt to this phase
 *              3. double *theta_i   -> coefficients for theta_i, one per phase
 *              4. double *theta_ij  -> coefficients for theta_ij, one per pair of phases
 *              5. double *theta_ijk -> coefficients for theta_ijk, one per triplet of phases
 *              6. double **Gamma -> interaction coefficients between two phases
 *              7. double ***Gamma_abc -> interaction coefficients among three phases
 *              8. long idx -> position of cell in 1D
 *              9. long NUMPHASES -> number of phases
 * Return:
 *              numerical evaluation of derivative of interpolation polynomial, as a double datatype
 */
#pragma acc routine seq
double calcDoubleObstacleDerivative(double **phi, long phase,
                                    double *theta_i, double *theta_ij, double *theta_ijk,
                                    double **Gamma, double ***Gamma_abc,
                                    long idx, long NUMPHASES)
{
    if (NUMPHASES < 2)
        return 0.0;
  
    long b,c;
    double sum = 0.0;
/*
    // Parallelize the first level loop using OpenACC
    #pragma acc parallel loop reduction(+:sum) present(phi, Gamma)
    for (b = 0; b < NUMPHASES; b++) {
        if (b != phase) {
            double phi_b = phi[b][idx];
            sum += (phi[phase][idx] * phi_b >= 0.0 ? 1.0 : -1.0) * Gamma[phase][b] * phi_b;
        }
    }
    sum *= 16.0 / (M_PI * M_PI);

    // Parallelize the nested loop using OpenACC
    #pragma acc parallel loop collapse(2) reduction(+:sum) present(phi, Gamma_abc)
    for (b = 0; b < NUMPHASES; b++) {
        for ( c = 0; c < NUMPHASES; c++) {
            if (b != phase && c != phase && b < c) {
                double phi_b = phi[b][idx];
                double phi_c = phi[c][idx];
                double phibphic = phi_b * phi_c;
                sum += (phi[phase][idx] * phibphic >= 0.0 ? 1.0 : -1.0) * Gamma_abc[phase][b][c] * phibphic;
            }
        }
    }*/

    return sum;
}
