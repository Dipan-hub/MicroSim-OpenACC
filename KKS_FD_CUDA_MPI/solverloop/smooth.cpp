#include "smooth.cuh"

#define phiAniso(phase, x, y, z) (phiAniso[(((phase)*3 + (x))*3 + (y))*3 + (z)])

#pragma acc kernels
void __smooth__(double **phi, double **phiNew,
                double *relaxCoeff, double *kappaPhi,
                double *dab, double *Rotation_matrix, double *Inv_rotation_matrix, int FUNCTION_ANISOTROPY,
                long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                long sizeX, long sizeY, long sizeZ,
                long xStep, long yStep, long padding,
                double DELTA_X, double DELTA_Y, double DELTA_Z,
                double DELTA_t)
{
    #pragma acc data copyin(phi[:NUMPHASES][:sizeX*sizeY*sizeZ]), \
                      copy(phiNew[:NUMPHASES][:sizeX*sizeY*sizeZ]), \
                      copyin(relaxCoeff[:NUMPHASES], kappaPhi[:NUMPHASES*NUMPHASES], \
                             dab[:NUMPHASES*NUMCOMPONENTS], Rotation_matrix[:9], Inv_rotation_matrix[:9])
    {
        #pragma acc parallel loop collapse(3) independent
        for (long i = padding; i < sizeX-padding; i++) {
            for (long j = padding; j < sizeY-padding; j++) {
                for (long k = padding; k < sizeZ-padding; k++) {
                    if ((DIMENSION < 2 && j > 0) || (DIMENSION < 3 && k > 0)) continue;

                    long index[3][3][3];
                    double phiAniso[MAX_NUM_PHASES*27];
                    double aniso[MAX_NUM_PHASES] = {0.0};
                    double dfdphiSum = 0.0;
                    long phase, p, x, y, z;

                    // Populate index matrix
                    for (x = 0; x < 3; x++) {
                        for (y = 0; y < 3; y++) {
                            for (z = 0; z < 3; z++) {
                                index[x][y][z] = (k+z-1) + (j+y-1)*yStep + (i+x-1)*xStep;
                            }
                        }
                    }

                    // Calculate phiAniso for each phase and direction
                    for (phase = 0; phase < NUMPHASES; phase++) {
                        for (x = 0; x < 3; x++) {
                            for (y = 0; y < 3; y++) {
                                for (z = 0; z < 3; z++) {
                                    phiAniso(phase, x, y, z) = phi[phase][index[x][y][z]];
                                }
                            }
                        }
                    }

                    // Compute anisotropy based on the FUNCTION_ANISOTROPY setting
                    if (FUNCTION_ANISOTROPY == 1 || FUNCTION_ANISOTROPY == 2) {
                        for (phase = 0; phase < NUMPHASES; phase++) {
                            aniso[phase] = calcAnisotropy_01(phiAniso, dab, kappaPhi, Rotation_matrix, Inv_rotation_matrix, phase, NUMPHASES, DIMENSION, DELTA_X, DELTA_Y, DELTA_Z);
                        }
                    }

                    // Compute dfdphiSum for each phase considering interactions between phases
                    for (phase = 0; phase < NUMPHASES; phase++) {
                        dfdphiSum = 0.0;
                        for (p = 0; p < NUMPHASES; p++) {
                            if (p == phase) continue;
                            dfdphiSum += 2.0 * (FUNCTION_ANISOTROPY == 0 ? kappaPhi[phase * NUMPHASES + p] : 1) * (aniso[p] - aniso[phase]);
                        }
                        phiNew[phase][index[1][1][1]] = phi[phase][index[1][1][1]] - DELTA_t * FunctionTau(phi, relaxCoeff, index[1][1][1], NUMPHASES) * dfdphiSum / NUMPHASES;
                    }
                }
            }


void smooth(double **phi, double **phiNew,
            domainInfo* simDomain, controls* simControls,
            simParameters* simParams, subdomainInfo* subdomain)
{
    long sizeX = subdomain->sizeX;
    long sizeY = subdomain->sizeY;
    long sizeZ = subdomain->sizeZ;
    long xStep = subdomain->xStep;
    long yStep = subdomain->yStep;
    long padding = subdomain->padding;
    double DELTA_X = simDomain->DELTA_X;
    double DELTA_Y = simDomain->DELTA_Y;
    double DELTA_Z = simDomain->DELTA_Z;
    double DELTA_t = simControls->DELTA_t;
    int FUNCTION_ANISOTROPY = 0;  // Adjust this based on actual control logic if variable.

    // Setup OpenACC data regions and manage data explicitly
    #pragma acc data copyin(phi[:simDomain->numPhases][sizeX*sizeY*sizeZ]), \
                      copy(phiNew[:simDomain->numPhases][sizeX*sizeY*sizeZ]), \
                      copyin(simParams->relax_coeff_dev[:simDomain->numPhases], \
                             simParams->kappaPhi_dev[:simDomain->numPhases*simDomain->numPhases], \
                             simParams->dab_dev[:simDomain->numComponents], \
                             simParams->Rotation_matrix_dev[:9], simParams->Inv_Rotation_matrix_dev[:9])
    {
        // Parallel execution region setup
        #pragma acc parallel loop collapse(3) independent
        for (long i = padding; i < sizeX - padding; i++) {
            for (long j = padding; j < sizeY - padding; j++) {
                for (long k = padding; k < sizeZ - padding; k++) {
                    if ((simDomain->DIMENSION < 2 && j > 0) || (simDomain->DIMENSION < 3 && k > 0)) continue;

                    // Directly call the adapted OpenACC version of __smooth__
                    // Note: __smooth__ needs to be adapted to use OpenACC directives if not already done
                    __smooth__(phi, phiNew,
                               simParams->relax_coeff_dev, simParams->kappaPhi_dev,
                               simParams->dab_dev, simParams->Rotation_matrix_dev, simParams->Inv_Rotation_matrix_dev, FUNCTION_ANISOTROPY,
                               simDomain->numPhases, simDomain->numComponents, simDomain->DIMENSION,
                               sizeX, sizeY, sizeZ,
                               xStep, yStep, padding,
                               DELTA_X, DELTA_Y, DELTA_Z, DELTA_t);
                }
            }
        }
    }
}
