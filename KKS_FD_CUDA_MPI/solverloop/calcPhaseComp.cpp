Here's the provided CUDA code converted to OpenACC C++. I've made the necessary changes to enable OpenACC directives and adapt the syntax accordingly:

```cpp
#include "calcPhaseComp.h"

void initMu(double **phi, double **comp, double **phaseComp, double **mu,
            long *thermo_phase, double temperature,
            long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
            long sizeX, long sizeY, long sizeZ,
            long xStep, long yStep, long padding)
{
    #pragma acc parallel loop collapse(3) present(phi[0:sizeX*sizeY*sizeZ], comp[0:sizeX*sizeY*sizeZ], phaseComp[0:(NUMCOMPONENTS-1)*NUMPHASES*sizeX*sizeY*sizeZ], mu[0:(NUMCOMPONENTS-1)*sizeX*sizeY*sizeZ], thermo_phase[0:NUMPHASES])
    for (long i = 0; i < sizeX; ++i) {
        for (long j = 0; j < sizeY; ++j) {
            for (long k = 0; k < sizeZ; ++k) {
                long idx = i*xStep + j*yStep + k;

                if (i < sizeX && ((j < sizeY && DIMENSION >= 2) || (DIMENSION == 1 && j == 0)) && ((k < sizeZ && DIMENSION == 3) || (DIMENSION < 3 && k == 0)))
                {
                    double y[MAX_NUM_COMP], mu0[MAX_NUM_COMP];
                    double sum = 0.0;

                    for (long phase = 0; phase < NUMPHASES; phase++)
                    {
                        // Bulk
                        if (phi[phase][idx] == 1.0)
                        {
                            sum = 0.0;

                            for (long i = 0; i < NUMCOMPONENTS-1; i++)
                            {
                                phaseComp[phase + NUMPHASES*i][idx] = comp[i][idx];
                                y[i] = comp[i][idx];
                                sum += y[i];
                            }

                            y[NUMCOMPONENTS-1] = 1.0 - sum;

                            (*Mu_tdb_dev[thermo_phase[phase]])(temperature, y, mu0);

                            for (long i = 0; i < NUMCOMPONENTS-1; i++)
                                mu[i][idx] = mu0[i];
                        }
                        else
                        {
                            for (long i = 0; i < NUMCOMPONENTS-1; i++)
                                phaseComp[phase + NUMPHASES*i][idx] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

void calcPhaseComp(double **phi, double **comp,
                   double **phaseComp, double **mu,
                   domainInfo* simDomain, controls* simControls,
                   simParameters* simParams, subdomainInfo* subdomain,
                   dim3 gridSize, dim3 blockSize)
{
    if (simControls->FUNCTION_F == 1 || simControls->FUNCTION_F == 3 || simControls->FUNCTION_F == 4)
    {
        #pragma acc parallel loop collapse(3) present(phi[0:subdomain->sizeX*subdomain->sizeY*subdomain->sizeZ], comp[0:subdomain->sizeX*subdomain->sizeY*subdomain->sizeZ], phaseComp[0:(simDomain->numComponents-1)*simDomain->numPhases*subdomain->sizeX*subdomain->sizeY*subdomain->sizeZ], mu[0:(simDomain->numComponents-1)*subdomain->sizeX*subdomain->sizeY*subdomain->sizeZ], simParams->F0_A_dev[0:(simDomain->numComponents-1)*(simDomain->numComponents-1)*simDomain->numPhases], simParams->F0_B_dev[0:(simDomain->numComponents-1)*simDomain->numPhases], simParams->F0_C_dev[0:simDomain->numPhases], simDomain->thermo_phase_dev[0:simDomain->numPhases])
        for (long i = 0; i < subdomain->sizeX; ++i) {
            for (long j = 0; j < subdomain->sizeY; ++j) {
                for (long k = 0; k < subdomain->sizeZ; ++k) {
                    long idx = i*subdomain->xStep + j*subdomain->yStep + k;
                    __calcPhaseComp__<<<1, 1>>>(phi, comp, phaseComp, simParams->F0_A_dev, simParams->F0_B_dev, simParams->F0_C_dev, simDomain->numPhases, simDomain->numComponents, simDomain->DIMENSION, subdomain->sizeX, subdomain->sizeY, subdomain->sizeZ, subdomain->xStep, subdomain->yStep, subdomain->padding);
                }
            }
        }
    }
    else if (simControls->FUNCTION_F == 2)
    {
        if (simControls->startTime == simControls->count && (simControls->restart == 0 && simControls->startTime == 0))
        {
            #pragma acc parallel loop collapse(3) present(phi[0:subdomain->sizeX*subdomain->sizeY*subdomain->sizeZ], comp[0:subdomain->sizeX*subdomain->sizeY*subdomain->sizeZ], phaseComp[0:(simDomain->numComponents-1)*simDomain->numPhases*subdomain->sizeX*subdomain->sizeY*subdomain->sizeZ], mu[0:(simDomain->numComponents-1)*subdomain->sizeX*subdomain->sizeY*subdomain->sizeZ], simDomain->thermo_phase_dev[0:simDomain->numPhases])
            for (long i = 0; i < subdomain->sizeX; ++i) {
                for (long j = 0; j < subdomain->sizeY; ++j) {
                    for (long k = 0; k < subdomain->sizeZ; ++k) {
                        long idx = i*subdomain->xStep + j*subdomain->yStep + k;
                        __initMu__<<<1, 1>>>(phi, comp, phaseComp, mu, simDomain->thermo_phase_dev, simParams->Teq, simDomain->numPhases, simDomain->numComponents, simDomain->DIMENSION, subdomain->sizeX, subdomain->sizeY, subdomain->sizeZ, subdomain->xStep, subdomain->yStep, subdomain->padding);
                    }
                }
            }
        }
        #pragma acc parallel loop collapse(3) present(phi[0:subdomain->sizeX*subdomain->sizeY*subdomain->sizeZ], comp[0:subdomain->sizeX*subdomain->sizeY*subdomain->sizeZ], phaseComp[0:(simDomain->numComponents-1)*simDomain->numPhases*subdomain->sizeX*subdomain->sizeY*subdomain
