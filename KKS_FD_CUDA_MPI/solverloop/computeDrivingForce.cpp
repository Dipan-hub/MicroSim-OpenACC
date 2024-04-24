#include "computeDrivingForce.h"

void computeDrivingForce_Chemical(double **phi, double **comp,
                                  double **dfdphi,
                                  double **phaseComp, double **mu,
                                  domainInfo* simDomain, controls* simControls,
                                  simParameters* simParams, subdomainInfo* subdomain,
                                  dim3 gridSize, dim3 blockSize)
{

    if (simControls->FUNCTION_F == 1 || simControls->FUNCTION_F == 3 || simControls->FUNCTION_F == 4)
    {
#pragma acc parallel loop collapse(3) present(phi[:][:], comp[:][:], dfdphi[:][:], phaseComp[:][:], \
                                            simParams->F0_A_dev[:], simParams->F0_B_dev[:], simParams->F0_C_dev[:], \
                                            simParams->theta_i_dev[:], simParams->theta_ij_dev[:], simParams->theta_ijk_dev[:])
        for (long i = subdomain->padding; i < subdomain->sizeX - subdomain->padding; ++i)
        {
            for (long j = (simDomain->DIMENSION == 1) ? 0 : subdomain->padding; j < ((simDomain->DIMENSION >= 2) ? subdomain->sizeY - subdomain->padding : 1); ++j)
            {
                for (long k = (simDomain->DIMENSION < 3) ? 0 : subdomain->padding; k < ((simDomain->DIMENSION == 3) ? subdomain->sizeZ - subdomain->padding : 1); ++k)
                {
                    long idx = i * subdomain->xStep + j * subdomain->yStep + k;

                    double psi = 0.0;

                    double Bpq_hphi[MAX_NUM_PHASES];

                    for (long p = 0; p < simDomain->numPhases; ++p)
                    {
                        Bpq_hphi[p] = dfdphi[p][idx];
                    }

                    for (long phase = 0; phase < simDomain->numPhases; ++phase)
                    {
                        if (simControls->ELASTICITY)
                        {
                            dfdphi[phase][idx] = 0.0;

                            for (long p = 0; p < simDomain->numPhases; ++p)
                            {
                                dfdphi[phase][idx] += calcInterp5thDiff(phi, phase, p, idx, simDomain->numPhases) * Bpq_hphi[p];
                            }
                        }

                        for (long p = 0; p < simDomain->numPhases; ++p)
                        {
                            psi = 0.0;

                            psi += calcPhaseEnergy(phaseComp, p, simParams->F0_A_dev, simParams->F0_B_dev, simParams->F0_C_dev, idx, simDomain->numPhases, simDomain->numComponents);

                            for (long component = 0; component < simDomain->numComponents - 1; ++component)
                            {
                                psi -= calcDiffusionPotential(phaseComp, p, component, simParams->F0_A_dev, simParams->F0_B_dev, idx, simDomain->numPhases, simDomain->numComponents) * phaseComp[(component * simDomain->numPhases) + p][idx];
                            }

                            psi *= calcInterp5thDiff(phi, p, phase, idx, simDomain->numPhases);

                            dfdphi[phase][idx] += psi / simParams->molarVolume;
                        }

                        dfdphi[phase][idx] += calcDoubleWellDerivative(phi, phase,
                                                                        simParams->theta_i_dev, simParams->theta_ij_dev, simParams->theta_ijk_dev,
                                                                        idx, simDomain->numPhases);
                    }
                }
            }
        }
    }
    else if (simControls->FUNCTION_F == 2)
    {
#pragma acc parallel loop collapse(3) present(phi[:][:], comp[:][:], dfdphi[:][:], phaseComp[:][:], mu[:][:], \
                                            simParams->theta_i_dev[:], simParams->theta_ij_dev[:], simParams->theta_ijk_dev[:], \
                                            simDomain->thermo_phase_dev[:])
        for (long i = 0; i < subdomain->sizeX; ++i)
        {
            for (long j = 0; j < ((simDomain->DIMENSION >= 2) ? subdomain->sizeY : 1); ++j)
            {
                for (long k = 0; k < ((simDomain->DIMENSION == 3) ? subdomain->sizeZ : 1); ++k)
                {
                    long idx = i * subdomain->xStep + j * subdomain->yStep + k;

                    double y[MAX_NUM_COMP];
                    double phaseEnergy = 0.0;
                    double sum = 0.0;
                    double psi = 0.0;

                    double Bpq_hphi[MAX_NUM_PHASES];

                    for (long p = 0; p < simDomain->numPhases; ++p)
                    {
                        Bpq_hphi[p] = dfdphi[p][idx];
                    }

                    for (long phase = 0; phase < simDomain->numPhases; ++phase)
                    {
                        if (simControls->ELASTICITY)
                        {
                            dfdphi[phase][idx] = 0.0;

                            for (long p = 0; p < simDomain->numPhases; ++p)
                            {
                                dfdphi[phase][idx] += calcInterp5thDiff(phi, phase, p, idx, simDomain->numPhases) * Bpq_hphi[p];
                            }
                        }

                        for (long p = 0; p < simDomain->numPhases; ++p)
                        {
                            sum = 0.0;

                            for (long is = 0; is < NUMCOMPONENTS-1; is++)
                            {
                                y[is] = phaseComp[is*NUMPHASES + p][idx];
                                sum += y[is];
                            }

                            y[simDomain->numComponents - 1] = 1.0 - sum;

                            (*free_energy_tdb_dev[thermo_phase[p]])(simParams->T, y, &phaseEnergy);

                            psi = 0.0;

                            psi += phaseEnergy;

                            for (long component = 0; component < simDomain->numComponents - 1; ++component)
                            {
                                psi -= mu[component][idx] * phaseComp[(component * simDomain->numPhases + p)][idx];
                            }

                            psi *= calcInterp5thDiff(phi, p, phase, idx, simDomain->numPhases);
                            dfdphi[phase][idx] += psi / simParams->molarVolume;
                        }

                        dfdphi[phase][idx] += calcDoubleWellDerivative(phi, phase,
                                                                        simParams->theta_i_dev, simParams->theta_ij_dev, simParams->theta_ijk_dev,
                                                                        idx, simDomain->numPhases);
                    }
                }
            }
        }
    }
}
```

#pragma acc kernels
void __computeDrivingForce_02__(double **phi, double **comp,
                                double **dfdphi, double **phaseComp,
                                double **mu,
                                double molarVolume,
                                double *theta_i, double *theta_ij, double *theta_ijk,
                                int ELASTICITY,
                                double temperature, long *thermo_phase,
                                long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                                long sizeX, long sizeY, long sizeZ,
                                long xStep, long yStep, long padding)
{
    /*
     * Get thread coordinates
     */
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    long j = threadIdx.y + blockIdx.y * blockDim.y;
    long k = threadIdx.z + blockIdx.z * blockDim.z;

    long idx = i * xStep + j * yStep + k;

    if (i < sizeX && ((j < sizeY && DIMENSION >= 2) || (DIMENSION == 1 && j == 0)) && ((k < sizeZ && DIMENSION == 3) || (DIMENSION < 3 && k == 0)))
    {
        /*
         * Calculate grand potential density for every phase
         */
        double y[MAX_NUM_COMP];
        double phaseEnergy = 0.0;
        double sum = 0.0;
        double psi = 0.0;

        double Bpq_hphi[MAX_NUM_PHASES];

        for (long p = 0; p < NUMPHASES; p++)
        {
            Bpq_hphi[p] = dfdphi[p][idx];
        }

#pragma acc parallel loop present(phaseComp[:][:], mu[:][:], dfdphi[:][:], thermo_phase[:])
        for (long phase = 0; phase < NUMPHASES; phase++)
        {
            if (ELASTICITY)
            {
                dfdphi[phase][idx] = 0.0;

                for (long p = 0; p < NUMPHASES; p++)
                {
                    dfdphi[phase][idx] += calcInterp5thDiff(phi, phase, p, idx, NUMPHASES) * Bpq_hphi[p];
                }
            }

            for (long p = 0; p < NUMPHASES; p++)
            {
                sum = 0.0;

                for (long is = 0; is < NUMCOMPONENTS - 1; is++)
                {
                    y[is] = phaseComp[is * NUMPHASES + p][idx];
                    sum += y[is];
                }

                y[NUMCOMPONENTS - 1] = 1.0 - sum;

                (*free_energy_tdb_dev[thermo_phase[p]])(temperature, y, &phaseEnergy);

                psi = 0.0;

                psi += phaseEnergy;

                for (long component = 0; component < NUMCOMPONENTS - 1; component++)
                {
                    psi -= mu[component][idx] * phaseComp[(component * NUMPHASES + p)][idx];
                }

                psi *= calcInterp5thDiff(phi, p, phase, idx, NUMPHASES);
                dfdphi[phase][idx] += psi / molarVolume;
            }

            dfdphi[phase][idx] += calcDoubleWellDerivative(phi, phase,
                                                            theta_i, theta_ij, theta_ijk,
                                                            idx, NUMPHASES);
        }
    }
}
Here's the continuation of the conversion for the `computeDrivingForce_Chemical` function:

```cpp
void computeDrivingForce_Chemical(double **phi, double **comp,
                                  double **dfdphi,
                                  double **phaseComp, double **mu,
                                  domainInfo* simDomain, controls* simControls,
                                  simParameters* simParams, subdomainInfo* subdomain,
                                  dim3 gridSize, dim3 blockSize)
{
    if (simControls->FUNCTION_F == 1 || simControls->FUNCTION_F == 3 || simControls->FUNCTION_F == 4)
    {
#pragma acc parallel loop collapse(3) present(phi[:][:], comp[:][:], dfdphi[:][:], phaseComp[:][:], \
                                            simParams->F0_A_dev[:], simParams->F0_B_dev[:], simParams->F0_C_dev[:], \
                                            simParams->theta_i_dev[:], simParams->theta_ij_dev[:], simParams->theta_ijk_dev[:])
        for (long i = subdomain->padding; i < subdomain->sizeX - subdomain->padding; ++i)
        {
            for (long j = (simDomain->DIMENSION == 1) ? 0 : subdomain->padding; j < ((simDomain->DIMENSION >= 2) ? subdomain->sizeY - subdomain->padding : 1); ++j)
            {
                for (long k = (simDomain->DIMENSION < 3) ? 0 : subdomain->padding; k < ((simDomain->DIMENSION == 3) ? subdomain->sizeZ - subdomain->padding : 1); ++k)
                {
                    long idx = i * subdomain->xStep + j * subdomain->yStep + k;

                    double psi = 0.0;

                    double Bpq_hphi[MAX_NUM_PHASES];

                    for (long p = 0; p < simDomain->numPhases; ++p)
                    {
                        Bpq_hphi[p] = dfdphi[p][idx];
                    }

                    for (long phase = 0; phase < simDomain->numPhases; ++phase)
                    {
                        if (simControls->ELASTICITY)
                        {
                            dfdphi[phase][idx] = 0.0;

                            for (long p = 0; p < simDomain->numPhases; ++p)
                            {
                                dfdphi[phase][idx] += calcInterp5thDiff(phi, phase, p, idx, simDomain->numPhases) * Bpq_hphi[p];
                            }
                        }

                        for (long p = 0; p < simDomain->numPhases; ++p)
                        {
                            double phaseEnergy = calcPhaseEnergy(phaseComp, p, simParams->F0_A_dev, simParams->F0_B_dev, simParams->F0_C_dev, idx, simDomain->numPhases, simDomain->numComponents);
                            double diffusionPotential = 0.0;

                            for (long component = 0; component < simDomain->numComponents - 1; ++component)
                            {
                                diffusionPotential -= calcDiffusionPotential(phaseComp, p, component, simParams->F0_A_dev, simParams->F0_B_dev, idx, simDomain->numPhases, simDomain->numComponents) * phaseComp[(component * simDomain->numPhases) + p][idx];
                            }

                            psi = phaseEnergy + diffusionPotential;

                            psi *= calcInterp5thDiff(phi, p, phase, idx, simDomain->numPhases);

                            dfdphi[phase][idx] += psi / simParams->molarVolume;
                        }

                        dfdphi[phase][idx] += calcDoubleWellDerivative(phi, phase,
                                                                        simParams->theta_i_dev, simParams->theta_ij_dev, simParams->theta_ijk_dev,
                                                                        idx, simDomain->numPhases);
                    }
                }
            }
        }
    }
    else if (simControls->FUNCTION_F == 2)
    {
#pragma acc parallel loop collapse(3) present(phi[:][:], comp[:][:], dfdphi[:][:], phaseComp[:][:], mu[:][:], \
                                            simParams->theta_i_dev[:], simParams->theta_ij_dev[:], simParams->theta_ijk_dev[:], \
                                            simDomain->thermo_phase_dev[:])
        for (long i = 0; i < subdomain->sizeX; ++i)
        {
            for (long j = 0; j < ((simDomain->DIMENSION >= 2) ? subdomain->sizeY : 1); ++j)
            {
                for (long k = 0; k < ((simDomain->DIMENSION == 3) ? subdomain->sizeZ : 1); ++k)
                {
                    long idx = i * subdomain->xStep + j * subdomain->yStep + k;

                    double y[MAX_NUM_COMP];
                    double phaseEnergy = 0.0;
                    double sum = 0.0;
                    double psi = 0.0;

                    double Bpq_hphi[MAX_NUM_PHASES];

                                       for (long p = 0; p < simDomain->numPhases; ++p)
                    {
                        Bpq_hphi[p] = dfdphi[p][idx];
                    }

                    for (long phase = 0; phase < simDomain->numPhases; ++phase)
                    {
                        if (simControls->ELASTICITY)
                        {
                            dfdphi[phase][idx] = 0.0;

                            for (long p = 0; p < simDomain->numPhases; ++p)
                            {
                                dfdphi[phase][idx] += calcInterp5thDiff(phi, phase, p, idx, simDomain->numPhases) * Bpq_hphi[p];
                            }
                        }

                        for (long p = 0; p < simDomain->numPhases; ++p)
                        {
                            sum = 0.0;

                            for (long is = 0; is < simDomain->numComponents - 1; ++is)
                            {
                                y[is] = phaseComp[is * simDomain->numPhases + p][idx];
                                sum += y[is];
                            }

                            y[simDomain->numComponents - 1] = 1.0 - sum;

                            (*free_energy_tdb_dev[simDomain->thermo_phase_dev[p]])(simParams->T, y, &phaseEnergy);

                            psi = 0.0;

                            psi += phaseEnergy;
                            for (long component = 0; component < simDomain->numComponents - 1; ++component)
                            {
                                psi -= mu[component][idx] * phaseComp[(component * simDomain->numPhases + p)][idx];
                            }

                            psi *= calcInterp5thDiff(phi, p, phase, idx, simDomain->numPhases);
                            dfdphi[phase][idx] += psi / simParams->molarVolume;
                        }

                        dfdphi[phase][idx] += calcDoubleWellDerivative(phi, phase,
                                                                        simParams->theta_i_dev, simParams->theta_ij_dev, simParams->theta_ijk_dev,
                                                                        idx, simDomain->numPhases);
                    }
                }
            }
        }
    }
}

