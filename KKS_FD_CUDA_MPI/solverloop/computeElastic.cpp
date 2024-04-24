#if ENABLE_CUFFTMP == 1

#include "box_iterator.hpp"
#include "computeElastic.cuh"

// OpenACC kernels use the pragma acc parallel for efficient parallel execution
#pragma acc kernels
void scale_output(BoxIterator<cufftDoubleComplex> begin, BoxIterator<cufftDoubleComplex> end,
                  double scalingFactor)
{
    #pragma acc parallel loop
    for (int i = 0; i < (end - begin); ++i)
    {
        begin[i] = {begin[i].x / scalingFactor, begin[i].y / scalingFactor};
    }
}

#pragma acc kernels
void setFieldZero(BoxIterator<cufftDoubleComplex> begin, BoxIterator<cufftDoubleComplex> end)
{
    #pragma acc parallel loop
    for (int i = 0; i < (end - begin); ++i)
    {
        begin[i] = {0.0, 0.0};
    }
}


#pragma acc kernels
void sumBoxIterator(BoxIterator<cufftDoubleComplex> begin, BoxIterator<cufftDoubleComplex> end, double *g_odata)
{
    double sdata[256]; // Make sure this size fits your actual block size used in CUDA

    #pragma acc parallel loop copyin(begin[0:(end-begin)]) copyout(sdata[0:256])
    for (int i = 0; i < (end - begin); ++i)
    {
        if (i < (end - begin))
            sdata[i % 256] = begin[i].x; // Simple approach, adjust for reduction
        else
            sdata[i % 256] = 0.0;
    }

    // Reduction must be handled manually in host code or using OpenACC reduction clause
    double sum = 0;
    #pragma acc parallel loop reduction(+:sum)
    for (int i = 0; i < 256; ++i)
    {
        sum += sdata[i];
    }

    *g_odata = sum; // Assuming single block operation for simplicity
}


#pragma acc kernels
void compute_hphi(double **phi, BoxIterator<cufftDoubleComplex> phiXtBegin, BoxIterator<cufftDoubleComplex> phiXtEnd,
                  long phase, long NUMPHASES, long xStep, long yStep, long sizeX, long sizeY, long sizeZ)
{
    #pragma acc parallel loop collapse(3)
    for (long z = 0; z < sizeZ; ++z)
    for (long y = 0; y < sizeY; ++y)
    for (long x = 0; x < sizeX; ++x)
    {
        const long idx = x*xStep + y*yStep + z;
        if (x != 0 && x != sizeX-1 && y != 0 && y != sizeY-1)
        {
            double phiValue = calcInterp5th(phi, phase, idx, NUMPHASES);

            long linearIndex = (x-1)*(sizeY-2)*(sizeZ-2) + (y-1)*(sizeZ-2) + (z-1);
            if (linearIndex < (phiXtEnd - phiXtBegin))
                phiXtBegin[linearIndex] = {phiValue, 0.0};
            else
                phiXtBegin[linearIndex] = {0.0, 0.0};
        }
    }
}

#pragma acc kernels
void compute_hphiAvg(BoxIterator<cufftDoubleComplex> begin, BoxIterator<cufftDoubleComplex> end, double avg)
{
    #pragma acc parallel loop
    for (int i = 0; i < (end - begin); ++i)
    {
        begin[i] = {begin[i].x - avg, 0.0};
    }
}


#pragma acc kernels
void copydfdphi(double **dfdphi, BoxIterator<cufftDoubleComplex> dfdphiXtBegin, BoxIterator<cufftDoubleComplex> dfdphiXtEnd,
                long phase, long xStep, long yStep, long sizeX, long sizeY, long sizeZ)
{
    #pragma acc parallel loop collapse(3)
    for (long z = 0; z < sizeZ; ++z)
    for (long y = 0; y < sizeY; ++y)
    for (long x = 0; x < sizeX; ++x)
    {
        if (x != 0 && x != sizeX-1 && y != 0 && y != sizeY-1)
        {
            long idx = x*xStep + y*yStep + z;
            long linearIndex = (x-1)*(sizeY-2)*(sizeZ-2) + (y-1)*(sizeZ-2) + (z-1);
            if (linearIndex < (dfdphiXtEnd - dfdphiXtBegin))
                dfdphi[phase][idx] = dfdphiXtBegin[linearIndex].x;
        }
    }
}

#pragma acc kernels
void calcdfEldphi(BoxIterator<cufftDoubleComplex> phiBegin, BoxIterator<cufftDoubleComplex> phiEnd,
                  BoxIterator<cufftDoubleComplex> dfEldphiBegin, BoxIterator<cufftDoubleComplex> dfEldphiEnd, double *Bpq)
{
    #pragma acc parallel loop
    for (int i = 0; i < (phiEnd - phiBegin); ++i)
    {
        if (i < (dfEldphiEnd - dfEldphiBegin))
        {
            dfEldphiBegin[i] = {
                dfEldphiBegin[i].x + Bpq[i] * phiBegin[i].x,
                dfEldphiBegin[i].y + Bpq[i] * phiBegin[i].y
            };
        }
    }
}

// Similar transformations should be applied to other parts of the CUDA code where required.


#pragma acc kernels
void calc_k(double *kx, double *ky, double *kz, BoxIterator<cufftDoubleComplex> phiBegin, BoxIterator<cufftDoubleComplex> phiEnd,
            double dx, double dy, double dz, long MESH_X, long MESH_Y, long MESH_Z, long numCells)
{
    #pragma acc parallel loop
    for (int tid = 0; tid < numCells; tid++)
    {
        auto phiIterator = phiBegin + tid;
        if (phiIterator < phiEnd)
        {
            long x = phiIterator.x();
            long y = phiIterator.y();
            long z = phiIterator.z();

            if (x <= MESH_X / 2)
                kx[tid] = 2.0 * M_PI * x / (MESH_X * dx);
            else
                kx[tid] = -2.0 * M_PI * (MESH_X - x) / (MESH_X * dx);

            if (y <= MESH_Y / 2)
                ky[tid] = 2.0 * M_PI * y / (MESH_Y * dy);
            else
                ky[tid] = -2.0 * M_PI * (MESH_Y - y) / (MESH_Y * dy);

            if (z <= MESH_Z / 2)
                kz[tid] = 2.0 * M_PI * z / (MESH_Z * dz);
            else
                kz[tid] = -2.0 * M_PI * (MESH_Z - z) / (MESH_Z * dz);
        }
    }
}

void calc_k(double *kx, double *ky, double *kz, cudaLibXtDesc *phiXt,
            long my_nx, long my_ny, long my_nz,
            domainInfo simDomain, controls simControls, simParameters simParams, subdomainInfo subdomain)
{
    // Assuming kx, ky, kz are already allocated and available on the host

    // Assuming we can obtain a pointer to the data from phiXt in a way compatible with OpenACC
    cufftDoubleComplex *phiData = static_cast<cufftDoubleComplex*>(phiXt->descriptor->data[0]);

    // Prepare the data region for the computation
    #pragma acc enter data copyin(phiData[0:my_nx*my_ny*my_nz])

    // Calculate wave numbers
    #pragma acc parallel loop present(phiData)
    for (int i = 0; i < my_nx*my_ny*my_nz; i++)
    {
        long x = i % simDomain.MESH_X; // Simplified calculation assuming flattened indexing
        long y = (i / simDomain.MESH_X) % simDomain.MESH_Y;
        long z = i / (simDomain.MESH_X * simDomain.MESH_Y);

        if (x <= simDomain.MESH_X / 2)
            kx[i] = 2.0 * M_PI * x / (simDomain.MESH_X * simDomain.DELTA_X);
        else
            kx[i] = -2.0 * M_PI * (simDomain.MESH_X - x) / (simDomain.MESH_X * simDomain.DELTA_X);

        if (y <= simDomain.MESH_Y / 2)
            ky[i] = 2.0 * M_PI * y / (simDomain.MESH_Y * simDomain.DELTA_Y);
        else
            ky[i] = -2.0 * M_PI * (simDomain.MESH_Y - y) / (simDomain.MESH_Y * simDomain.DELTA_Y);

        if (z <= simDomain.MESH_Z / 2)
            kz[i] = 2.0 * M_PI * z / (simDomain.MESH_Z * simDomain.DELTA_Z);
        else
            kz[i] = -2.0 * M_PI * (simDomain.MESH_Z - z) / (simDomain.MESH_Z * simDomain.DELTA_Z);
    }

    // Exit the data region
    #pragma acc exit data delete(phiData)
}


void moveTocudaLibXtDesc(double **phi, cudaLibXtDesc *phiXt[MAX_NUM_PHASES],
                         domainInfo simDomain, controls simControls, simParameters simParams, subdomainInfo subdomain,
                         MPI_Comm comm)
{
    double hphiSum = 0, hphiLocal = 0, hphiAvg = 0;

    for (long p = 0; p < simDomain.numPhases; p++)
    {
        if (simControls.eigenSwitch[p] == 0)
            continue;

        auto[phiBegin, phiEnd] = BoxIterators(CUFFT_XT_FORMAT_INPLACE, CUFFT_Z2Z,
                                              subdomain.rank, subdomain.size, simDomain.MESH_X, simDomain.MESH_Y, simDomain.MESH_Z, (cufftDoubleComplex*)phiXt[p]->descriptor->data[0]);

        #pragma acc parallel loop reduction(+:hphiSum)
        for (int i = 0; i < std::distance(phiBegin, phiEnd); i++)
        {
            compute_hphi(phi, phiBegin, phiEnd, p, simDomain.numPhases, subdomain.xStep, subdomain.yStep, subdomain.sizeX, subdomain.sizeY, subdomain.sizeZ);
            hphiSum += phiBegin[i].x; // Simplified reduction
        }

        MPI_Allreduce(&hphiLocal, &hphiAvg, 1, MPI_DOUBLE, MPI_SUM, comm);
        hphiAvg /= (double)simDomain.MESH_X * simDomain.MESH_Y * simDomain.MESH_Z;

        #pragma acc parallel loop
        for (int i = 0; i < std::distance(phiBegin, phiEnd); i++)
        {
            phiBegin[i].x -= hphiAvg; // Adjust average directly
        }
    }
}

void moveFromcudaLibXtDesc(double **dfdphi, cudaLibXtDesc *dfdphiXt[MAX_NUM_PHASES],
                           domainInfo simDomain, controls simControls, simParameters simParams, subdomainInfo subdomain)
{
    for (long p = 0; p < simDomain.numPhases; p++)
    {
        if (simControls.eigenSwitch[p] == 0)
            continue;

        auto[dfdphiBegin, dfdphiEnd] = BoxIterators(CUFFT_XT_FORMAT_INPLACE, CUFFT_Z2Z,
                                                    subdomain.rank, subdomain.size, simDomain.MESH_X, simDomain.MESH_Y, simDomain.MESH_Z, (cufftDoubleComplex*)dfdphiXt[p]->descriptor->data[0]);

        #pragma acc parallel loop
        for (int i = 0; i < std::distance(dfdphiBegin, dfdphiEnd); i++)
        {
            dfdphi[p][i] = dfdphiBegin[i].x * simDomain.MESH_X * simDomain.MESH_Y * simDomain.MESH_Z;
        }
    }
}
void computeDrivingForce_Elastic(cudaLibXtDesc *phi[MAX_NUM_PHASES], cudaLibXtDesc *dfeldphi[MAX_NUM_PHASES], double **Bpq,
                                 domainInfo simDomain, controls simControls, simParameters simParams, subdomainInfo subdomain,
                                 MPI_Comm comm)
{
    for (long p = 0; p < simDomain.numPhases; p++)
    {
        if (simControls.eigenSwitch[p] == 0)
            continue;

        auto[dfEldphiBegin, dfEldphiEnd] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_Z2Z,
                                                        subdomain.rank, subdomain.size, simDomain.MESH_X, simDomain.MESH_Y, simDomain.MESH_Z, (cufftDoubleComplex*)dfeldphi[p]->descriptor->data[0]);

        #pragma acc parallel loop
        for (int i = 0; i < std::distance(dfEldphiBegin, dfEldphiEnd); i++)
        {
            dfEldphiBegin[i] = {0.0, 0.0}; // Resetting the field to zero
        }

        for (long q = 0; q < simDomain.numPhases; q++)
        {
            if (simControls.eigenSwitch[q] == 0)
                continue;

            auto[phiBegin, phiEnd] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_Z2Z,
                                                  subdomain.rank, subdomain.size, simDomain.MESH_X, simDomain.MESH_Y, simDomain.MESH_Z, (cufftDoubleComplex*)phi[q]->descriptor->data[0]);

            #pragma acc parallel loop
            for (int j = 0; j < std::distance(phiBegin, phiEnd); j++)
            {
                dfEldphiBegin[j].x += phiBegin[j].x * Bpq[p * simDomain.numPhases + q];
                dfEldphiBegin[j].y += phiBegin[j].y * Bpq[p * simDomain.numPhases + q];
            }
        }
    }
}

#endif


