#include "updateComposition.cuh"

void updateComposition(double **phi, double **phiNew,
                       double **comp, double **compNew,
                       double **phaseComp,
                       double *F0_A, double *F0_B,
                       double *mobility, double *diffusivity, double *kappaPhi, double *theta_ij,
                       long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                       long sizeX, long sizeY, long sizeZ,
                       long xStep, long yStep, long padding, int antiTrapping,
                       double DELTA_X, double DELTA_Y, double DELTA_Z,
                       double DELTA_t)
{
    #pragma acc parallel loop collapse(3) present(phi, phiNew, comp, compNew, phaseComp, F0_A, F0_B, mobility, diffusivity, kappaPhi, theta_ij)
    for (long k = 0; k < sizeZ; k++) {
        for (long j = 0; j < sizeY; j++) {
            for (long i = 0; i < sizeX; i++) {
                if (!(i >= padding && i < sizeX-padding && ((j >= padding && j < sizeY-padding && DIMENSION >= 2) || (DIMENSION == 1 && j == 0)) && ((k >= padding && k < sizeZ-padding && DIMENSION == 3) || (DIMENSION < 3 && k == 0))))
                    continue;

                long index[3][3][3];
                double mu[7], effMobility[7], mobilityLocal;
                double J_xp = 0.0, J_xm = 0.0, J_yp = 0.0, J_ym = 0.0, J_zp = 0.0, J_zm = 0.0;
                double alpha[MAX_NUM_PHASES-1][MAX_NUM_COMP-1][7] = {0.0};
                double modgradphi[MAX_NUM_PHASES][7] = {0.0};
                double dphidt[MAX_NUM_PHASES][7] = {0.0};
                double gradx_phi[MAX_NUM_PHASES][5] = {0.0}, grady_phi[MAX_NUM_PHASES][5] = {0.0}, gradz_phi[MAX_NUM_PHASES][5] = {0.0};
                double phix[MAX_NUM_PHASES][7] = {0.0}, phiy[MAX_NUM_PHASES][7] = {0.0}, phiz[MAX_NUM_PHASES][7] = {0.0};
                double alphidot[MAX_NUM_PHASES-1][MAX_NUM_COMP-1][7] = {0.0}, scalprodct[MAX_NUM_PHASES-1][7] = {0.0};
                double jatr[MAX_NUM_COMP-1] = {0.0}, jatc[MAX_NUM_PHASES-1][MAX_NUM_COMP-1] = {0.0}, jat[MAX_NUM_PHASES-1][MAX_NUM_COMP-1][7] = {0.0};
                long component, component2, phase;
                int interface = 1;
                double divphi1 = 0.0, divphi2 = 0.0;
                long maxPos = DIMENSION == 3 ? 7 : (DIMENSION == 2 ? 5 : 3);
 if (i >= padding && i < sizeX-padding && ((j >= padding && j < sizeY-padding && DIMENSION >= 2) || (DIMENSION == 1 && j == 0)) && ((k >= padding && k < sizeZ-padding && DIMENSION == 3) || (DIMENSION < 3 && k == 0)))
    {
          for (long x = 0; x < 3; x++) {
                    for (long y = 0; y < 3; y++) {
                        for (long z = 0; z < 3; z++) {
                            index[x][y][z] = (k+z-1) + (j+y-1)*yStep + (i+x-1)*xStep;
                        }
                    }
                }
                for (long phase = 0; phase < NUMPHASES; phase++) {
                        double divphi1 = (phi[phase][index[2][1][1]] - 2.0 * phi[phase][index[1][1][1]] + phi[phase][index[0][1][1]]) / (DELTA_X * DELTA_X);
                        if (DIMENSION >= 2)
                            divphi1 += (phi[phase][index[1][2][1]] - 2.0 * phi[phase][index[1][1][1]] + phi[phase][index[1][0][1]]) / (DELTA_Y * DELTA_Y);
                        if (DIMENSION == 3)
                            divphi1 += (phi[phase][index[1][1][2]] - 2.0 * phi[phase][index[1][1][1]] + phi[phase][index[1][1][0]]) / (DELTA_Z * DELTA_Z);

                        double divphi2 = (phiNew[phase][index[2][1][1]] - 2.0 * phiNew[phase][index[1][1][1]] + phiNew[phase][index[0][1][1]]) / (DELTA_X * DELTA_X);
                        if (DIMENSION >= 2)
                            divphi2 += (phiNew[phase][index[1][2][1]] - 2.0 * phiNew[phase][index[1][1][1]] + phiNew[phase][index[1][0][1]]) / (DELTA_Y * DELTA_Y);
                        if (DIMENSION == 3)
                            divphi2 += (phiNew[phase][index[1][1][2]] - 2.0 * phiNew[phase][index[1][1][1]] + phiNew[phase][index[1][1][0]]) / (DELTA_Z * DELTA_Z);

                        if (fabs(divphi1) < 1e-3 / DELTA_X && fabs(divphi2) < 1e-3/DELTA_X)
                interface = 0;
        }
                long maxPos = (DIMENSION == 3) ? 7 : ((DIMENSION == 2) ? 5 : 3);

                    if (interface && antiTrapping) {
                        // Antitrapping Calculations
                        for (long phase = 0; phase < NUMPHASES-1; phase++) {
                            double A1 = sqrt(2.0 * kappaPhi[phase * NUMPHASES + NUMPHASES - 1] / theta_ij[phase * NUMPHASES + NUMPHASES - 1]);
                            for (long component = 0; component < NUMCOMPONENTS-1; component++) {
                                alpha[phase][component][0] = A1 * (phaseComp[component * NUMPHASES + NUMPHASES - 1][index[1][1][1]] - phaseComp[component * NUMPHASES + phase][index[1][1][1]]);
                                alpha[phase][component][1] = A1 * (phaseComp[component * NUMPHASES + NUMPHASES - 1][index[2][1][1]] - phaseComp[component * NUMPHASES + phase][index[1][1][1]]);
                                alpha[phase][component][2] = A1 * (phaseComp[component * NUMPHASES + NUMPHASES - 1][index[0][1][1]] - phaseComp[component * NUMPHASES + phase][index[0][1][1]]);

                                if (DIMENSION >= 2) {
                                    alpha[phase][component][3] = A1 * (phaseComp[component * NUMPHASES + NUMPHASES - 1][index[1][2][1]] - phaseComp[component * NUMPHASES + phase][index[1][2][1]]);
                                    alpha[phase][component][4] = A1 * (phaseComp[component * NUMPHASES + NUMPHASES - 1][index[1][0][1]] - phaseComp[component * NUMPHASES + phase][index[1][0][1]]);
                                }
                                if (DIMENSION == 3) {
                                    alpha[phase][component][3] = A1*(phaseComp[component*NUMPHASES + NUMPHASES-1][index[1][2][1]] - phaseComp[component*NUMPHASES + phase][index[1][2][1]]);
                                    alpha[phase][component][4] = A1*(phaseComp[component*NUMPHASES + NUMPHASES-1][index[1][0][1]] - phaseComp[component*NUMPHASES + phase][index[1][0][1]]);

                                    alpha[phase][component][5] = A1 * (phaseComp[component * NUMPHASES + NUMPHASES - 1][index[1][1][2]] - phaseComp[component * NUMPHASES + phase][index[1][1][2]]);
                                    alpha[phase][component][6] = A1 * (phaseComp[component * NUMPHASES + NUMPHASES - 1][index[1][1][0]] - phaseComp[component*NUMPHASES + phase][index[1][1][0]]);

    }
                }
                   dphidt[phase][0] = (phiNew[phase][index[1][1][1]] - phi[phase][index[1][1][1]])/DELTA_t;

                dphidt[phase][1] = (phiNew[phase][index[2][1][1]] - phi[phase][index[2][1][1]])/DELTA_t;
                dphidt[phase][2] = (phiNew[phase][index[0][1][1]] - phi[phase][index[0][1][1]])/DELTA_t;

                        if (DIMENSION == 2) {
                        dphidt[phase][3] = (phiNew[phase][index[1][2][1]] - phi[phase][index[1][2][1]]) / DELTA_t;
                        dphidt[phase][4] = (phiNew[phase][index[1][0][1]] - phi[phase][index[1][0][1]]) / DELTA_t;
                    } else if (DIMENSION == 3) {
                        dphidt[phase][3] = (phiNew[phase][index[1][2][1]] - phi[phase][index[1][2][1]]) / DELTA_t;
                        dphidt[phase][4] = (phiNew[phase][index[1][0][1]] - phi[phase][index[1][0][1]]) / DELTA_t;
                        dphidt[phase][5] = (phiNew[phase][index[1][1][2]] - phi[phase][index[1][1][2]]) / DELTA_t;
                        dphidt[phase][6] = (phiNew[phase][index[1][1][0]] - phi[phase][index[1][1][0]]) / DELTA_t;
                    }
                }
                    // Calculation of gradient in x, y, and z directions
                    for (long phase = 0; phase < NUMPHASES; phase++) {
                        gradx_phi[phase][0] = (phi[phase][index[2][1][1]] - phi[phase][index[0][1][1]]) / (2.0 * DELTA_X);
                        if (DIMENSION == 2) {
                            gradx_phi[phase][1] = (phi[phase][index[2][2][1]] - phi[phase][index[0][2][1]]) / (2.0 * DELTA_X);
                            gradx_phi[phase][2] = (phi[phase][index[2][0][1]] - phi[phase][index[0][0][1]]) / (2.0 * DELTA_X);
                            grady_phi[phase][0] = (phi[phase][index[1][2][1]] - phi[phase][index[1][0][1]]) / (2.0 * DELTA_Y);
                            grady_phi[phase][1] = (phi[phase][index[2][2][1]] - phi[phase][index[2][0][1]]) / (2.0 * DELTA_Y);
                            grady_phi[phase][2] = (phi[phase][index[0][2][1]] - phi[phase][index[0][0][1]]) / (2.0 * DELTA_Y);
                        }
                        if (DIMENSION == 3) {
                           dphidt[phase][3] = (phiNew[phase][index[i][j][1]] - phi[phase][index[i][j][1]]) / DELTA_t;
                           dphidt[phase][4] = (phiNew[phase][index[i][0][1]] - phi[phase][index[i][0][1]]) / DELTA_t;
                           dphidt[phase][5] = (phiNew[phase][index[i][1][j]] - phi[phase][index[i][1][j]]) / DELTA_t;
                           dphidt[phase][6] = (phiNew[phase][index[i][1][0]] - phi[phase][index[i][1][0]]) / DELTA_t;
                       }
               }
#pragma acc parallel loop present(phi, gradx_phi, grady_phi, gradz_phi, index)
for (int phase = 0; phase < NUMPHASES; ++phase)
{
    gradx_phi[phase][0] = (phi[phase][index[2][1][1]] - phi[phase][index[0][1][1]]) / (2.0 * DELTA_X);

    if (DIMENSION == 2)
    {
        gradx_phi[phase][1] = (phi[phase][index[2][2][1]] - phi[phase][index[0][2][1]]) / (2.0 * DELTA_X);
        gradx_phi[phase][2] = (phi[phase][index[2][0][1]] - phi[phase][index[0][0][1]]) / (2.0 * DELTA_X);

        grady_phi[phase][0] = (phi[phase][index[1][2][1]] - phi[phase][index[1][0][1]]) / (2.0 * DELTA_Y);
        grady_phi[phase][1] = (phi[phase][index[2][2][1]] - phi[phase][index[2][0][1]]) / (2.0 * DELTA_Y);
        grady_phi[phase][2] = (phi[phase][index[0][2][1]] - phi[phase][index[0][0][1]]) / (2.0 * DELTA_Y);
    }
    else if (DIMENSION == 3)
    {
        gradx_phi[phase][1] = (phi[phase][index[2][2][1]] - phi[phase][index[0][2][1]]) / (2.0 * DELTA_X);
        gradx_phi[phase][2] = (phi[phase][index[2][0][1]] - phi[phase][index[0][0][1]]) / (2.0 * DELTA_X);
        gradx_phi[phase][3] = (phi[phase][index[2][1][2]] - phi[phase][index[0][1][2]]) / (2.0 * DELTA_X);
        gradx_phi[phase][4] = (phi[phase][index[2][1][0]] - phi[phase][index[0][1][0]]) / (2.0 * DELTA_X);

        grady_phi[phase][0] = (phi[phase][index[1][2][1]] - phi[phase][index[1][0][1]]) / (2.0 * DELTA_Y);
        grady_phi[phase][1] = (phi[phase][index[2][2][1]] - phi[phase][index[2][0][1]]) / (2.0 * DELTA_Y);
        grady_phi[phase][2] = (phi[phase][index[0][2][1]] - phi[phase][index[0][0][1]]) / (2.0 * DELTA_Y);
        grady_phi[phase][3] = (phi[phase][index[1][2][2]] - phi[phase][index[1][0][2]]) / (2.0 * DELTA_Y);
        grady_phi[phase][4] = (phi[phase][index[1][2][0]] - phi[phase][index[1][0][0]]) / (2.0 * DELTA_Y);

        gradz_phi[phase][0] = (phi[phase][index[1][1][2]] - phi[phase][index[1][1][0]]) / (2.0 * DELTA_Z);
        gradz_phi[phase][1] = (phi[phase][index[2][1][2]] - phi[phase][index[2][1][0]]) / (2.0 * DELTA_Z);
        gradz_phi[phase][2] = (phi[phase][index[0][1][2]] - phi[phase][index[0][1][0]]) / (2.0 * DELTA_Z);
        gradz_phi[phase][3] = (phi[phase][index[1][2][2]] - phi[phase][index[1][2][0]]) / (2.0 * DELTA_Z);
        gradz_phi[phase][4] = (phi[phase][index[1][0][2]] - phi[phase][index[1][0][0]]) / (2.0 * DELTA_Z);
    }
}
         #pragma acc parallel loop present(phi, gradx_phi, grady_phi, gradz_phi, phix, phiy, phiz, alphidot, modgradphi, scalprodct, jat, diffusivity, index)
for (int phase = 0; phase < NUMPHASES; ++phase)
{
    phix[phase][0] = gradx_phi[phase][0];
    phix[phase][1] = (phi[phase][index[2][1][1]] - phi[phase][index[1][1][1]]) / (DELTA_X);
    phix[phase][2] = (phi[phase][index[1][1][1]] - phi[phase][index[0][1][1]]) / (DELTA_X);

    if (DIMENSION == 2)
    {
        phix[phase][3] = (gradx_phi[phase][0] + gradx_phi[phase][1]) / 2.0;
        phix[phase][4] = (gradx_phi[phase][0] + gradx_phi[phase][2]) / 2.0;
    }
    else if (DIMENSION == 3)
    {
        phix[phase][3] = (gradx_phi[phase][0] + gradx_phi[phase][1]) / 2.0;
        phix[phase][4] = (gradx_phi[phase][0] + gradx_phi[phase][2]) / 2.0;
        phix[phase][5] = (gradx_phi[phase][0] + gradx_phi[phase][3]) / 2.0;
        phix[phase][6] = (gradx_phi[phase][0] + gradx_phi[phase][4]) / 2.0;
    }

    if (DIMENSION >= 2)
    {
        phiy[phase][0] = grady_phi[phase][0];
        phiy[phase][1] = (grady_phi[phase][0] + grady_phi[phase][1]) / 2.0;
        phiy[phase][2] = (grady_phi[phase][0] + grady_phi[phase][2]) / 2.0;
        phiy[phase][3] = (phi[phase][index[1][2][1]] - phi[phase][index[1][1][1]]) / (DELTA_Y);
        phiy[phase][4] = (phi[phase][index[1][1][1]] - phi[phase][index[1][0][1]]) / (DELTA_Y);

        if (DIMENSION == 3)
        {
            phiy[phase][5] = (grady_phi[phase][0] + grady_phi[phase][3]) / 2.0;
            phiy[phase][6] = (grady_phi[phase][0] + grady_phi[phase][4]) / 2.0;

            phiz[phase][0] = gradz_phi[phase][0];
            phiz[phase][1] = (gradz_phi[phase][0] + gradz_phi[phase][1]) / 2.0;
            phiz[phase][2] = (gradz_phi[phase][0] + gradz_phi[phase][2]) / 2.0;
            phiz[phase][3] = (gradz_phi[phase][0] + gradz_phi[phase][3]) / 2.0;
            phiz[phase][4] = (gradz_phi[phase][0] + gradz_phi[phase][4]) / 2.0;
            phiz[phase][5] = (phi[phase][index[1][1][2]] - phi[phase][index[1][1][1]]) / (DELTA_Z);
            phiz[phase][6] = (phi[phase][index[1][1][1]] - phi[phase][index[1][1][0]]) / (DELTA_Z);
        }
    }
}

#pragma acc parallel loop present(alphidot, dphidt, phix, phiy, phiz, scalprodct, jat, diffusivity)
for (int phase = 0; phase < NUMPHASES - 1; ++phase)
{
    for (int component = 0; component < NUMCOMPONENTS - 1; ++component)
    {
        alphidot[phase][component][1] = ((alpha[phase][component][0] * dphidt[phase][0]) + (alpha[phase][component][1] * dphidt[phase][1])) / 2.0;
        alphidot[phase][component][2] = ((alpha[phase][component][0] * dphidt[phase][0]) + (alpha[phase][component][2] * dphidt[phase][2])) / 2.0;

        if (DIMENSION == 2)
        {
            alphidot[phase][component][3] = ((alpha[phase][component][0] * dphidt[phase][0]) + (alpha[phase][component][3] * dphidt[phase][3])) / 2.0;
            alphidot[phase][component][4] = ((alpha[phase][component][0] * dphidt[phase][0]) + (alpha[phase][component][4] * dphidt[phase][4])) / 2.0;
        }
        else if (DIMENSION == 3)
        {
            alphidot[phase][component][3] = ((alpha[phase][component][0] * dphidt[phase][0]) + (alpha[phase][component][3] * dphidt[phase][3])) / 2.0;
            alphidot[phase][component][4] = ((alpha[phase][component][0] * dphidt[phase][0]) + (alpha[phase][component][4] * dphidt[phase][4])) / 2.0;
            alphidot[phase][component][5] = ((alpha[phase][component][0] * dphidt[phase][0]) + (alpha[phase][component][5] * dphidt[phase][5])) / 2.0;
            alphidot[phase][component][6] = ((alpha[phase][component][0] * dphidt[phase][0]) + (alpha[phase][component][6] * dphidt[phase][6])) / 2.0;
        }
    }
}  
         #pragma acc parallel loop present(phix, phiy, phiz, modgradphi, scalprodct, alphidot, jat, diffusivity)
for (int phase = 0; phase < NUMPHASES; ++phase)
{
    for (long iter = 0; iter < maxPos; ++iter)
    {
        modgradphi[phase][iter] = phix[phase][iter] * phix[phase][iter];

        if (DIMENSION == 2)
        {
            modgradphi[phase][iter] += phiy[phase][iter] * phiy[phase][iter];
        }
        else if (DIMENSION == 3)
        {
            modgradphi[phase][iter] += phiy[phase][iter] * phiy[phase][iter];
            modgradphi[phase][iter] += phiz[phase][iter] * phiz[phase][iter];
        }

        modgradphi[phase][iter] = sqrt(modgradphi[phase][iter]);
    }
}

#pragma acc parallel loop present(phix, phiy, phiz, modgradphi, scalprodct)
for (int phase = 0; phase < NUMPHASES - 1; ++phase)
{
    for (long iter = 0; iter < maxPos; ++iter)
    {
        scalprodct[phase][iter] = -1.0 * (phix[phase][iter] * phix[NUMPHASES - 1][iter] + phiy[phase][iter] * phiy[NUMPHASES - 1][iter] + phiz[phase][iter] * phiz[NUMPHASES - 1][iter]);

        if (modgradphi[NUMPHASES - 1][iter] > 0.0)
        {
            scalprodct[phase][iter] /= (modgradphi[phase][iter] * modgradphi[NUMPHASES - 1][iter]);
        }
    }
}

#pragma acc parallel loop present(alphidot, phix, phiy, phiz, modgradphi, jat, diffusivity)
for (int phase = 0; phase < NUMPHASES - 1; ++phase)
{
    for (int component = 0; component < NUMCOMPONENTS - 1; ++component)
    {
        double diffLocal = 1.0 - diffusivity[(component + phase * (NUMCOMPONENTS - 1)) * (NUMCOMPONENTS - 1) + component] / diffusivity[(component + (NUMPHASES - 1) * (NUMCOMPONENTS - 1)) * (NUMCOMPONENTS - 1) + component];

        jat[phase][component][1] = (alphidot[phase][component][1] * phix[phase][1] / modgradphi[phase][1]) * diffLocal * fabs(scalprodct[phase][1]);
        jat[phase][component][2] = (alphidot[phase][component][2] * phix[phase][2] / modgradphi[phase][2]) * diffLocal * fabs(scalprodct[phase][2]);

        if (DIMENSION == 2)
        {
            jat[phase][component][3] = (alphidot[phase][component][3] * phiy[phase][3] / modgradphi[phase][3]) * diffLocal * fabs(scalprodct[phase][3]);
            jat[phase][component][4] = (alphidot[phase][component][4] * phiy[phase][4] / modgradphi[phase][4]) * diffLocal * fabs(scalprodct[phase][4]);
        }
        else if (DIMENSION == 3)
        {
            jat[phase][component][3] = ((alphidot[phase][component][3] * phiy[phase][3]) / (modgradphi[phase][3])) * diffLocal * fabs(scalprodct[phase][3]);
            jat[phase][component][4] = ((alphidot[phase][component][4] * phiy[phase][4]) / (modgradphi[phase][4])) * diffLocal * fabs(scalprodct[phase][4]);

            jat[phase][component][5] = ((alphidot[phase][component][5] * phiz[phase][5]) / (modgradphi[phase][5])) * diffLocal * fabs(scalprodct[phase][5]);
            jat[phase][component][6] = ((alphidot[phase][component][6] * phiz[phase][6]) / (modgradphi[phase][6])) * diffLocal * fabs(scalprodct[phase][6]);
        }
    }
}

#pragma acc parallel loop present(jat, modgradphi)
for (int phase = 0; phase < NUMPHASES - 1; ++phase)
{
    for (int component = 0; component < NUMCOMPONENTS - 1; ++component)
    {
        for (long iter = 0; iter < maxPos; ++iter)
        {
            if (modgradphi[phase][iter] == 0.0)
            {
                jat[phase][component][iter] = 0.0;
            }
        }
    }
}
            #pragma acc parallel loop present(phix, phiy, phiz, modgradphi, scalprodct, alphidot, jat, diffusivity)
for (int phase = 0; phase < NUMPHASES; ++phase)
{
    for (long iter = 0; iter < maxPos; ++iter)
    {
        modgradphi[phase][iter] = phix[phase][iter] * phix[phase][iter];

        if (DIMENSION == 2)
        {
            modgradphi[phase][iter] += phiy[phase][iter] * phiy[phase][iter];
        }
        else if (DIMENSION == 3)
        {
            modgradphi[phase][iter] += phiy[phase][iter] * phiy[phase][iter];
            modgradphi[phase][iter] += phiz[phase][iter] * phiz[phase][iter];
        }

        modgradphi[phase][iter] = sqrt(modgradphi[phase][iter]);
    }
}

#pragma acc parallel loop present(phix, phiy, phiz, modgradphi, scalprodct)
for (int phase = 0; phase < NUMPHASES - 1; ++phase)
{
    for (long iter = 0; iter < maxPos; ++iter)
    {
        scalprodct[phase][iter] = -1.0 * (phix[phase][iter] * phix[NUMPHASES - 1][iter] + phiy[phase][iter] * phiy[NUMPHASES - 1][iter] + phiz[phase][iter] * phiz[NUMPHASES - 1][iter]);

        if (modgradphi[NUMPHASES - 1][iter] > 0.0)
        {
            scalprodct[phase][iter] /= (modgradphi[phase][iter] * modgradphi[NUMPHASES - 1][iter]);
        }
    }
}

#pragma acc parallel loop present(alphidot, phix, phiy, phiz, modgradphi, jat, diffusivity)
for (int phase = 0; phase < NUMPHASES - 1; ++phase)
{
    for (int component = 0; component < NUMCOMPONENTS - 1; ++component)
    {
        double diffLocal = 1.0 - diffusivity[(component + phase * (NUMCOMPONENTS - 1)) * (NUMCOMPONENTS - 1) + component] / diffusivity[(component + (NUMPHASES - 1) * (NUMCOMPONENTS - 1)) * (NUMCOMPONENTS - 1) + component];

        jat[phase][component][1] = (alphidot[phase][component][1] * phix[phase][1] / modgradphi[phase][1]) * diffLocal * fabs(scalprodct[phase][1]);
        jat[phase][component][2] = (alphidot[phase][component][2] * phix[phase][2] / modgradphi[phase][2]) * diffLocal * fabs(scalprodct[phase][2]);

        if (DIMENSION == 2)
        {
            jat[phase][component][3] = (alphidot[phase][component][3] * phiy[phase][3] / modgradphi[phase][3]) * diffLocal * fabs(scalprodct[phase][3]);
            jat[phase][component][4] = (alphidot[phase][component][4] * phiy[phase][4] / modgradphi[phase][4]) * diffLocal * fabs(scalprodct[phase][4]);
        }
        else if (DIMENSION == 3)
        {
            jat[phase][component][3] = ((alphidot[phase][component][3] * phiy[phase][3]) / (modgradphi[phase][3])) * diffLocal * fabs(scalprodct[phase][3]);
            jat[phase][component][4] = ((alphidot[phase][component][4] * phiy[phase][4]) / (modgradphi[phase][4])) * diffLocal * fabs(scalprodct[phase][4]);

            jat[phase][component][5] = ((alphidot[phase][component][5] * phiz[phase][5]) / (modgradphi[phase][5])) * diffLocal * fabs(scalprodct[phase][5]);
            jat[phase][component][6] = ((alphidot[phase][component][6] * phiz[phase][6]) / (modgradphi[phase][6])) * diffLocal * fabs(scalprodct[phase][6]);
        }
    }
}

#pragma acc parallel loop present(jat, modgradphi)
for (int phase = 0; phase < NUMPHASES - 1; ++phase)
{
    for (int component = 0; component < NUMCOMPONENTS - 1; ++component)
    {
        for (long iter = 0; iter < maxPos; ++iter)
        {
            if (modgradphi[phase][iter] == 0.0)
            {
                jat[phase][component][iter] = 0.0;
            }
        }
    }
}

     #pragma acc parallel loop present(jat, jatc, phi, index, mobility, mu, comp, compNew)
for (int phase = 0; phase < NUMPHASES - 1; ++phase)
{
    for (int component = 0; component < NUMCOMPONENTS - 1; ++component)
    {
        jatc[phase][component] = (jat[phase][component][1] - jat[phase][component][2]) / DELTA_X;
        if (DIMENSION >= 2)
            jatc[phase][component] += (jat[phase][component][3] - jat[phase][component][4]) / DELTA_Y;
        if (DIMENSION == 3)
            jatc[phase][component] += (jat[phase][component][5] - jat[phase][component][6]) / DELTA_Z;
    }
}

double J_xp, J_xm, J_yp, J_ym, J_zp, J_zm;
#pragma acc parallel loop present(jatr, jatc, phi, index, mobility)
for (int component = 0; component < NUMCOMPONENTS - 1; ++component)
{
    
    {
        for (int phase = 0; phase < NUMPHASES - 1; ++phase)
        {
            jatr[component] += jatc[phase][component];
        }
    }
    else
    {    for (component = 0; component < NUMCOMPONENTS-1; component++)
              jatr[component] = 0.0;
    }
}

#pragma acc data copy(effMobility[0:7])
{
    #pragma acc parallel loop
    for (int component = 0; component < NUMCOMPONENTS-1; component++)
    {
        double J_xp = 0.0, J_xm = 0.0, J_yp = 0.0, J_ym = 0.0, J_zp = 0.0, J_zm = 0.0;

        for (int component2 = 0; component2 < NUMCOMPONENTS-1; component2++)
        {
            #pragma acc loop seq
            for (int k = 0; k < 7; k++)
                effMobility[k] = 0.0;

            #pragma acc loop
            for (int phase = 0; phase < NUMPHASES; phase++)
            {
                double mobilityLocal = mobility[(component2 + phase*(NUMCOMPONENTS-1))*(NUMCOMPONENTS-1) + component];
                
               #pragma acc loop seq
        for (int idx = 0; idx < ((DIMENSION == 3) ? 7 : 5); idx++)
        {
            if (phi[phase][index[indices[idx][0]][indices[idx][1]][indices[idx][2]]] > 0.999)
                effMobility[idx] = mobilityLocal;
            else if (phi[phase][index[indices[idx][0]][indices[idx][1]][indices[idx][2]]] > 0.001)
                effMobility[idx] += mobilityLocal * calcInterp5th(phi, phase, index[indices[idx][0]][indices[idx][1]][indices[idx][2]], NUMPHASES);
        }
            }
        }
    }
}      
          // Include OpenACC header


void updateComposition_02(double **phi, double **phiNew,
                          double **comp, double **compNew, double **mu,
                          double **phaseComp, long *thermo_phase,
                          double *diffusivity, double *kappaPhi, double *theta_ij,
                          double temperature, double molarVolume,
                          long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                          long sizeX, long sizeY, long sizeZ,
                          long xStep, long yStep, long padding,
                          double DELTA_X, double DELTA_Y, double DELTA_Z,
                          double DELTA_t)
{
    // Define the size of the grid
    long size = sizeX * sizeY * ((DIMENSION == 3) ? sizeZ : 1);

    // Using OpenACC parallel loop to offload the computation
    #pragma acc parallel loop copyin(phi[0:NUMPHASES][0:size], mu[0:NUMPHASES][0:size]) \
                              copyout(phiNew[0:NUMPHASES][0:size], compNew[0:NUMCOMPONENTS][0:size])
    for (long idx = 0; idx < size; idx++)
    {
        long i = idx / (sizeY * sizeZ);
        long j = (idx / sizeZ) % sizeY;
        long k = idx % sizeZ;

        double muLocal[7];
        double effMobility[7];
        double J_xp = 0.0, J_xm = 0.0, J_yp = 0.0, J_ym = 0.0, J_zp = 0.0, J_zm = 0.0;

        // Placeholder for detailed computation logic
        // Implement computations here using the indices (i, j, k)
        // You can also use OpenACC data clauses to manage memory for other arrays as needed
    }
}
         double alpha[MAX_NUM_PHASES-1][MAX_NUM_COMP-1][7];
    double modgradphi[MAX_NUM_PHASES][7];
    double dphidt[MAX_NUM_PHASES][7];
    double gradx_phi[MAX_NUM_PHASES][5];
    double grady_phi[MAX_NUM_PHASES][5];
    double gradz_phi[MAX_NUM_PHASES][5];
    double phix[MAX_NUM_PHASES][7], phiy[MAX_NUM_PHASES][7], phiz[MAX_NUM_PHASES][7];
    double alphidot[MAX_NUM_PHASES-1][MAX_NUM_COMP-1][7];
    double scalprodct[MAX_NUM_PHASES-1][7];
    double jatr[MAX_NUM_COMP-1];
    double jatc[MAX_NUM_PHASES-1][MAX_NUM_COMP-1];
    double jat[MAX_NUM_PHASES-1][MAX_NUM_COMP-1][7];
    long component, component2, phase;
    double tol = 1e-6;

   #pragma acc data copyin(phi[:NUMPHASES][:sizeX*sizeY*sizeZ], phaseComp[:NUMPHASES*NUMCOMPONENTS][:sizeX*sizeY*sizeZ], \
                            kappaPhi[:NUMPHASES], theta_ij[:NUMPHASES*NUMCOMPONENTS]) \
                     copyout(phiNew[:NUMPHASES][:sizeX*sizeY*sizeZ])
    {
        #pragma acc parallel loop collapse(3) 
        for (long i = padding; i < sizeX - padding; i++) {
            for (long j = (DIMENSION >= 2 ? padding : 0); j < (DIMENSION >= 2 ? sizeY - padding : 1); j++) {
                for (long k = (DIMENSION == 3 ? padding : 0); k < (DIMENSION == 3 ? sizeZ - padding : 1); k++) {
                    // Check the boundary conditions
                    if (i >= padding && i < sizeX-padding && ((j >= padding && j < sizeY-padding && DIMENSION >= 2) || (DIMENSION == 1 && j == 0)) &&
                        ((k >= padding && k < sizeZ-padding && DIMENSION == 3) || (DIMENSION < 3 && k == 0))) {

                        // Initialize index array
                        long index[3][3][3];
                        // Calculate the indices
                        for (long x = 0; x < 3; x++) {
                            for (long y = 0; y < 3; y++) {
                                for (long z = 0; z < 3; z++) {
                                    index[x][y][z] = (k + z - 1) + (j + y - 1) * yStep + (i + x - 1) * xStep;
                                }
                            }
                        }

                        long maxPos = (DIMENSION == 3 ? 7 : (DIMENSION == 2 ? 5 : 3));
                        // Computation for each phase and component
                        for (long phase = 0; phase < NUMPHASES-1; phase++) {
                            double A1 = sqrt(2.0 * kappaPhi[phase * NUMPHASES + NUMPHASES - 1] / theta_ij[phase * NUMPHASES + NUMPHASES - 1]);
                            for (long component = 0; component < NUMCOMPONENTS-1; component++) {
                                for (long pos = 0; pos < maxPos; pos++) {
                                    // Compute alpha
                                    alpha[phase][component][pos] = A1 * (phaseComp[component * NUMPHASES + NUMPHASES - 1][index[1][1][1]] - phaseComp[component * NUMPHASES + phase][index[1][1][1]]);
                                    // Compute delta phi over delta time
                                    dphidt[phase][pos] = (phiNew[phase][index[1][1][1]] - phi[phase][index[1][1][1]]) / DELTA_t;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
