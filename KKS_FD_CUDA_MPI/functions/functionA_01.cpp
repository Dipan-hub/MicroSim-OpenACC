#include "functionA_01.h"

#define phi(phase, x, y, z) (phi[(((phase)*3 + (x))*3 + (y))*3 + (z)])

#pragma acc routine seq
double calcAnisotropy_01(double phi[MAX_NUM_PHASES*27],
                         double *dab, double *eps_ab,
                         double *Rotation_matrix, double *Inv_rotation_matrix,
                         long phase, long NUMPHASES, long DIMENSION,
                         double DELTA_X, double DELTA_Y, double DELTA_Z)
{
    /*
     * Phi Stencil
     *
     * 0 -> x-1 or y-1 or z-1
     * 1 -> x   or y   or z
     * 2 -> x+1 or y+1 or z+1
     *
     */

    /*
     * Stencil
     *
     * 0 -> centre  (i,     j,     k    )
     * 1 -> left    (i-1/2, j,     k    )
     * 2 -> right   (i+1/2, j,     k    )
     * 3 -> top     (i,     j-1/2, k    )
     * 4 -> bottom  (i,     j+1/2, k    )
     * 5 -> front   (i,     j,     k+1/2)
     * 6 -> back    (i,     j,     k-1/2)
     *
     */

    double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

    long i;
    long ip1;
    long maxPos = 3;
    if (DIMENSION == 2)
        maxPos = 5;
    else if (DIMENSION == 3)
        maxPos = 7;

    // [alpha/beta][stencil location][direction]
    double gradPhiMid[2][7][3] = {0.0};

    // [alpha/beta][stencil location]
    double phiMid[2][7] = {0.0};

    // [stencil location][direction]
    double qMid[7][3] = {0.0};

    // [direction]
    double dqdphi[3] = {0.0};

    double Rotated_vector[3] = {0.0};

    // [stencil location]
    double ac[7] = {0.0}, qab2[7] = {0.0};

    // [stencil location][direction]
    double dadq[7][3] = {0.0}, Rotated_dadq[7][3] = {0.0};

    /*
     *  Find phiMid and gradPhiMid for the phase being solved (alpha)
     */

//  #pragma acc data copyin(phi[:MAX_NUM_PHASES*27], dab[:], eps_ab[:], Rotation_matrix[:9], Inv_rotation_matrix[:9]) \
  //                    copy(gradPhiMid, phiMid, qMid, dqdphi, Rotated_vector, ac, qab2, dadq, Rotated_dadq)
  //  {

    phiMid[0][centre] = phi(phase, 1, 1, 1);

    phiMid[0][left]   = (phi(phase, 0, 1, 1) + phi(phase, 1, 1, 1))/2.0;
    phiMid[0][right]  = (phi(phase, 2, 1, 1) + phi(phase, 1, 1, 1))/2.0;

    if (DIMENSION == 2)
    {
        phiMid[0][top]    = (phi(phase, 1, 0, 1) + phi(phase, 1, 1, 1))/2.0;
        phiMid[0][bottom] = (phi(phase, 1, 2, 1) + phi(phase, 1, 1, 1))/2.0;
    }
    else if (DIMENSION == 3)
    {
        phiMid[0][top]    = (phi(phase, 1, 0, 1) + phi(phase, 1, 1, 1))/2.0;
        phiMid[0][bottom] = (phi(phase, 1, 2, 1) + phi(phase, 1, 1, 1))/2.0;

        phiMid[0][front]  = (phi(phase, 1, 1, 0) + phi(phase, 1, 1, 1))/2.0;
        phiMid[0][back]   = (phi(phase, 1, 1, 2) + phi(phase, 1, 1, 1))/2.0;
    }

    /*
     * First index: 0, denoting phase alpha
     * Second index: Location in stencil
     * Third index: Direction
     */

    // (i+1, j, k) - (i-1, j, k)
    gradPhiMid[0][centre][ix] = (phi(phase, 2, 1, 1) - phi(phase, 0, 1, 1))/(2.0*DELTA_X);

    // (i, j, k) - (i-1, j, k)
    gradPhiMid[0][left][ix]   = (phi(phase, 1, 1, 1) - phi(phase, 0, 1, 1))/(DELTA_X);
    // (i+1, j, k) - (i, j, k)
    gradPhiMid[0][right][ix]  = (phi(phase, 2, 1, 1) - phi(phase, 1, 1, 1))/(DELTA_X);

    if (DIMENSION == 2)
    {
        // (i+1, j-1, k) - (i-1, j-1, k)
        gradPhiMid[0][top][ix]    = ((phi(phase, 2, 0, 1) - phi(phase, 0, 0, 1))/(2.0*DELTA_X) + gradPhiMid[0][centre][ix])/2.0;
        // (i+1, j+1, k) - (i-1, j+1, k)
        gradPhiMid[0][bottom][ix] = ((phi(phase, 2, 2, 1) - phi(phase, 0, 2, 1))/(2.0*DELTA_X) + gradPhiMid[0][centre][ix])/2.0;

        // (i, j+1, k) - (i, j-1, k)
        gradPhiMid[0][centre][iy] = (phi(phase, 1, 2, 1) - phi(phase, 1, 0, 1))/(2.0*DELTA_Y);

        // (i-1, j+1, k) - (i-1, j-1, k)
        gradPhiMid[0][left][iy]   = ((phi(phase, 0, 2, 1) - phi(phase, 0, 0, 1))/(2.0*DELTA_Y) + gradPhiMid[0][centre][iy])/2.0;
        // (i+1, j+1, k) - (i+1, j-1, k)
        gradPhiMid[0][right][iy]  = ((phi(phase, 2, 2, 1) - phi(phase, 2, 0, 1))/(2.0*DELTA_Y) + gradPhiMid[0][centre][iy])/2.0;

        // (i, j, k) - (i, j-1, k)
        gradPhiMid[0][top][iy]    = (phi(phase, 1, 1, 1) - phi(phase, 1, 0, 1))/(DELTA_Y);
        // (i, j+1, k) - (i, j, k)
        gradPhiMid[0][bottom][iy] = (phi(phase, 1, 2, 1) - phi(phase, 1, 1, 1))/(DELTA_Y);
    }

    else if (DIMENSION == 3)
    {
        // (i+1, j-1, k) - (i-1, j-1, k)
        gradPhiMid[0][top][ix]    = ((phi(phase, 2, 0, 1) - phi(phase, 0, 0, 1))/(2.0*DELTA_X) + gradPhiMid[0][centre][ix])/2.0;
        // (i+1, j+1, k) - (i-1, j+1, k)
        gradPhiMid[0][bottom][ix] = ((phi(phase, 2, 2, 1) - phi(phase, 0, 2, 1))/(2.0*DELTA_X) + gradPhiMid[0][centre][ix])/2.0;

        // (i+1, j, k-1) - (i-1, j, k-1)
        gradPhiMid[0][front][ix]  = ((phi(phase, 2, 1, 0) - phi(phase, 0, 1, 0))/(2.0*DELTA_X) + gradPhiMid[0][centre][ix])/2.0;
        // (i+1, j, k+1) - (i-1, j, k+1)
        gradPhiMid[0][back][ix]   = ((phi(phase, 2, 1, 2) - phi(phase, 0, 1, 2))/(2.0*DELTA_X) + gradPhiMid[0][centre][ix])/2.0;

        // (i, j+1, k) - (i, j-1, k)
        gradPhiMid[0][centre][iy] = (phi(phase, 1, 2, 1) - phi(phase, 1, 0, 1))/(2.0*DELTA_Y);

        // (i-1, j+1, k) - (i-1, j-1, k)
        gradPhiMid[0][left][iy]   = ((phi(phase, 0, 2, 1) - phi(phase, 0, 0, 1))/(2.0*DELTA_Y) + gradPhiMid[0][centre][iy])/2.0;
        // (i+1, j+1, k) - (i+1, j-1, k)
        gradPhiMid[0][right][iy]  = ((phi(phase, 2, 2, 1) - phi(phase, 2, 0, 1))/(2.0*DELTA_Y) + gradPhiMid[0][centre][iy])/2.0;

        // (i, j, k) - (i, j-1, k)
        gradPhiMid[0][top][iy]    = (phi(phase, 1, 1, 1) - phi(phase, 1, 0, 1))/(DELTA_Y);
        // (i, j+1, k) - (i, j, k)
        gradPhiMid[0][bottom][iy] = (phi(phase, 1, 2, 1) - phi(phase, 1, 1, 1))/(DELTA_Y);

        // (i, j+1, k-1) - (i, j-1, k-1)
        gradPhiMid[0][front][iy]  = ((phi(phase, 1, 2, 0) - phi(phase, 1, 0, 0))/(2.0*DELTA_Y) + gradPhiMid[0][centre][iy])/2.0;
        // (i, j+1, k+1) - (i, j-1, k+1)
        gradPhiMid[0][back][iy]   = ((phi(phase, 1, 2, 2) - phi(phase, 1, 0, 2))/(2.0*DELTA_Y) + gradPhiMid[0][centre][iy])/2.0;

        // (i, j, k+1) - (i, j, k-1)
        gradPhiMid[0][centre][iz] = (phi(phase, 1, 1, 2) - phi(phase, 1, 1, 0))/(2.0*DELTA_Z);

        // (i-1, j, k+1) - (i-1, j, k-1)
        gradPhiMid[0][left][iz]   = ((phi(phase, 0, 1, 2) - phi(phase, 0, 1, 0))/(2.0*DELTA_Z) + gradPhiMid[0][centre][iz])/2.0;
        // (i+1, j, k+1) - (i+1, j, k-1)
        gradPhiMid[0][right][iz]  = ((phi(phase, 2, 1, 2) - phi(phase, 2, 1, 0))/(2.0*DELTA_Z) + gradPhiMid[0][centre][iz])/2.0;

        // (i, j-1, k+1) - (i, j-1, k-1)
        gradPhiMid[0][top][iz]    = ((phi(phase, 1, 0, 2) - phi(phase, 1, 0, 0))/(2.0*DELTA_Z) + gradPhiMid[0][centre][iz])/2.0;
        // (i, j+1, k+1) - (i, j+1, k-1)
        gradPhiMid[0][bottom][iz] = ((phi(phase, 1, 2, 2) - phi(phase, 1, 2, 0))/(2.0*DELTA_Z) + gradPhiMid[0][centre][iz])/2.0;

        // (i, j, k) - (i, j, k-1)
        gradPhiMid[0][front][iz]  = (phi(phase, 1, 1, 1) - phi(phase, 1, 1, 0))/(DELTA_Z);
        // (i, j, k+1) - (i, j, k)
        gradPhiMid[0][back][iz]   = (phi(phase, 1, 1, 2) - phi(phase, 1, 1, 1))/(DELTA_Z);
    }

    for (ip1 = 0; ip1 < NUMPHASES; ip1++)
    {
        if (ip1 == phase)
            continue;

        /*
         *  Find phiMid and gradPhiMid for phase beta
         */

        phiMid[1][centre] = phi(ip1, 1, 1, 1);

        phiMid[1][left]   = (phi(ip1, 1, 1, 1) + phi(ip1, 0, 1, 1))/2.0;
        phiMid[1][right]  = (phi(ip1, 2, 1, 1) + phi(ip1, 1, 1, 1))/2.0;

        if (DIMENSION == 2)
        {
            phiMid[1][top]    = (phi(ip1, 1, 0, 1) + phi(ip1, 1, 1, 1))/2.0;
            phiMid[1][bottom] = (phi(ip1, 1, 2, 1) + phi(ip1, 1, 1, 1))/2.0;
        }
        else if (DIMENSION == 3)
        {
            phiMid[1][top]    = (phi(ip1, 1, 0, 1) + phi(ip1, 1, 1, 1))/2.0;
            phiMid[1][bottom] = (phi(ip1, 1, 2, 1) + phi(ip1, 1, 1, 1))/2.0;

            phiMid[1][front]  = (phi(ip1, 1, 1, 0) + phi(ip1, 1, 1, 1))/2.0;
            phiMid[1][back]   = (phi(ip1, 1, 1, 2) + phi(ip1, 1, 1, 1))/2.0;
        }

        /*
         * First index: 1, denoting phase beta
         * Second index: Location in stencil
         * Third index: Direction
         */

        gradPhiMid[1][centre][ix] = (phi(ip1, 2, 1, 1) - phi(ip1, 0, 1, 1))/(2.0*DELTA_X);
        gradPhiMid[1][left][ix]   = (phi(ip1, 1, 1, 1) - phi(ip1, 0, 1, 1))/(DELTA_X);
        gradPhiMid[1][right][ix]  = (phi(ip1, 2, 1, 1) - phi(ip1, 1, 1, 1))/(DELTA_X);

        if (DIMENSION == 2)
        {
            gradPhiMid[1][top][ix]    = ((phi(ip1, 2, 0, 1) - phi(ip1, 0, 0, 1))/(2.0*DELTA_X) + gradPhiMid[1][centre][ix])/2.0;
            gradPhiMid[1][bottom][ix] = ((phi(ip1, 2, 2, 1) - phi(ip1, 0, 2, 1))/(2.0*DELTA_X) + gradPhiMid[1][centre][ix])/2.0;

            gradPhiMid[1][centre][iy] = (phi(ip1, 1, 2, 1) - phi(ip1, 1, 0, 1))/(2.0*DELTA_Y);
            gradPhiMid[1][left][iy]   = ((phi(ip1, 0, 2, 1) - phi(ip1, 0, 0, 1))/(2.0*DELTA_Y) + gradPhiMid[1][centre][iy])/2.0;
            gradPhiMid[1][right][iy]  = ((phi(ip1, 2, 2, 1) - phi(ip1, 2, 0, 1))/(2.0*DELTA_Y) + gradPhiMid[1][centre][iy])/2.0;
            gradPhiMid[1][top][iy]    = (phi(ip1, 1, 1, 1) - phi(ip1, 1, 0, 1))/(DELTA_Y);
            gradPhiMid[1][bottom][iy] = (phi(ip1, 1, 2, 1) - phi(ip1, 1, 1, 1))/(DELTA_Y);
        }
        else if (DIMENSION == 3)
        {
            gradPhiMid[1][top][ix]    = ((phi(ip1, 2, 0, 1) - phi(ip1, 0, 0, 1))/(2.0*DELTA_X) + gradPhiMid[1][centre][ix])/2.0;
            gradPhiMid[1][bottom][ix] = ((phi(ip1, 2, 2, 1) - phi(ip1, 0, 2, 1))/(2.0*DELTA_X) + gradPhiMid[1][centre][ix])/2.0;
            gradPhiMid[1][front][ix]  = ((phi(ip1, 2, 1, 0) - phi(ip1, 0, 1, 0))/(2.0*DELTA_X) + gradPhiMid[1][centre][ix])/2.0;
            gradPhiMid[1][back][ix]   = ((phi(ip1, 2, 1, 2) - phi(ip1, 0, 1, 2))/(2.0*DELTA_X) + gradPhiMid[1][centre][ix])/2.0;

            gradPhiMid[1][centre][iy] = (phi(ip1, 1, 2, 1) - phi(ip1, 1, 0, 1))/(2.0*DELTA_Y);
            gradPhiMid[1][left][iy]   = ((phi(ip1, 0, 2, 1) - phi(ip1, 0, 0, 1))/(2.0*DELTA_Y) + gradPhiMid[1][centre][iy])/2.0;
            gradPhiMid[1][right][iy]  = ((phi(ip1, 2, 2, 1) - phi(ip1, 2, 0, 1))/(2.0*DELTA_Y) + gradPhiMid[1][centre][iy])/2.0;
            gradPhiMid[1][top][iy]    = (phi(ip1, 1, 1, 1) - phi(ip1, 1, 0, 1))/(DELTA_Y);
            gradPhiMid[1][bottom][iy] = (phi(ip1, 1, 2, 1) - phi(ip1, 1, 1, 1))/(DELTA_Y);
            gradPhiMid[1][front][iy]  = ((phi(ip1, 1, 2, 0) - phi(ip1, 1, 0, 0))/(2.0*DELTA_Y) + gradPhiMid[1][centre][iy])/2.0;
            gradPhiMid[1][back][iy]   = ((phi(ip1, 1, 2, 2) - phi(ip1, 1, 0, 2))/(2.0*DELTA_Y) + gradPhiMid[1][centre][iy])/2.0;

            gradPhiMid[1][centre][iz] = (phi(ip1, 1, 1, 2) - phi(ip1, 1, 1, 0))/(2.0*DELTA_Z);
            gradPhiMid[1][left][iz]   = ((phi(ip1, 0, 1, 2) - phi(ip1, 0, 1, 0))/(2.0*DELTA_Z) + gradPhiMid[1][centre][iz])/2.0;
            gradPhiMid[1][right][iz]  = ((phi(ip1, 2, 1, 2) - phi(ip1, 2, 1, 0))/(2.0*DELTA_Z) + gradPhiMid[1][centre][iz])/2.0;
            gradPhiMid[1][top][iz]    = ((phi(ip1, 1, 0, 2) - phi(ip1, 1, 0, 0))/(2.0*DELTA_Z) + gradPhiMid[1][centre][iz])/2.0;
            gradPhiMid[1][bottom][iz] = ((phi(ip1, 1, 2, 2) - phi(ip1, 1, 2, 0))/(2.0*DELTA_Z) + gradPhiMid[1][centre][iz])/2.0;
            gradPhiMid[1][front][iz]  = (phi(ip1, 1, 1, 1) - phi(ip1, 1, 1, 0))/(DELTA_Z);
            gradPhiMid[1][back][iz]   = (phi(ip1, 1, 1, 2) - phi(ip1, 1, 1, 1))/(DELTA_Z);
        }

        /*
         * q_x_i-1/2, q_x_i+1/2, q_x_j-1/2, q_x_j+1/2, q_x_k-1/2, q_x_k+1/2, ... , q_z_k+1/2
         *
         * q^{\alpha\beta}_{x}_{i-1/2,j,k} = \phi_{\alpha}_{i-1/2,j,k}\nabla\phi_{\beta}_{i-1/2,j,k} - \phi_{\beta}_{i-1/2,j,k}\nabla\phi_{\alpha}_{i-1/2,j,k}
         *
         */
        #pragma acc parallel loop
        for (i = 0; i < maxPos; i++)
        {
            qMid[i][ix] = phiMid[0][i]*gradPhiMid[1][i][ix] - phiMid[1][i]*gradPhiMid[0][i][ix];
            if (DIMENSION == 2)
            {
                qMid[i][iy] = phiMid[0][i]*gradPhiMid[1][i][iy] - phiMid[1][i]*gradPhiMid[0][i][iy];
            }
            else if (DIMENSION == 3)
            {
                qMid[i][iy] = phiMid[0][i]*gradPhiMid[1][i][iy] - phiMid[1][i]*gradPhiMid[0][i][iy];
                qMid[i][iz] = phiMid[0][i]*gradPhiMid[1][i][iz] - phiMid[1][i]*gradPhiMid[0][i][iz];
            }

            multiply(Rotation_matrix, qMid[i], Rotated_vector, phase, ip1, NUMPHASES, DIMENSION);

            qMid[i][ix] = Rotated_vector[ix];
            if (DIMENSION == 2)
            {
                qMid[i][iy] = Rotated_vector[iy];
            }
            else if (DIMENSION == 3)
            {
                qMid[i][iy] = Rotated_vector[iy];
                qMid[i][iz] = Rotated_vector[iz];
            }
        }

        /*
         * Get a_{c}(q_{ab}) and da_{c}/dq
         */
        #pragma acc parallel loop
        for (i = 0; i < maxPos; i++)
        {
            ac[i] = anisotropy_01_function_ac(qMid[i], phase, ip1, dab, NUMPHASES);
            anisotropy_01_dAdq(qMid[i], dadq[i], phase, ip1, dab, NUMPHASES);
            multiply(Inv_rotation_matrix, dadq[i], Rotated_dadq[i], phase, ip1, NUMPHASES, DIMENSION);
        }

        dqdphi[ix] = gradPhiMid[1][centre][ix];
        if (DIMENSION == 2)
        {
            dqdphi[iy] = gradPhiMid[1][centre][iy];
        }
        else if (DIMENSION == 3)
        {
            dqdphi[iy] = gradPhiMid[1][centre][iy];
            dqdphi[iz] = gradPhiMid[1][centre][iz];
        }

        multiply(Rotation_matrix, dqdphi, Rotated_vector, phase, ip1, NUMPHASES, DIMENSION);

        dqdphi[ix] = Rotated_vector[ix];
        if (DIMENSION == 2)
        {
            dqdphi[iy] = Rotated_vector[iy];
        }
        else if (DIMENSION == 3)
        {
            dqdphi[iy] = Rotated_vector[iy];
            dqdphi[iz] = Rotated_vector[iz];
        }

        #pragma acc parallel loop reduction(+:sum1, sum2, sum3)
        for (i = 0; i < maxPos; i++)
        {
            multiply(Rotation_matrix, gradPhiMid[1][i], Rotated_vector, phase, ip1, NUMPHASES, DIMENSION);

            gradPhiMid[1][i][ix] = Rotated_vector[ix];
            if (DIMENSION == 2)
            {
                gradPhiMid[1][i][iy] = Rotated_vector[iy];
            }
            else if (DIMENSION == 3)
            {
                gradPhiMid[1][i][iy] = Rotated_vector[iy];
                gradPhiMid[1][i][iz] = Rotated_vector[iz];
            }

            multiply(Rotation_matrix, gradPhiMid[0][i], Rotated_vector, phase, ip1, NUMPHASES, DIMENSION);

            qab2[i] = (Rotated_vector[ix]*gradPhiMid[1][i][ix] + Rotated_vector[iy]*gradPhiMid[1][i][iy] + Rotated_vector[iz]*gradPhiMid[1][i][iz]);

            multiply(Inv_rotation_matrix, gradPhiMid[1][i], Rotated_vector, phase, ip1, NUMPHASES, DIMENSION);

            gradPhiMid[1][i][ix] = Rotated_vector[ix];
            if (DIMENSION == 2)
            {
                gradPhiMid[1][i][iy] = Rotated_vector[iy];
            }
            else if (DIMENSION == 3)
            {
                gradPhiMid[1][i][iy] = Rotated_vector[iy];
                gradPhiMid[1][i][iz] = Rotated_vector[iz];
            }
        }

        /*
         * sum1 ->  eps_ab * div(2ac(q) * dacdq * dqdgradphi_a * gradphi_a * gradphi_b)
         *          where dqdgradphi_a = -phi_b
         *
         */

        /*
         * (i+1/2, j, k) - (i-1/2, j, k)
         */
        sum1 = (2.0*eps_ab[phase*NUMPHASES + ip1]*ac[right]*Rotated_dadq[right][ix]*-phiMid[1][right]*qab2[right])/DELTA_X;
        sum1 -= (2.0*eps_ab[phase*NUMPHASES + ip1]*ac[left]*Rotated_dadq[left][ix]*-phiMid[1][left]*qab2[left])/DELTA_X;

        /*
         * (i, j+1/2, k) - (i, j-1/2, k)
         */
        if (DIMENSION == 2)
        {
            sum1 += (2.0*eps_ab[phase*NUMPHASES + ip1]*ac[bottom]*Rotated_dadq[bottom][iy]*-phiMid[1][bottom]*qab2[bottom])/DELTA_Y;
            sum1 -= (2.0*eps_ab[phase*NUMPHASES + ip1]*ac[top]*Rotated_dadq[top][iy]*-phiMid[1][top]*qab2[top])/DELTA_Y;
        }

        /*
         * (i, j, k+1/2) - (i, j, k-1/2)
         */
        else if (DIMENSION == 3)
        {
            sum1 += (2.0*eps_ab[phase*NUMPHASES + ip1]*ac[bottom]*Rotated_dadq[bottom][iy]*-phiMid[1][bottom]*qab2[bottom])/DELTA_Y;
            sum1 -= (2.0*eps_ab[phase*NUMPHASES + ip1]*ac[top]*Rotated_dadq[top][iy]*-phiMid[1][top]*qab2[top])/DELTA_Y;

            sum1 += (2.0*eps_ab[phase*NUMPHASES + ip1]*ac[back]*Rotated_dadq[back][iz]*-phiMid[1][back]*qab2[back])/DELTA_Z;
            sum1 -= (2.0*eps_ab[phase*NUMPHASES + ip1]*ac[front]*Rotated_dadq[front][iz]*-phiMid[1][front]*qab2[front])/DELTA_Z;
        }

        /*
         * sum2 -> eps_ab * div(ac*ac*gradphi_b)
         *
         */
        sum2 = eps_ab[phase*NUMPHASES + ip1]*ac[right]*ac[right]*gradPhiMid[1][right][ix]/DELTA_X;
        sum2 -= eps_ab[phase*NUMPHASES + ip1]*ac[left]*ac[left]*gradPhiMid[1][left][ix]/DELTA_X;

        if (DIMENSION == 2)
        {
            sum2 += eps_ab[phase*NUMPHASES + ip1]*ac[bottom]*ac[bottom]*gradPhiMid[1][bottom][iy]/DELTA_Y;
            sum2 -= eps_ab[phase*NUMPHASES + ip1]*ac[top]*ac[top]*gradPhiMid[1][top][iy]/DELTA_Y;
        }

        else if (DIMENSION == 3)
        {
            sum2 += eps_ab[phase*NUMPHASES + ip1]*ac[bottom]*ac[bottom]*gradPhiMid[1][bottom][iy]/DELTA_Y;
            sum2 -= eps_ab[phase*NUMPHASES + ip1]*ac[top]*ac[top]*gradPhiMid[1][top][iy]/DELTA_Y;

            sum2 += eps_ab[phase*NUMPHASES + ip1]*ac[back]*ac[back]*gradPhiMid[1][back][iz]/DELTA_Z;
            sum2 -= eps_ab[phase*NUMPHASES + ip1]*ac[front]*ac[front]*gradPhiMid[1][front][iz]/DELTA_Z;
        }

        /*
         * sum3 -> eps_ab * -2ac * dadq * dqdphi_a * gradphi_a * gradphi_b
         *
         *         dqdphi_a = gradphi_b
         */
        if (DIMENSION == 1)
        {
            sum3 = -2.0*eps_ab[phase*NUMPHASES + ip1]*ac[centre]
            *(dadq[centre][ix]*dqdphi[ix])
            *(qab2[centre]);
        }
        else if (DIMENSION == 2)
        {
            sum3 = -2.0*eps_ab[phase*NUMPHASES + ip1]*ac[centre]
            *(dadq[centre][ix]*dqdphi[ix] + dadq[centre][iy]*dqdphi[iy])
            *(qab2[centre]);
        }
        else if (DIMENSION == 3)
        {
            sum3 = -2.0*eps_ab[phase*NUMPHASES + ip1]*ac[centre]
            *(dadq[centre][ix]*dqdphi[ix] + dadq[centre][iy]*dqdphi[iy] + dadq[centre][iz]*dqdphi[iz])
            *(qab2[centre]);
        }
    }
//    } for #pragma

    return -1.0*(sum1 + sum2 + sum3);
}
