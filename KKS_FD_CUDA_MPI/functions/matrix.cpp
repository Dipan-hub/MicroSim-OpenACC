#include "matrix.h" // Make sure the include guard is updated appropriately
#include <cmath> // Required for std::fabs and std::pow

/*
#include <cstdlib> // For dynamic memory allocation


// Helper functions for dynamic memory allocation
double** MallocM(long size) {
    double** matrix = new double*[size];
    for (long i = 0; i < size; ++i) {
        matrix[i] = new double[size];
    }
    return matrix;
}

double* MallocV(long size) {
    return new double[size];
}

void FreeM(double** matrix, long size) {
    for (long i = 0; i < size; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void free(double* vector) {
    delete[] vector;
}

*/


// Sum of two vectors
#pragma acc kernels
void vectorsum(double *y1, double *y2, double *sum, long size) {
    #pragma acc loop independent
    for (int j = 0; j < size; j++) {
        sum[j] = y1[j] + y2[j];
    }
}

// Multiplication of matrix and vector
#pragma acc kernels
void multiply(double **inv, double *y, double *prod, long size) {
    #pragma acc loop independent
    for (int i = 0; i < size; i++) {
        double sum = 0;
        #pragma acc loop reduction(+:sum)
        for (int j = 0; j < size; j++) {
            sum += inv[i][j] * y[j];
        }
        prod[i] = sum;
    }
}

// Multiplication of two matrices
#pragma acc kernels
void multiply2d(double **m1, double **m2, double **prod, long size) {
    #pragma acc loop independent
    for (int k = 0; k < size; k++) {
        #pragma acc loop independent
        for (int i = 0; i < size; i++) {
            double sum = 0;
                                                   //The reduction clause correctly handles sum calculations in both substitution functions.
            #pragma acc loop reduction(+:sum)    
            for (int j = 0; j < size; j++) {
                sum += m1[k][j] * m2[j][i];
            }
            prod[k][i] = sum;
        }
    }
}

// Matrix inversion using LU decomposition
#pragma acc kernels
void matinvnew(double **coeffmatrix, double **inv, long size) {
    int i, j, k;
    int *tag = new int[size]; // Dynamic allocation for tag
    double **factor = MallocM(size, size);
    double **iden = MallocM(size, size);
    double **inv1 = MallocM(size, size);
    double **prod = MallocM(size, size);
    double *vec1 = MallocV(size);
    double *vec = MallocV(size);
    double fact;

    // Making the Upper Triangular Matrix
    #pragma acc loop independent
    for (k = 0; k < size; k++) {
        tag[k] = k;
    }

    for (k = 0; k < size; k++) {
        pivot(coeffmatrix, factor, k, tag, size); // This needs to be adjusted if using OpenACC
        #pragma acc loop independent
        for (i = k + 1; i < size; i++) {
            fact = -coeffmatrix[i][k] / coeffmatrix[k][k];
            factor[i][k] = -fact;
            #pragma acc loop independent
            for (j = k; j <= size - 1; j++) {
                coeffmatrix[i][j] = fact * coeffmatrix[k][j] + coeffmatrix[i][j];
            }
        }
    }

    #pragma acc loop independent
    for (i = 0; i < size; i++) {
        #pragma acc loop independent
        for (j = 0; j < size; j++) {
            if (i == j) {
                factor[i][j] = 1;
            } else if (j > i) {
                factor[i][j] = 0;
            }
        }
    }

    // Identity matrix setup
    #pragma acc loop independent
    for (i = 0; i < size; i++) {
        #pragma acc loop independent
        for (j = 0; j < size; j++) {
            iden[i][j] = (i == j) ? 1 : 0;
        }
    }

    // Forward and backward substitution
    for (i = 0; i < size; i++) {
        substitutef(factor, iden, i, vec1, size); // Adjust if using OpenACC
        substituteb(coeffmatrix, vec1, vec, size); // Adjust if using OpenACC
        #pragma acc loop independent
        for (j = 0; j < size; j++) {
            inv1[j][i] = vec[j];
        }
    }

    colswap(inv1, inv, tag, size); // Adjust if using OpenACC
    multiply2d(factor, coeffmatrix, prod, size); // Already adjusted for OpenACC
    rowswap(prod, coeffmatrix, tag, size); // Adjust if using OpenACC

    FreeM(factor, size);
    FreeM(iden, size);
    FreeM(inv1, size);
    FreeM(prod, size);
    free(vec1);
    free(vec);
    delete[] tag; // Clean up dynamic allocation
}

// Back Substitution
#pragma acc kernels
void substituteb(double **fac, double *y, double *vec, long size) {
    int i, j;
    double sum;
    vec[size-1] = y[size-1] / fac[size-1][size-1];  // More readable without pow
    for (i = size-2; i >= 0; i--) {
        sum = 0.0;
        #pragma acc loop reduction(-:sum)
        for (j = i + 1; j < size; j++) {
            sum += fac[i][j] * vec[j];
        }
        vec[i] = (y[i] - sum) / fac[i][i];
    }
}

// Forward Substitution
#pragma acc kernels
void substitutef(double **fac, double **y1, int index, double *vec, long size) {
    int i, j;
    double d[size], sum;  // Variable length array (VLA) can be problematic, consider using malloc if issues arise

    #pragma acc loop independent
    for (i = 0; i < size; i++) {
        d[i] = y1[i][index];
    }
    vec[0] = d[0];
    for (i = 1; i < size; i++) {
        sum = 0.0;
        #pragma acc loop reduction(+:sum)
        for (j = 0; j < i; j++) {
            sum += fac[i][j] * vec[j];
        }
        vec[i] = d[i] + sum;
    }
}

// Modulus operator (Utilized in pivot function)
double mod(double k) {
    return (k < 0) ? -k : k;
}

// Pivoting
#pragma acc kernels
void pivot(double **coeffmatrix, double **factor, int k, int *tag, long size) {
    double swap, big;
    int tagswap, i, tag1;
    big = mod(coeffmatrix[k][k]);
    tag1 = k;
    #pragma acc loop independent
    for (i = k + 1; i < size; i++) {
        if (mod(coeffmatrix[i][k]) > big) {
            tag1 = i;
            big = coeffmatrix[i][k];
        }
    }
    tagswap = tag[k];
    tag[k] = tag[tag1];
    tag[tag1] = tagswap;

    #pragma acc loop independent
    for (i = 0; i < size; i++) {
        swap = coeffmatrix[k][i];
        coeffmatrix[k][i] = coeffmatrix[tag1][i];
        coeffmatrix[tag1][i] = swap;
    }
    #pragma acc loop independent
    for (i = 0; i < k; i++) {
        swap = factor[k][i];
        factor[k][i] = factor[tag1][i];
        factor[tag1][i] = swap;
    }
}

// Swapping Columns
#pragma acc kernels
void colswap(double **m1, double **m2, int *tag, long size) {
    int j, k, p;
    #pragma acc loop independent
    for (k = 0; k < size; k++) {
        #pragma acc loop independent
        for (j = 0; j < size; j++) {
            #pragma acc loop independent
            for (p = 0; p < size; p++) {
                m2[p][tag[j]] = m1[p][j];
            }
        }
    }
}

// Switching rows
#pragma acc kernels
void rowswap(double **m1, double **m2, int *tag, long size) {
    int j, k, p;
    #pragma acc loop independent
    for (k = 0; k < size; k++) {
        #pragma acc loop independent
        for (j = 0; j < size; j++) {
            #pragma acc loop independent
            for (p = 0; p < size; p++) {
                m2[tag[j]][p] = m1[j][p];
            }
        }
    }
}


/*
 *  LU Decomposition
 */
#pragma acc routine seq
int LUPDecompose(double **A, int N, double Tol, int *P) {
    int i, j, k, imax;
    double maxA, absA;

    #pragma acc loop independent
    for (i = 0; i <= N; i++) {
        P[i] = i;  // Unit permutation matrix, P[N] initialized with N
    }

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        #pragma acc loop independent reduction(max:maxA)
        for (k = i; k < N; k++) {
            absA = fabs(A[k][i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }

        if (maxA < Tol) return 0;  // Failure, matrix is degenerate

        if (imax != i) {
            // Pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            // Pivoting rows of A
            #pragma acc atomic update
            {
                double *temp = A[i];
                A[i] = A[imax];
                A[imax] = temp;
            }

            // Counting pivots starting from N (for determinant)
          /*For operations that modify data used across iterations, I used #pragma acc atomic update to ensure correct execution and prevent data race conditions.*/
            #pragma acc atomic update
            P[N]++;
        }

        #pragma acc loop independent
        for (j = i + 1; j < N; j++) {
            A[j][i] /= A[i][i];
            #pragma acc loop independent
            for (k = i + 1; k < N; k++) {
                A[j][k] -= A[j][i] * A[i][k];
            }
        }
    }
    return 1;  // Decomposition done
}

/*
 *  Inversion after LU-decomposition
 */
#pragma acc kernels
void LUPInvert(double **A, int *P, int N, double **IA) {
    #pragma acc loop independent
    for (int j = 0; j < N; j++) {
        #pragma acc loop independent
        for (int i = 0; i < N; i++) {
            IA[i][j] = (P[i] == j) ? 1.0 : 0.0;
            #pragma acc loop independent
            for (int k = 0; k < i; k++) {
                IA[i][j] -= A[i][k] * IA[k][j];
            }
        }
        #pragma acc loop independent
        for (int i = N - 1; i >= 0; i--) {
            #pragma acc loop independent
            for (int k = i + 1; k < N; k++) {
                IA[i][j] -= A[i][k] * IA[k][j];
            }
            IA[i][j] /= A[i][i];
        }
    }
}

/*
 *  Matrix Multiplication AB = C
 *  N is the dimension of all 3 matrices
 */
#pragma acc kernels
void matrixMultiply(double **A, double **B, double **C, int N) {
    #pragma acc loop independent
    for (int i = 0; i < N; i++) {
        #pragma acc loop independent
        for (int j = 0; j < N; j++) {
            double sum = 0.0;    //I used reduction(+:sum) for the matrix multiplication to accumulate values correctly during parallel computation.
            #pragma acc loop reduction(+:sum)
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

/*
 * Matrix multiplication for a specialized matrix structure (N*N*3*3)
 */
#pragma acc kernels
void multiply(double *A, double *B, double *C, long ip1, long ip2, long NUMPHASES, int N) {
    int i, j;
    double sum;

    #pragma acc loop independent
    for (i = 0; i < N; i++) {
        sum = 0.0;
        #pragma acc loop reduction(+:sum)
        for (j = 0; j < N; j++) {
            sum += A[((ip1 * NUMPHASES + ip2) * 3 + i) * 3 + j] * B[j];
        }
        C[i] = sum;
    }
}

/*
 * LU Decomposition for a 2D array
 */
#pragma acc routine seq
int LUPDecomposeC1(double A[][MAX_NUM_COMP], long N, double Tol, int *P) {
    long i, j, k, imax;
    double maxA, absA, ptr;

    #pragma acc loop independent
    for (i = 0; i <= N; i++) {
        P[i] = i;
    }

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        #pragma acc loop independent reduction(max:maxA)
        for (k = i; k < N; k++) {
            absA = fabs(A[k][i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }

        if (maxA < Tol) return 0;  // Matrix is degenerate

        if (imax != i) {
            // Pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            // Pivoting rows of A
            #pragma acc atomic update
            {
                for (j = 0; j < N; j++) {
                    ptr = A[i][j];
                    A[i][j] = A[imax][j];
                    A[imax][j] = ptr;
                }
            }

            P[N]++;
        }

        #pragma acc loop independent
        for (j = i + 1; j < N; j++) {
            A[j][i] /= A[i][i];
            #pragma acc loop independent
            for (k = i + 1; k < N; k++) {
                A[j][k] -= A[j][i] * A[i][k];
            }
        }
    }
    return 1;  // Decomposition successful
}

/*
 * LU Decomposition for a 1D array treated as a 2D array
 */
#pragma acc routine seq
int LUPDecomposeC2(double *A, long N, double Tol, int *P) {
    long i, j, k, imax;
    double maxA, absA, ptr;

    #pragma acc loop independent
    for (i = 0; i <= N; i++) {
        P[i] = i;
    }

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        #pragma acc loop independent reduction(max:maxA)
        for (k = i; k < N; k++) {
            absA = fabs(A[k * N + i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }

        if (maxA < Tol) return 0;  // Matrix is degenerate

        if (imax != i) {
            // Pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            // Pivoting rows of A
            #pragma acc atomic update
            {
                for (j = 0; j < N; j++) {
                    ptr = A[i * N + j];
                    A[i * N + j] = A[imax * N + j];
                    A[imax * N + j] = ptr;
                }
            }

            P[N]++;
        }

        #pragma acc loop independent
        for (j = i + 1; j < N; j++) {
            A[j * N + i] /= A[i * N + i];
            #pragma acc loop independent
            for (k = i + 1; k < N; k++) {
                A[j * N + k] -= A[j * N + i] * A[i * N + k];
            }
        }
    }
    return 1;  // Decomposition successful
}

/*
 * Solves the system Ax = b using the LU decomposition stored in A with pivot indices P
 * Forward and backward substitution to solve LUx = Pb
 */
#pragma acc kernels
void LUPSolveC1(double A[][MAX_NUM_COMP], int *P, double *b, long N, double *x) {
    #pragma acc loop independent
    for (long i = 0; i < N; i++) {
        x[i] = b[P[i]];
        #pragma acc loop seq
        for (long k = 0; k < i; k++) {
            x[i] -= A[i][k] * x[k];
        }
    }

    for (long i = N - 1; i >= 0; i--) {
        #pragma acc loop seq
        for (long k = i + 1; k < N; k++) {
            x[i] -= A[i][k] * x[k];
        }
        x[i] /= A[i][i];
    }
}

/*
 * Solves the system Ax = b using the LU decomposition for a flattened matrix A
 */
#pragma acc kernels
void LUPSolveC2(double A[], int *P, double *b, long N, double *x) {
    #pragma acc loop independent
    for (long i = 0; i < N; i++) {
        x[i] = b[P[i]];
        #pragma acc loop seq
        for (long k = 0; k < i; k++) {
            x[i] -= A[i * N + k] * x[k];
        }
    }

    for (long i = N - 1; i >= 0; i--) {
        #pragma acc loop seq
        for (long k = i + 1; k < N; k++) {
            x[i] -= A[i * N + k] * x[k];
        }
        x[i] /= A[i * N + i];
    }
}

/*
 * Inverts the matrix A using the LU decomposition results
 */
#pragma acc kernels
void LUPInvertC1(double A[][MAX_NUM_COMP], int *P, long N, double IA[][MAX_NUM_COMP]) {
    #pragma acc loop independent
    for (long j = 0; j < N; j++) {
        #pragma acc loop independent
        for (long i = 0; i < N; i++) {
            IA[i][j] = (P[i] == j) ? 1.0 : 0.0;

            #pragma acc loop seq
            for (long k = 0; k < i; k++) {
                IA[i][j] -= A[i][k] * IA[k][j];
            }
        }

        for (long i = N - 1; i >= 0; i--) {
            #pragma acc loop seq
            for (long k = i + 1; k < N; k++) {
                IA[i][j] -= A[i][k] * IA[k][j];
            }
            IA[i][j] /= A[i][i];
        }
    }
}

/*
 * Inverts the matrix A using the LU decomposition for a flattened matrix A
 */
#pragma acc kernels
void LUPInvertC2(double A[], int *P, long N, double IA[][MAX_NUM_COMP]) {
    #pragma acc loop independent
    for (long j = 0; j < N; j++) {
        #pragma acc loop independent
        for (long i = 0; i < N; i++) {
            IA[i][j] = (P[i] == j) ? 1.0 : 0.0;

            #pragma acc loop seq
            for (long k = 0; k < i; k++) {
                IA[i][j] -= A[i * N + k] * IA[k][j];
            }
        }

        for (long i = N - 1; i >= 0; i--) {
            #pragma acc loop seq
            for (long k = i + 1; k < N; k++) {
                IA[i][j] -= A[i * N + k] * IA[k][j];
            }
            IA[i][j] /= A[i * N + i];
        }
    }
}

/*
 * LU Decomposition for phase components
 */
#pragma acc routine seq
int LUPDecomposePC1(double A[][MAX_NUM_PHASE_COMP], long N, double Tol, int *P) {
    long i, j, k, imax;
    double maxA, absA, ptr;

    #pragma acc loop independent
    for (i = 0; i <= N; i++) {
        P[i] = i;
    }

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        #pragma acc loop independent reduction(max:maxA)
        for (k = i; k < N; k++) {
            absA = fabs(A[k][i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }

        if (maxA < Tol) return 0; // Matrix is degenerate

        if (imax != i) {
            // Pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            // Pivoting rows of A
            #pragma acc atomic update
            for (j = 0; j < N; j++) {
                ptr = A[i][j];
                A[i][j] = A[imax][j];
                A[imax][j] = ptr;
            }

            P[N]++;
        }

        #pragma acc loop independent
        for (j = i + 1; j < N; j++) {
            A[j][i] /= A[i][i];
            #pragma acc loop independent
            for (k = i + 1; k < N; k++) {
                A[j][k] -= A[j][i] * A[i][k];
            }
        }
    }

    return 1; // Decomposition successful
}
/*
 * LU Decomposition with a flattened array represented as 1D treated as 2D
 */
#pragma acc routine seq
int LUPDecomposePC2(double A[(MAX_NUM_PHASE_COMP)*(MAX_NUM_PHASE_COMP)], long N, double Tol, int *P) {
    long i, j, k, imax;
    double maxA, absA;

    #pragma acc loop independent
    for (i = 0; i <= N; i++) {
        P[i] = i;  // Initialize permutation matrix
    }

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        #pragma acc loop independent reduction(max:maxA)
        for (k = i; k < N; k++) {
            absA = fabs(A[k * N + i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }

        if (maxA < Tol) return 0; // Matrix is degenerate

        if (imax != i) {
            // Pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            // Pivoting rows of A
            #pragma acc loop independent
            for (j = 0; j < N; j++) {
                double tmp = A[i * N + j];
                A[i * N + j] = A[imax * N + j];
                A[imax * N + j] = tmp;
            }

            #pragma acc atomic update
            P[N]++;
        }

        #pragma acc loop independent
        for (j = i + 1; j < N; j++) {
            A[j * N + i] /= A[i * N + i];
            #pragma acc loop independent
            for (k = i + 1; k < N; k++) {
                A[j * N + k] -= A[j * N + i] * A[i * N + k];
            }
        }
    }

    return 1;  // Decomposition successful
}

/*
 * Solving LU system for phase components
 */
#pragma acc kernels
void LUPSolvePC1(double A[][MAX_NUM_PHASE_COMP], int *P, double *b, long N, double *x) {
    #pragma acc loop independent
    for (long i = 0; i < N; i++) {
        x[i] = b[P[i]];
        #pragma acc loop seq
        for (long k = 0; k < i; k++) {
            x[i] -= A[i][k] * x[k];
        }
    }

    #pragma acc loop independent
    for (long i = N - 1; i >= 0; i--) {
        #pragma acc loop seq
        for (long k = i + 1; k < N; k++) {
            x[i] -= A[i][k] * x[k];
        }
        x[i] /= A[i][i];
    }
}

/*
 * Solving LU system using forward and backward substitution for phase components
 */
#pragma acc kernels
void LUPSolvePC2(double A[(MAX_NUM_PHASE_COMP)*(MAX_NUM_PHASE_COMP)], int *P, double *b, long N, double *x) {
    #pragma acc loop independent
    for (long i = 0; i < N; i++) {
        x[i] = b[P[i]];

        #pragma acc loop seq
        for (long k = 0; k < i; k++) {
            x[i] -= A[i * N + k] * x[k];
        }
    }

    #pragma acc loop independent
    for (long i = N - 1; i >= 0; i--) {
        #pragma acc loop seq
        for (long k = i + 1; k < N; k++) {
            x[i] -= A[i * N + k] * x[k];
        }
        x[i] /= A[i * N + i];
    }
}


/*
 * Inverting a matrix using LU decomposition for phase components
 */
#pragma acc kernels
void LUPInvertPC1(double A[][MAX_NUM_PHASE_COMP], int *P, long N, double IA[][MAX_NUM_PHASE_COMP]) {
    #pragma acc loop independent
    for (long j = 0; j < N; j++) {
        #pragma acc loop independent
        for (long i = 0; i < N; i++) {
            IA[i][j] = (P[i] == j) ? 1.0 : 0.0;

            #pragma acc loop seq
            for (long k = 0; k < i; k++) {
                IA[i][j] -= A[i][k] * IA[k][j];
            }
        }

        #pragma acc loop independent
        for (long i = N - 1; i >= 0; i--) {
            #pragma acc loop seq
            for (long k = i + 1; k < N; k++) {
                IA[i][j] -= A[i][k] * IA[k][j];
            }
            IA[i][j] /= A[i][i];
        }
    }
}

/*
 * Inverting a matrix using the results of LU decomposition for phase components
 */
#pragma acc kernels
void LUPInvertPC2(double A[(MAX_NUM_PHASE_COMP)*(MAX_NUM_PHASE_COMP)], int *P, long N, double IA[][MAX_NUM_PHASE_COMP]) {
    #pragma acc loop independent
    for (long j = 0; j < N; j++) {
        #pragma acc loop independent
        for (long i = 0; i < N; i++) {
            IA[i][j] = (P[i] == j) ? 1.0 : 0.0;

            #pragma acc loop seq
            for (long k = 0; k < i; k++) {
                IA[i][j] -= A[i * N + k] * IA[k][j];
            }
        }

        #pragma acc loop independent
        for (long i = N - 1; i >= 0; i--) {
            #pragma acc loop seq
            for (long k = i + 1; k < N; k++) {
                IA[i][j] -= A[i * N + k] * IA[k][j];
            }
            IA[i][j] /= A[i * N + i];
        }
    }
}


