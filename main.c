#include <stdio.h>
#include <mpi.h>
#include <memory.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
/*
 * @author Ruijie Fang
 * Actually _working_ matrix multiplication
 * using two modes:
 * 1) Asynchronous sending
 * 2) Scatterv/Gatherv
 */

#define traditional
//#define async
//#define doVerify
double *A, *B, *C;
const static double eps = 0.000000001;

#if defined(traditional) && defined(async)

__________________________ Compile Error: You cannot define both modes!!! [] __________________________
#endif

void verify(unsigned size)
{
    assert(C);
    double d = 0;
    for (unsigned i = 0; i < size; ++i)
        for (unsigned j = 0; j < size; ++j) {
            d = 0;
            for (unsigned k = 0; k < size; ++k)
                d += A[i * size + k] * B[k * size + j];
            if (!(C[i * size + j] <= d + eps && C[i * size + j] >= d - eps)) {
                printf("err @ [%u,%u]: %.3f!=%.3f\n", i, j, d, C[i * size + j]);
            }
        }
}

int main(int argc, char **argv)
{
    MPI_Request req;
    double start, end;
    int rank, size, N, P, i, j, k, *scatter_cnts, *displs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    assert(argc == 2);

    N = (int) strtol(argv[1], NULL, 10);
    P = N / size;
    assert(N > 0);

    start = MPI_Wtime();
    B = malloc(sizeof(double) * N * N);
    if (rank == 0) {
        A = malloc(sizeof(double) * N * N);
        C = malloc(sizeof(double) * N * N);
        for (i = 0; i < N; ++i)
            for (j = 0; j < N; ++j)
                A[i * N + j] = i + j, B[i * N + j] = i + j;
    }
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Master: distributing data...\n");
#ifdef traditional
        i = 1;
        for (; i < size - 1; ++i)
            MPI_Send(A + i * P * N, P * N, MPI_DOUBLE, i, 2, MPI_COMM_WORLD);
        if (i < size)
            MPI_Send(A + i * P * N, (N - P * i) * N, MPI_DOUBLE, i, 2, MPI_COMM_WORLD);
#endif
#ifdef async
        // async
        for (i = 1; i < size - 1; ++i)
            MPI_Isend(A + i * P * N, P * N, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &req);
        if (i < size)
            MPI_Isend(A + i * P * N, (N - P * i) * N, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &req);
#endif
        printf("Master: distribution done.\n");
        sleep(3);
    } else {
#if defined(traditional) || defined(async)
        if (rank != size - 1) {
            A = malloc(sizeof(double) * P * N);
            C = malloc(sizeof(double) * P * N);
            MPI_Recv(A, P * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            P = N - (size - 1) * P;
            A = malloc(sizeof(double) * P * N);
            C = malloc(sizeof(double) * P * N);
            MPI_Recv(A, P * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
#endif
    }
    memset(C, 0, sizeof(double) * N * P);
    for (i = 0; i < P; ++i)
        for (k = 0; k < N; ++k)
            for (j = 0; j < N; ++j)
                C[i * N + j] += A[i * N + k] * B[k * N + j];

#if defined(traditional) || defined(async)
    if (rank == 0) {
        printf("Master: collecting data...\n");
        i = 1;
        for (; i < size - 1; ++i)
            MPI_Recv(C + i * P * N, P * N, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (i < size)
            MPI_Recv(C + i * P * N, (N - P * i) * N, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else
        MPI_Send(C, N * P, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
#endif

    if (rank == 0) {

        end = MPI_Wtime();
        printf("MM: N=%d,P=%d finished in %f time.\n", N, size, end - start);
#ifdef doVerify
        verify(N);
#endif
    }
    free(A);
    free(B);
    free(C);
    MPI_Finalize();
    return 0;
}
