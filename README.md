# matrix-mpi
MPI matrix multiplication benchmark

This repository includes a parallelized O(n^3) matrix multiplication implementation that distributes both the A matrix and the rows of B matrix in MPI.

It uses row-major storage for matrices and uses `MPI_Wtime()` to time everything.

The results can be verified using `verify()` function, but it is currently commented out.

It also includes a second mode that uses asynchronous Send/Recv facilities. Based on our experiments async is slower than sync, and the overhead is approximately linear (due to the extra overhead to manage the state of each requests).

To use, please read the main.c file and compile with the correct macros turned on (use macros to specify SIZE and MODE).

Copyright &copy; Rui-Jie Fang, 2018.
