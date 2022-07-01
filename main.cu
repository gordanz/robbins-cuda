#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#include "simulate.h"
#include "seeding.h"
#include "utils.h"

int main()
{

    cudaFree(0);
    timer();

    // 1. Constants
    ///////////////////////////
    const int nSecretaries = 1000;
    const int nSimulations = 1<<15;
    const int nThreads = 1<<10; //
    const int nBlocks = 1<<5;
    int nThreadsPerBlock = nThreads / nBlocks;
    int nsim = nSimulations / nThreads;

    ln();dhline();
    printf("Number Of Secretaries \t\t\t\t%d", nSecretaries);
    printf("\nNumber Of Blocks \t\t\t\t\t%d", nBlocks);
    printf("\nNumber Of ThreadsPerBlock \t\t\t%d", nThreadsPerBlock);
    printf("\nNumber Of Threads \t\t\t\t\t%d", nThreads);
    printf("\nNumber Of Simulations (per thread) \t%d", nsim);
    printf("\nNumber Of Simulations (total) \t\t%d", nSimulations);
    printf("\nProcessor \t\t\t\t\t\t\tGPU (Cuda)");
    ln(); hline();


    // 2. Memory allocation and seeding
    ///////////////////////////////////////

    // Host
    uint64_t *h_sd = (uint64_t*) malloc(4 *  nThreads * sizeof(uint64_t));
    double *h_res  = (double*)   malloc(     nThreads * sizeof(double));

    // Device
    REAL *d_x;
    SECINT *d_left, *d_right, *d_nlch;
    uint64_t *d_sd;
    double *d_res;
    cudaMalloc( (uint64_t**) &d_sd, 4 * nThreads *  sizeof (uint64_t) );
    cudaMalloc( (REAL**) &d_x, nThreads * nSecretaries * sizeof (REAL) );
    cudaMalloc( (SECINT**) &d_left, nThreads * nSecretaries * sizeof (SECINT) );
    cudaMalloc( (SECINT**) &d_right, nThreads * nSecretaries * sizeof (SECINT) );
    cudaMalloc( (SECINT**) &d_nlch, nThreads * nSecretaries * sizeof (SECINT) );
    cudaMalloc( (double**) &d_res, nThreads * sizeof (double) );
    // Seed and Copy from host to device
    seeds(h_sd, nThreads);
    cudaMemcpy(d_sd, h_sd, 4 * nThreads *  sizeof (uint64_t), cudaMemcpyHostToDevice);

    // 3. Run simulations
    /////////////////////////////////

    simulate<<<nBlocks,nThreadsPerBlock>>>(
        nsim, nSecretaries,
        d_sd,
        d_x, d_left, d_right, d_nlch,
        d_res );

    cudaDeviceSynchronize();

    // 4. Copy results back to host
    ///////////////////////////////////
    cudaMemcpy(h_res, d_res,  nThreads *  sizeof (double), cudaMemcpyDeviceToHost);

    // 5. Output and Statistics
    ////////////////////
    //  printf("Values:\n");
    //  for (int i = 0; i < nThreads; ++i)
    //  {
    //      printf("%.3f ", h_res[i]);
    //  }
    // ln();
    printf("Statistics:");
    // statistics(res,nThreads,0.367879441171442);
    // statistics_compare(h_res,nThreads,2.0/nSecretaries);
    statistics(h_res,nThreads);
    hline();
    timer();
    dhline();


    // 6. Cleanup
    ///////////////////////////

    free(h_sd);
    free(h_res);

    cudaFree(d_x);
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_nlch);
    cudaFree(d_res);


}
