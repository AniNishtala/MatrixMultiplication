#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
__global__ void matmultiplication(int r1, int c1, int c2, double *firstmat, double *secondmat, double *resultmat)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	double sum = 0;
	if(x >= r1 || y >= c2)
		return;
	for(int k = 0; k < c1; k++)
	{
		sum += firstmat[x * c1 + k] * secondmat[k * c2 + y];
	}
	resultmat[x * c2 + y] = sum;
}
int matrixmultiplication(int r1, int c1, int r2, int c2, double *firstmat, double *secondmat, double *result)
{
    if (c1 != r2) {
        return 0;
    }

    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < c1; k++)
            {
                sum += firstmat[i * c1 + k] * secondmat[k * c2 + j];
            }
            result[i * c2 + j] = sum;
        }
    }
    return 1;
}
void printmatrix(char *name, int rows, int cols, double *matrix)
{
    printf("%s (%d x %d):\n", name, rows, cols);
    for(int i = 0; i < rows; i++)
    {
        printf("[ ");
        for(int j = 0; j < cols; j++)
        {
            printf("%7.1f ", matrix[i * cols + j]);
        }
        printf("]\n");
    }
    printf("\n");
}
int main(void)
{
    int row;
    int col;
    int row1;
    int col1;

    srand((unsigned int)time(NULL));

    // Get the dimensions of both matrices.
    printf("Enter rows: ");
    if(scanf("%d %d", &row, &row1) != 2 || row <= 0 || row1 <= 0)
    {
        printf("Invalid row count.\n");
        return 1;
    }

    printf("Enter columns: ");
    if(scanf("%d %d", &col, &col1) != 2 || col <= 0 || col1 <= 0 || col != row1)
    {
        printf("Invalid column count.\n");
        return 1;
    }

    // Calculate the number of bytes needed for each matrix.
    int firstBytes = (int) row * col * sizeof(double);

    int secondBytes = (int) row1 * col1 * sizeof(double);

    int resultBytes = (int) row * col1 * sizeof(double);

    // Allocate memory for the host matrices.
    double *firstmat = (double *)malloc(firstBytes);

    double *secondmat = (double *)malloc(secondBytes);

    double *resultmat = (double *)malloc(resultBytes);

    // Fill the first matrix with random values. Modify with functioncall later
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            firstmat[i * col + j] = rand() % 10;
        }
    }

    // Fill the second matrix with random values.
    for(int i = 0; i < row1; i++)
    {
        for(int j = 0; j < col1; j++)
        {
            secondmat[i * col1 + j] = rand() % 10;
        }
    }

    // Run the CPU matrix multiplication and measure its time.
    clock_t start = clock();

    int retvalue = matrixmultiplication(row,col,row1,col1,firstmat,secondmat,resultmat);

    clock_t end = clock();

    if(!retvalue)
    {
        printf("Cannot multiply matrices.\n");

        free(firstmat);
        free(secondmat);
        free(resultmat);

        return 1;
    }

    double cpuSeconds = (double)(end - start) / CLOCKS_PER_SEC;

    printf("CPU time: %.9f seconds\n", cpuSeconds);

    // Declare pointers for memory on the GPU.
    double *firstmat_d;
    double *secondmat_d;
    double *resultmat_d;

    // Allocate memory on the GPU.
    cudaMalloc((void **)&firstmat_d, firstBytes);
    cudaMalloc((void **)&secondmat_d, secondBytes);
    cudaMalloc((void **)&resultmat_d, resultBytes);

    // Copy the input matrices from the CPU to the GPU.
    cudaMemcpy(firstmat_d,firstmat,firstBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(secondmat_d,secondmat,secondBytes,cudaMemcpyHostToDevice);

    // define the block dims. 
    dim3 block(32, 32);

    dim3 grid((row + block.x - 1) / block.x,(col1 + block.y - 1) / block.y);

    // Start measuring GPU execution time. Experiment with nvprof later
    start = clock();

    // init CUDA kernel.
    matmultiplication<<<grid, block>>>(row,col,col1,firstmat_d,secondmat_d,resultmat_d);

    // Block CPU & Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy data back
    cudaMemcpy(resultmat,resultmat_d,resultBytes,cudaMemcpyDeviceToHost);

    // Timer finish
    end = clock();

    double gpuSeconds = (double)(end - start) / CLOCKS_PER_SEC;

    printf("GPU time: %.9f seconds\n", gpuSeconds);

    // Print the result matrix.
//    printmatrix("Result Matrix is ",row,col1,resultmat);

    // Free GPU memory.
    cudaFree(firstmat_d);
    cudaFree(secondmat_d);
    cudaFree(resultmat_d);

    // Free CPU memory.
    free(firstmat);
    free(secondmat);
    free(resultmat);

    return 0;
}
