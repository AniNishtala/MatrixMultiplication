#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stddef.h>

#define GPU_TILE_SIZE 32 // defines the tile size as 32 used for the tiled kernel function

//matrix multiplication implemented using tiles + shared memory approach
__global__ void tiled_mat_mult(int r1, int c1, int c2, double *firstmat, double *secondmat, double *resultmat)
{
    //define the onchip memory based on the tile size that each block shares. 32x32x8Bx2 = 16KB
    __shared__ double Tile1[GPU_TILE_SIZE][GPU_TILE_SIZE];
    __shared__ double Tile2[GPU_TILE_SIZE][GPU_TILE_SIZE];

    // x = row, y = column
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    double sum = 0.0;
    int numTiles = (c1 + GPU_TILE_SIZE - 1) / GPU_TILE_SIZE;

    for(int t = 0; t < numTiles; t++)
    {
        int aCol = t * GPU_TILE_SIZE + threadIdx.y;
        int bRow = t * GPU_TILE_SIZE + threadIdx.x;

        // Load firstmat tile. ThreadIdx.x is row & threadIdx.y is col
        if(x < r1 && aCol < c1)
        {
            Tile1[threadIdx.x][threadIdx.y] = firstmat[x * c1 + aCol];
        }
        else
        {
            Tile1[threadIdx.x][threadIdx.y] = 0.0;
        }

        // Load secondmat tile.
        if(bRow < c1 && y < c2)
        {
            Tile2[threadIdx.x][threadIdx.y] = secondmat[bRow * c2 + y];
        }
        else
        {
            Tile2[threadIdx.x][threadIdx.y] = 0.0;
        }

        __syncthreads();

        for(int k = 0; k < GPU_TILE_SIZE; k++)
        {
            sum += Tile1[threadIdx.x][k] * Tile2[k][threadIdx.y];
        }

        __syncthreads();
    }

    if(x < r1 && y < c2)
    {
        resultmat[x * c2 + y] = sum;
    }
}
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

int matrixmult(int r1, int c1, int r2, int c2, double *firstmat, double *secondmat, double *result)
{
    if (r1 <= 0 || c1 <= 0 || r2 <= 0 || c2 <= 0 || c1 != r2 || !firstmat || !secondmat || !result)
    {
        return 0;
    }

    for (int i = 0; i < r1; i++)
    {
        double *result_row = result + (size_t)i * c2;
        double *first_row = firstmat + (size_t)i * c1;
        // Initialize the result row before accxumulating.
        for (int j = 0; j < c2; j++)
        {
            result_row[j] = 0.0;
        }
        for (int k = 0; k < c1; k++)
        {
            double a = first_row[k];
            double *second_row = secondmat + (size_t)k * c2;
            for (int j = 0; j < c2; j++)
            {
                result_row[j] += a * second_row[j];
            }
        }
    }
    return 1;
}
// 
int matrixmultiplication(int r1, int c1, int r2, int c2, double *firstmat, double *secondmat, double *result)
{
    if (c1 != r2)
    {
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

//function to generate input matricies
void fillmatrix(int rows, int cols, double *matrix)
{
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            matrix[i * cols + j] = rand() % 10;
        }
    }
}

//function to print the matrix for verification. used for small matricies 
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
    size_t firstBytes = (size_t)row * col * sizeof(double);

    size_t secondBytes = (size_t)row1 * col1 * sizeof(double);

    size_t resultBytes = (size_t)row * col1 * sizeof(double);

    double *firstmat = (double *)malloc(firstBytes);

    double *secondmat = (double *)malloc(secondBytes);

    double *resultmat = (double *)malloc(resultBytes);

    double *resultmat2 = (double *)malloc(resultBytes);

    double *resultmatGPU = (double *)malloc(resultBytes);

    double *resultmat_tiled_GPU = (double *)malloc(resultBytes);

    if(!firstmat || !secondmat || !resultmat || !resultmat2 || !resultmatGPU || !resultmat_tiled_GPU)
    {
        printf("Host memory allocation failed.\n");
        return 1;
    }
    // Fill both input matrices.
    fillmatrix(row, col, firstmat);
    fillmatrix(row1, col1, secondmat);

    // init CPU matrixmultiplication.
    clock_t start = clock();

    matrixmultiplication(row, col, row1, col1, firstmat, secondmat, resultmat);

    clock_t end = clock();

    double cpuTime1 = (double)(end - start) / CLOCKS_PER_SEC;

    printf("CPU basic time: %.9f seconds\n", cpuTime1);
    
    //init optimized CPU matmult
    start = clock();

    matrixmult(row, col, row1, col1, firstmat, secondmat, resultmat2);

    end = clock();

    double cpuTime2 = (double)(end - start) / CLOCKS_PER_SEC;

    printf("CPU optimized time: %.9f seconds\n", cpuTime2);


    //GPU alloc
    double *firstmat_d;
    double *secondmat_d;
    double *resultmat_d;

    cudaMalloc((void **)&firstmat_d, firstBytes);
    cudaMalloc((void **)&secondmat_d, secondBytes);
    cudaMalloc((void **)&resultmat_d, resultBytes);

    cudaMemcpy(firstmat_d, firstmat, firstBytes, cudaMemcpyHostToDevice);

    cudaMemcpy(secondmat_d, secondmat, secondBytes, cudaMemcpyHostToDevice);

    //define 1024 threads per block (32x32)
    dim3 block(GPU_TILE_SIZE, GPU_TILE_SIZE);

    //x for rows and y for cols
    dim3 grid((row + block.x - 1) / block.x, (col1 + block.y - 1) / block.y);

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    //basic GPU matmult
    cudaEventRecord(startEvent);

    matmultiplication<<<grid, block>>>(row, col, col1, firstmat_d, secondmat_d, resultmat_d);

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float gpuMilliseconds1 = 0.0f;

    cudaEventElapsedTime(&gpuMilliseconds1, startEvent, stopEvent);

    cudaMemcpy(resultmatGPU, resultmat_d, resultBytes, cudaMemcpyDeviceToHost);

    printf("GPU untiled time: %.9f seconds\n", gpuMilliseconds1 / 1000.0);

    //init tiled approach
    cudaEventRecord(startEvent);

    tiled_mat_mult<<<grid, block>>>(row, col, col1, firstmat_d, secondmat_d, resultmat_d);

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float gpuMilliseconds2 = 0.0f;

    cudaEventElapsedTime(&gpuMilliseconds2, startEvent, stopEvent);

    cudaMemcpy(resultmat_tiled_GPU, resultmat_d, resultBytes, cudaMemcpyDeviceToHost);

    printf("GPU tiled time: %.9f seconds\n", gpuMilliseconds2 / 1000.0);

    // printmatrix("Result", row, col1, resultGPU2);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaFree(firstmat_d);
    cudaFree(secondmat_d);
    cudaFree(resultmat_d);

    free(firstmat);
    free(secondmat);
    free(resultmat);
    free(resultmat2);
    free(resultmatGPU);
    free(resultmat_tiled_GPU);
    cudaFree(firstmat_d);
    cudaFree(secondmat_d);
    cudaFree(resultmat_d);
    return 0;
}