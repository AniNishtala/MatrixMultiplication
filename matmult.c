#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int matrixmult(int r1, int c1, int r2, int c2, double firstmat[r1][c1], double secondmat[r2][c2], double result[r1][c2])
{
    if (r1 <= 0 || c1 <= 0 || r2 <= 0 || c2 <= 0 || c1 != r2 || !firstmat || !secondmat || !result)
    {
        return 0;
    }

    for (int i = 0; i < r1; i++)
    {
        double *result_row = result[i];
        double *first_row = firstmat[i];
        // Initialize the result row before accumulating.
        for (int j = 0; j < c2; j++)
        {
            result_row[j] = 0.0;
        }
        for (int k = 0; k < c1; k++)
        {
            double a = first_row[k];
            double *second_row = secondmat[k];
            for (int j = 0; j < c2; j++)
            {
                result_row[j] += a * second_row[j];
            }
        }
    }
    return 1;
}

//basic implementation
int matrixmultiplication(int r1, int c1, int r2, int c2, double firstmat[r1][c1], double secondmat[r2][c2], double result[r1][c2])
{
    //r1 = row1, c1 = column1, r2 = row2, c2 = column2. 1 refers to mat1 dimensions and 2 refers to mat2 dimensions.
    if (r1 <= 0 || c1 <= 0 || r2 <= 0 || c2 <= 0 || c1 != r2 || !firstmat || !secondmat || !result)
    {
        return 0;
    }

    for(int i = 0; i < r1; i++)
    {
        for(int j = 0; j < c2; j++)
        {
            double sum = 0;
            for(int k = 0; k < c1; k++)
            {
                sum += firstmat[i][k] * secondmat[k][j];
            }
            result[i][j] = sum;
        }
    }
    return 1;
}

void printmatrix(const char *name, int rows, int cols, double matrix[rows][cols])
{
    printf("%s (%d x %d):\n", name, rows, cols);

    for (int i = 0; i < rows; i++) {
        printf("[ ");

        for (int j = 0; j < cols; j++) {
            printf("%7.1f ", matrix[i][j]);
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
    if (scanf("%d %d", &row, &row1) != 2 || row <= 0 || row1 <= 0)
    {
        printf("Invalid row count.\n");
        return 1;
    }

    printf("Enter columns: ");
    if (scanf("%d %d", &col, &col1) != 2 || col <= 0 || col1 <= 0 || col != row1)
    {
        printf("Invalid column count.\n");
        return 1;
    }

    // Declare the arrays
    double (*firstmat)[col] = malloc((size_t)row * sizeof(*firstmat));
    double (*secondmat)[col1] = malloc((size_t)row1 * sizeof(*secondmat));
    double (*resultmat)[col1] = malloc((size_t)row * sizeof(*resultmat));

    if (!firstmat || !secondmat || !resultmat)
    {
        printf("Memory allocation failed.\n");
        free(firstmat);
        free(secondmat);
        free(resultmat);
        return 1;
    }

    // Fill the matrix with some random values.
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            firstmat[i][j] = rand() % 10;
        }
    }
//  printf(" MatrixA print is below\n");
    //  printmatrix("Matrix A", row, col, firstmat);

    for(int i = 0; i < row1; i++)
        for(int j = 0; j < col1; j++)
            secondmat[i][j] = rand() % 10;
    //  printf("matrix b print is below\n");
    //  printmatrix("Matrix B", row1, col1, secondmat);

    clock_t start = clock();
