#include <bits/stdc++.h>
#include <pthread.h>

#define max_thread_count 16 // Ryzen 5800H has 16 threads. Ryzen 9950X has 32 threads.
using namespace std;
pthread_mutex_t matrix_lock;
struct threaddata
{
	vector<vector<double>>* firstmat;
	vector<vector<double>>* secondmat;
	vector<vector<double>>* resultmat;
	int startrow;
	int endrow;
};

void matrixmultiplication(vector<vector<double>>& firstmat, vector<vector<double>>& secondmat, vector<vector<double>>& resultmat)
{
	for(int i = 0; i < firstmat.size(); i++)
	{
		for(int j = 0; j < secondmat.at(0).size(); j++)
		{
			double sum = 0;
			for(int k = 0; k < firstmat.at(i).size(); k++)
			{
				sum += firstmat.at(i).at(k) * secondmat.at(k).at(j);
			}
		resultmat.at(i).at(j) = sum;
		}
	}
	return;
}

//parallel matrix multiplication using 16 threads
void matmult(vector<vector<double>>& firstmat, vector<vector<double>>& secondmat, vector<vector<double>>& resultmat)
{
	pthread_t threads[max_thread_count];
	threaddata data[max_thread_count];
	auto worker = [](void* argument) -> void*
	{
		threaddata* data = (threaddata*)argument;
		for(int i = data->startrow; i < data->endrow; i++)
		{
			for(int j = 0; j < data->secondmat->at(0).size(); j++)
			{
				double sum = 0;
				for(int k = 0; k < data->firstmat->at(i).size(); k++)
				{
					sum += data->firstmat->at(i).at(k) * data->secondmat->at(k).at(j);
				}
				pthread_mutex_lock(&matrix_lock);
				data->resultmat->at(i).at(j) = sum;
				pthread_mutex_unlock(&matrix_lock);
			}
		}
		return NULL;
	};
	//define and create 16 threads
	for(int i = 0; i < max_thread_count; i++)
	{
		data[i].firstmat = &firstmat;
		data[i].secondmat = &secondmat;
		data[i].resultmat = &resultmat;

		data[i].startrow =
			i * firstmat.size() / max_thread_count;

		data[i].endrow =
			(i + 1) * firstmat.size()/ max_thread_count;
		pthread_create(&threads[i],NULL,worker,&data[i]);
	}
	//wait for all 16 threads to finish
	for(int i = 0; i < max_thread_count; i++)
	{
		pthread_join(threads[i], NULL);
	}
	return;
}
void printmatrix(const vector<vector<double>>& matrix)
{
    uint rows = matrix.size();
    uint cols = matrix.empty() ? 0 : matrix.at(0).size();

    cout << "Matrix (" << rows << " x " << cols << "):\n";

    for (int i = 0; i < matrix.size(); i++)
    {
        cout << "[ ";

        for (int j = 0; j < matrix.at(i).size(); j++)
        {
            cout << fixed << setprecision(2) << setw(8) << matrix.at(i).at(j) << " ";
        }

        cout << "]\n";
    }

    cout << "\n";
}

void fillmatrix(vector<vector<double>>& matrix)
{
	srand(time(NULL));
	for(int i = 0; i < matrix.size(); i++)
	{
		for(int j = 0; j < matrix.at(i).size(); j++)
		{
			matrix.at(i).at(j) = rand() % 10;
		}
	}
}
double amdahls_law(double parallel_fraction, int thread_count)
{
	double serial_fraction = 1.0 - parallel_fraction;
	double speedup = 1.0 / (serial_fraction + (parallel_fraction / thread_count));
	return speedup;
}
int main()
{
	pthread_mutex_init(&matrix_lock, NULL);
	int row, col;
	int row1, col1;
	cout << "Enter rows and columns for first matrix: ";
	cin >> row >> col;
    	cout << "Enter rows and columns for second matrix: ";
    	cin >> row1 >> col1;
	//define the matricies
	vector<vector<double>> firstmat(row, vector<double>(col));
	vector<vector<double>> secondmat(row1, vector<double>(col1));
	vector<vector<double>> sequentialresult(row,vector<double>(col1));
	vector<vector<double>> parallelresult(row,vector<double>(col1));
	//generate input matrix
	fillmatrix(firstmat);
	sleep(1);
	fillmatrix(secondmat);

	//print matrix for verification purpose
//	printmatrix(firstmat);
	cout << "-------------------" << endl;
//	printmatrix(secondmat);
	//perform matrix mult
	auto start = chrono::steady_clock::now();
	matrixmultiplication(firstmat, secondmat,sequentialresult);
	auto end = chrono::steady_clock::now();
	chrono::duration<double, milli> sequential_elapsed = end - start;
//	printmatrix(sequentialresult);
	cout << "2 sticks of RAM COSTS $900. 67 on the merry christmas" << endl;
	cout << "Matrix multiplication took " << sequential_elapsed.count() << " milliseconds. " << endl;
	cout << "----------------------" << endl << " Implementing parallelized version " << endl;
	auto parastart = chrono::steady_clock::now();
	matmult(firstmat, secondmat,parallelresult);
	auto paraend = chrono::steady_clock::now();
	chrono::duration<double, milli> parallel_elapsed = paraend - parastart;
	cout << "Parallel matrix multiplication took " << parallel_elapsed.count() << " milliseconds.\n";
	//call amdahls law function to compute the speedup performance
	double parallel_fraction = 0.95;
	double measured_speedup = sequential_elapsed.count() / parallel_elapsed.count();
	double performance_improvement = (measured_speedup - 1.0) * 100.0;
	cout << "The speedup performance determined from Amdahl's Law is " << amdahls_law(parallel_fraction, max_thread_count) << endl;
	cout << "Actual Speedup performance is " << measured_speedup << endl;
	pthread_mutex_destroy(&matrix_lock);
	return 0;
}
