#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <algorithm>

using namespace std;
using namespace std::chrono;

const int SIZE = 1024;
const int BLOCK_SIZE = 128;

vector<vector<double>> A(SIZE, vector<double>(SIZE));
vector<vector<double>> B(SIZE, vector<double>(SIZE));
vector<vector<double>> C(SIZE, vector<double>(SIZE));

void initializeMatrices() {
    srand(42);
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
            C[i][j] = 0;
        }
    }
}

double checksum() {
    double sum = 0;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            sum += C[i][j];
    return sum;
}

int taskBasedMultiply(int threads) {
    int totalTasks = 0;

    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single
        {
            for (int ii = 0; ii < SIZE; ii += BLOCK_SIZE) {
                for (int jj = 0; jj < SIZE; jj += BLOCK_SIZE) {

                    totalTasks++;

                    #pragma omp task firstprivate(ii, jj)
                    {
                        for (int i = ii; i < min(ii + BLOCK_SIZE, SIZE); i++) {
                            for (int j = jj; j < min(jj + BLOCK_SIZE, SIZE); j++) {
                                double sum = 0;

                                for (int k = 0; k < SIZE; k++) {
                                    sum += A[i][k] * B[k][j];
                                }

                                C[i][j] = sum;
                            }
                        }
                    }
                }
            }

            #pragma omp taskwait
        }
    }

    return totalTasks;
}

int main() {
    int threads;

    cout << "Enter number of threads: ";
    cin >> threads;

    initializeMatrices();

    cout << "\nOpenMP Task-Based Matrix Multiplication" << endl;
    cout << "Matrix size: " << SIZE << " x " << SIZE << endl;
    cout << "Threads: " << threads << endl;
    cout << "Block size: " << BLOCK_SIZE << " x " << BLOCK_SIZE << endl;

    auto start = high_resolution_clock::now();
    int totalTasks = taskBasedMultiply(threads);
    auto end = high_resolution_clock::now();

    double timeTaken = duration<double>(end - start).count();

    cout << "Total tasks: " << totalTasks << endl;
    cout << "Execution time: " << timeTaken << " seconds" << endl;
    cout << "Checksum: " << checksum() << endl;

    return 0;
}
