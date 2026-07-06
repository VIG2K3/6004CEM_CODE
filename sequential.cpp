#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

const int SIZE = 1024;

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

void sequentialMultiply() {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            double sum = 0;
            for (int k = 0; k < SIZE; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main() {
    initializeMatrices();

    cout << "Sequential Matrix Multiplication" << endl;
    cout << "Matrix size: " << SIZE << " x " << SIZE << endl;

    auto start = high_resolution_clock::now();
    sequentialMultiply();
    auto end = high_resolution_clock::now();

    double timeTaken = duration<double>(end - start).count();

    cout << "Execution time: " << timeTaken << " seconds" << endl;
    cout << "Checksum: " << checksum() << endl;

    return 0;
}
