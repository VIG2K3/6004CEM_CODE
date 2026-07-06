#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <algorithm>

using namespace std;

static void initializeMatrices(vector<double>& A, vector<double>& B, vector<double>& C, int N) {
    srand(42);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = rand() % 10;
            B[i * N + j] = rand() % 10;
            C[i * N + j] = 0.0;
        }
    }
}

static double checksum(const vector<double>& C) {
    double sum = 0.0;
    for (double v : C) sum += v;
    return sum;
}

static void multiplyRows(const vector<double>& localA,
                         const vector<double>& B,
                         vector<double>& localC,
                         int rows,
                         int N) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += localA[i * N + k] * B[k * N + j];
            }
            localC[i * N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);

    vector<double> A;
    vector<double> B(N * N);
    vector<double> C;

    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N);
        initializeMatrices(A, B, C, N);
        cout << "MPI Distributed Matrix Multiplication" << endl;
        cout << "Matrix size: " << N << " x " << N << endl;
        cout << "Processes: " << worldSize << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double totalStart = MPI_Wtime();

    // Collective operation: distribute full B matrix to every process.
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int baseRows = N / worldSize;
    int extraRows = N % worldSize;
    int myRows = baseRows + (rank < extraRows ? 1 : 0);
    int myOffset = rank * baseRows + min(rank, extraRows);

    vector<double> localA(myRows * N);
    vector<double> localC(myRows * N, 0.0);

    if (rank == 0) {
        // Master-worker topology: rank 0 sends each worker its row block.
        for (int p = 1; p < worldSize; p++) {
            int rows = baseRows + (p < extraRows ? 1 : 0);
            int offset = p * baseRows + min(p, extraRows);
            int header[2] = {offset, rows};
            MPI_Send(header, 2, MPI_INT, p, 0, MPI_COMM_WORLD);
            MPI_Send(A.data() + offset * N, rows * N, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
        }
        copy(A.begin() + myOffset * N, A.begin() + (myOffset + myRows) * N, localA.begin());
    } else {
        int header[2];
        MPI_Recv(header, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        myOffset = header[0];
        myRows = header[1];
        localA.resize(myRows * N);
        localC.assign(myRows * N, 0.0);
        MPI_Recv(localA.data(), myRows * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double computeStart = MPI_Wtime();
    multiplyRows(localA, B, localC, myRows, N);
    double computeEnd = MPI_Wtime();
    double localComputeTime = computeEnd - computeStart;

    double maxComputeTime = 0.0;
    MPI_Reduce(&localComputeTime, &maxComputeTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        copy(localC.begin(), localC.end(), C.begin() + myOffset * N);
        for (int p = 1; p < worldSize; p++) {
            int rows = baseRows + (p < extraRows ? 1 : 0);
            int offset = p * baseRows + min(p, extraRows);
            MPI_Recv(C.data() + offset * N, rows * N, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(localC.data(), myRows * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double totalEnd = MPI_Wtime();

    if (rank == 0) {
        double totalTime = totalEnd - totalStart;
        double commOverhead = totalTime - maxComputeTime;
        double ratio = (maxComputeTime > 0.0) ? commOverhead / maxComputeTime : 0.0;

        cout << fixed << setprecision(6);
        cout << "Total execution time: " << totalTime << " seconds" << endl;
        cout << "Max computation time: " << maxComputeTime << " seconds" << endl;
        cout << "Estimated communication/synchronization overhead: " << commOverhead << " seconds" << endl;
        cout << "Communication-to-computation ratio: " << ratio << endl;
        cout << "Checksum: " << checksum(C) << endl;
    }

    MPI_Finalize();
    return 0;
}
