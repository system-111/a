// SPPU LP5 Practical 5: Implement HPC application for AI/ML domain
// Application: Parallel K-Means Clustering using OpenMP
//g++ -fopenmp HPC.cpp -o HPC.exe
//.\HPC.exe

#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cstdlib>
#include <ctime>

using namespace std;

// Structure to represent a 2D data point
struct Point {
    float x, y;
    int cluster;
};

// Function to calculate Euclidean distance between two points
float calculateDistance(Point p1, Point p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

// --- Parallel K-Means Algorithm ---
void parallelKMeans(vector<Point>& points, int K, int iterations) {
    int N = points.size();
    vector<Point> centroids(K);

    // Step 1: Initialize centroids randomly (picking the first K points for simplicity)
    for (int i = 0; i < K; i++) {
        centroids[i] = points[i];
    }

    // Main K-Means Loop
    for (int iter = 0; iter < iterations; iter++) {
        
        // Step 2: Assign points to the nearest centroid (Highly Parallelizable)
        // We use OpenMP here because every point can be calculated independently
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            float min_dist = 1e9; // Start with a very large distance
            int best_cluster = 0;

            for (int j = 0; j < K; j++) {
                float dist = calculateDistance(points[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            points[i].cluster = best_cluster; // Assign to closest cluster
        }

        // Step 3: Recalculate centroids based on new assignments
        vector<float> sumX(K, 0.0), sumY(K, 0.0);
        vector<int> counts(K, 0);

        // Accumulate the sums and counts
        for (int i = 0; i < N; i++) {
            int c = points[i].cluster;
            sumX[c] += points[i].x;
            sumY[c] += points[i].y;
            counts[c]++;
        }

        // Calculate the new averages to form the new centroids
        for (int j = 0; j < K; j++) {
            if (counts[j] > 0) {
                centroids[j].x = sumX[j] / counts[j];
                centroids[j].y = sumY[j] / counts[j];
            }
        }
    }
}

// --- Sequential K-Means Algorithm (For Performance Comparison) ---
void sequentialKMeans(vector<Point>& points, int K, int iterations) {
    int N = points.size();
    vector<Point> centroids(K);

    for (int i = 0; i < K; i++) {
        centroids[i] = points[i];
    }

    for (int iter = 0; iter < iterations; iter++) {
        // Sequential assignment
        for (int i = 0; i < N; i++) {
            float min_dist = 1e9;
            int best_cluster = 0;
            for (int j = 0; j < K; j++) {
                float dist = calculateDistance(points[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            points[i].cluster = best_cluster;
        }

        vector<float> sumX(K, 0.0), sumY(K, 0.0);
        vector<int> counts(K, 0);

        for (int i = 0; i < N; i++) {
            int c = points[i].cluster;
            sumX[c] += points[i].x;
            sumY[c] += points[i].y;
            counts[c]++;
        }

        for (int j = 0; j < K; j++) {
            if (counts[j] > 0) {
                centroids[j].x = sumX[j] / counts[j];
                centroids[j].y = sumY[j] / counts[j];
            }
        }
    }
}

int main() {
    int N = 500000; // Total data points (Half a million)
    int K = 5;      // Number of clusters
    int iterations = 10; // Number of training epochs

    cout << "--- SPPU LP5: Practical 5 (HPC in AI/ML) ---" << endl;
    cout << "Algorithm: K-Means Clustering" << endl;
    cout << "Dataset:   " << N << " data points" << endl;
    cout << "Clusters:  " << K << "\n" << endl;

    // Generate random synthetic dataset
    vector<Point> dataset(N);
    srand(time(0));
    for (int i = 0; i < N; i++) {
        dataset[i].x = (float)(rand() % 1000);
        dataset[i].y = (float)(rand() % 1000);
    }

    // Create copies for fair comparison
    vector<Point> seq_data = dataset;
    vector<Point> par_data = dataset;
    double start_time, end_time;

    // 1. Execute Sequential ML Training
    cout << "Training Sequential Model..." << endl;
    start_time = omp_get_wtime();
    sequentialKMeans(seq_data, K, iterations);
    end_time = omp_get_wtime();
    cout << "Sequential Training Time : " << (end_time - start_time) << " seconds.\n" << endl;

    // 2. Execute Parallel ML Training
    cout << "Training Parallel Model..." << endl;
    start_time = omp_get_wtime();
    parallelKMeans(par_data, K, iterations);
    end_time = omp_get_wtime();
    cout << "Parallel Training Time   : " << (end_time - start_time) << " seconds.\n" << endl;

    cout << "ML Application execution complete." << endl;

    return 0;
}

/*

Install the missing pthreads library (Quickest Fix)

Open the Start Menu and search for MinGW Installation Manager (or go to C:\MinGW\bin\mingw-get.exe and run it).

In the left panel, click on Basic Setup or All Packages -> MinGW -> MinGW Base System.

Scroll through the list on the right and look for packages named:

mingw32-pthreads-w32 (Look for the dev and bin classes)

Right-click on them and select Mark for Installation.

Click on the Installation menu at the top left, and select Apply Changes.

Once it finishes downloading, close your VS Code/PowerShell completely, reopen it, and run your compile command again:

*/