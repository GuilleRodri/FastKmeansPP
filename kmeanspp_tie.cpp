#include <iostream>
#include <string>
#include <filesystem>
#include <utility>
#include <numeric>
#include <cmath>
#include <limits>
#include <random>
#include <set>
#include <queue>
#include <iterator>
#include <chrono>
#include <fstream>
#include <deque>
#include <cstring>
#include <Eigen/Dense>
#include <ctime>
#include <time.h>
#include <sys/resource.h>

using namespace std;
using Eigen::VectorXd;
using Eigen::Map;

namespace fs = filesystem;
int inputNClusters = 4096;
int inputSeed = 0;

string inputDataset;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix;


void read_parameters(int argc, char **argv) {
  int iarg = 1;
  while (iarg < argc) {
    if (strcmp(argv[iarg],"-input")==0) inputDataset = (argv[++iarg]);
    if (strcmp(argv[iarg],"-n_clusters")==0) inputNClusters = atoi(argv[++iarg]);
    if (strcmp(argv[iarg],"-seed")==0) inputSeed = atoi(argv[++iarg]);
    iarg++;
  }
}


template <typename T>
void printVector(const vector<T>& vec) {
  for (const auto& element : vec) {
    cout << element << " ";
  }
  cout << endl;
}


template <typename T>
void printMatrix(const vector<vector<T>>& mat) {
  for (const auto& vec : mat) {
    printVector(vec);
  }
}


matrix readFile(const string& filename) {
  vector<vector<double>> data;
  ifstream file(filename);

  string line;
  while (getline(file, line)) {
    vector<double> row;
    stringstream lineStream(line);
    string cell;

    while (getline(lineStream, cell, ',')) {
      row.push_back(stod(cell));
    }

    data.push_back(row);
  }
  
  int nSamples = data.size();
  int nDimensions = data[0].size();
  matrix M(nSamples, nDimensions);

  for (int i = 0; i < nSamples; ++i) {
    M.row(i) = Eigen::Map<Eigen::VectorXd>(data[i].data(), data[i].size());
  }
  
  return M;
}



// Accelerated
struct Point {
  double ub;
  int id;
};

struct Cluster {
  double radius;
  double sum;
  vector<Point> points;
};


// Accelerated KMeans++
void acceleratedKmeansPlusPlus_TIE(const matrix& M, int nClusters, int seed) {

  // Start timing
  struct rusage usageStart, usageEnd;
  getrusage(RUSAGE_SELF, &usageStart);

  int nSamples = M.rows();
  int nDimensions = M.cols();

  matrix centers(nClusters, nDimensions);
  vector<Cluster> clusters(nClusters);

  // Random number generation setup
  random_device rd;
  mt19937 gen(seed);
  uniform_real_distribution<> distrib_double(0.0, 1.0);
  uniform_int_distribution<> distrib_int(0, nSamples - 1);
    
  // Select first centroid randomly
  int newCenterID = distrib_int(gen);
  centers.row(0) = M.row(newCenterID);

  double radius = 0.0;
  double totalSum = 0.0;
  clusters[0].points.reserve(nSamples);
  for (int i = 0; i < nSamples; ++i) {
    if (i == newCenterID) continue;
    double currentSED = (centers.row(0) - M.row(i)).squaredNorm();
    if (currentSED > radius) radius = currentSED;
    totalSum += currentSED;
    clusters[0].points.emplace_back(totalSum,i);
  }
  clusters[0].radius = radius;
  clusters[0].sum = totalSum;

  vector<double> cumSum(nClusters, 0.0);
  cumSum[0] = totalSum;
  
  for (int j = 1; j < nClusters; ++j) {
  
    // Select next center
    double randomValue = distrib_double(gen) * totalSum;
    auto it = lower_bound(cumSum.begin(), cumSum.begin()+j, randomValue);
    int nextCluster = distance(cumSum.begin(),it);
    if (nextCluster > 0) randomValue -=  cumSum[nextCluster-1];
    
    const auto& selectedPoints = clusters[nextCluster].points;
    auto itp = lower_bound(selectedPoints.begin(), selectedPoints.end(), randomValue, 
               [](const Point& point, double value) {return point.ub < value;});
    newCenterID = itp->id;
    centers.row(j) = M.row(newCenterID);
    
    // Calculate Pairwise Distances between centers
    const auto squaredDistances = (centers.topRows(j).rowwise() - centers.row(j)).rowwise().squaredNorm()/4.0;

    clusters[j].points.clear();
    clusters[j].points.reserve((unsigned)nSamples/j);

    double radius_j = 0.0;
    double sum_j = 0.0;
    totalSum = 0.0;
    
    for (int k = 0; k < j; ++k) {
      
      double dist = squaredDistances(k);
      
      if (clusters[k].radius > dist) {
        
        double radius_k = 0.0;
        double sum_k = 0.0;
        double cumSED = 0.0;
        vector<Point> newPointList;
        newPointList.reserve(clusters[k].points.size());
        
        for (const auto& point : clusters[k].points) {
        
          int pointID = point.id;
          double bestSED = point.ub - cumSED;
          cumSED = point.ub;
                
          if (pointID == newCenterID) continue;
         
          if (bestSED > dist) {
                        
            double currentSED =  (centers.row(j) - M.row(pointID)).squaredNorm();                
            if (currentSED < bestSED) {
              if (currentSED > radius_j) radius_j = currentSED; 
              sum_j += currentSED;
              clusters[j].points.emplace_back(sum_j, pointID);
                                
            } else {
              if (bestSED > radius_k) radius_k = bestSED;
              sum_k += bestSED;
              newPointList.emplace_back(sum_k, pointID);
            }
              
          } else {  
            if (bestSED > radius_k) radius_k = bestSED;
            sum_k += bestSED;
            newPointList.emplace_back(sum_k, pointID);
          }
        }
        clusters[k].radius = radius_k;
        clusters[k].sum = sum_k;
        clusters[k].points = move(newPointList);
      }
      totalSum += clusters[k].sum;
      cumSum[k] = totalSum;
    }
    clusters[j].radius = radius_j;
    clusters[j].sum = sum_j;  
    totalSum += sum_j;
    cumSum[j] = totalSum;
  }
  
  // End timing
  getrusage(RUSAGE_SELF, &usageEnd);

  // Calculate and return the difference in user and system time
  double startTime = (double)usageStart.ru_utime.tv_sec + (double)usageStart.ru_utime.tv_usec * 1.0E-6 +
                       (double)usageStart.ru_stime.tv_sec + (double)usageStart.ru_stime.tv_usec * 1.0E-6;
  double endTime = (double)usageEnd.ru_utime.tv_sec + (double)usageEnd.ru_utime.tv_usec * 1.0E-6 +
                     (double)usageEnd.ru_stime.tv_sec + (double)usageEnd.ru_stime.tv_usec * 1.0E-6;

  cout << "time " << endTime - startTime << endl;
  cout << "score " << totalSum << endl;
  cout << centers << endl;
}



/* ---------- MAIN ---------- */

int main(int argc, char **argv) {

  if (argc < 3) {
    cout << "Use: main -input <input dataset> ..." << endl;
    exit(1);
  }
  else read_parameters(argc,argv);

  auto M = readFile(inputDataset);
  
  cout << fixed << setprecision(6) << "n " << M.rows() << " d " << M.cols() << " k " << inputNClusters << " s " << inputSeed << endl;
  acceleratedKmeansPlusPlus_TIE(M, inputNClusters, inputSeed);
  
  return 0;
}


