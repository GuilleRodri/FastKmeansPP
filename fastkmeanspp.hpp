#ifndef FASTKMEANSPP_HPP
#define FASTKMEANSPP_HPP

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>
#include <sys/resource.h>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

namespace FastKMeansPP {

  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  
  struct Point { double ub; int id; };
  struct Cluster { double radius, sum; vector<Point> points; };
  
  struct PointN { double ub, norm; int id; };
  struct ClusterN { double radius, upperNorm, lowerNorm, sum; };
  
  Matrix readFile(const std::string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    if (!file) throw runtime_error("Cannot open file: " + filename);

    string line;
    while (getline(file, line)) {
      // Skip empty lines
      if (line.find_first_not_of(" \t\r\n") == string::npos) continue;

      // Decide delimiter (comma or tab). Default: comma if present, else tab.
      char delim = '\t';
      if (line.find(',') != string::npos) delim = ',';

      vector<double> row;
      string cell;
      stringstream lineStream(line);

      while (getline(lineStream, cell, delim)) {
        // Trim whitespace
        auto start = cell.find_first_not_of(" \t\r\n");
        auto end   = cell.find_last_not_of(" \t\r\n");
        if (start == string::npos) continue;
        string token = cell.substr(start, end - start + 1);

        row.push_back(stod(token));
      }

      if (!row.empty()) data.push_back(std::move(row));
    }

    if (data.empty()) throw runtime_error("No data read from file: " + filename);

    int nSamples = (int)data.size();
    int nDimensions = (int)data[0].size();
    for (int i = 1; i < nSamples; ++i) {
      if ((int)data[i].size() != nDimensions) {
        throw runtime_error("Inconsistent number of columns in: " + filename);
      }
    }

    Matrix M(nSamples, nDimensions);
    for (int i = 0; i < nSamples; ++i) {
      M.row(i) = Eigen::Map<Eigen::VectorXd>(data[i].data(), nDimensions);
    }
    return M;
  }


  namespace Internal {

    inline int selectCentroid(const std::vector<double>& weights, double randomValue) {
      double cumulative = 0.0;
      for (size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i];
        if (randomValue <= cumulative) return static_cast<int>(i);
      }
      return static_cast<int>(weights.size() - 1);
    }

    inline double updatePartition(FastKMeansPP::ClusterN& P, vector<FastKMeansPP::PointN>& points){
      double radius = 0.0;
      double upperNorm = 0.0;
      double lowerNorm = numeric_limits<double>::max();
      double sum = 0.0;
      for (auto& point : points) {
        double currentED = sqrt(point.ub); 
        if (point.ub > radius) radius = point.ub;
        double tmp = point.norm + currentED;
        if (tmp > upperNorm) upperNorm = tmp;
        tmp = point.norm - currentED;
        if (tmp < lowerNorm) lowerNorm = tmp;
        sum += point.ub;
        point.ub = sum;
      }
      P.radius = radius;
      P.upperNorm = upperNorm;
      P.lowerNorm = lowerNorm;
      P.sum = sum;
      return sum;
}


  } // namespace Internal

  // Standard KMeans++ initialization
  double KMeansPlusPlus(const Matrix& M, Matrix& centers, int nClusters, unsigned seed = 0) {
  
    const int n = M.rows();
    
    // Random number generation setup
    mt19937 gen(seed);
    uniform_int_distribution<> pick(0, n - 1);
    uniform_real_distribution<> u(0, 1);

    centers = Matrix(nClusters, M.cols());
    vector<double> d(n, numeric_limits<double>::infinity());
    vector<bool> isCenter(n, false);
    double score = 0.0;

    // First center
    int idx0 = pick(gen);
    centers.row(0) = M.row(idx0);
    isCenter[idx0] = true;
    for (int i = 0; i < n; ++i) if (i != idx0)
      d[i] = (M.row(i) - centers.row(0)).squaredNorm();

    // Remaining centers
    for (int k = 1; k < nClusters; ++k) {
      score = accumulate(d.begin(), d.end(), 0.0);
      int next = Internal::selectCentroid(d, u(gen) * score);
      centers.row(k) = M.row(next);
      isCenter[next] = true;
      d[next] = 0.0;
      for (int i = 0; i < n; ++i) if (!isCenter[i]) {
        double dist = (M.row(i) - centers.row(k)).squaredNorm();
        d[i] = min(d[i], dist);
      }
    }
    return accumulate(d.begin(), d.end(), 0.0);
  }

  // Accelerated KMeans++ (TIE)
  double AcceleratedKMeansPlusPlusTIE(const Matrix& M, Matrix& centers, int nClusters, unsigned seed = 0) {
    int nSamples = M.rows();
    int nDimensions = M.cols();
    
    // Random number generation setup
    mt19937 gen(seed);
    uniform_real_distribution<> u(0,1);
    uniform_int_distribution<> pick(0, nSamples - 1);

    centers = Matrix(nClusters, nDimensions);
    vector<Cluster> clusters(nClusters);

    // Select first centroid randomly
    int newCenterID = pick(gen);
    centers.row(0) = M.row(newCenterID);

    double radius = 0.0;
    double totalSum = 0.0;
    clusters[0].points.reserve(nSamples);
    for (int i = 0; i < nSamples; ++i) {
      if (i == newCenterID) continue;
      double currentSED = (centers.row(0) - M.row(i)).squaredNorm();
      if (currentSED > radius) radius = currentSED;
      totalSum += currentSED;
      clusters[0].points.push_back({totalSum,i});
    }
    clusters[0].radius = radius;
    clusters[0].sum = totalSum;

    vector<double> cumSum(nClusters, 0.0);
    cumSum[0] = totalSum;
  
    for (int j = 1; j < nClusters; ++j) {
  
      // Select next center
      double randomValue = u(gen) * totalSum;
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
                clusters[j].points.push_back({sum_j, pointID});
                                
              } else {
                if (bestSED > radius_k) radius_k = bestSED;
                sum_k += bestSED;
                newPointList.push_back({sum_k, pointID});
              }
              
            } else {  
              if (bestSED > radius_k) radius_k = bestSED;
              sum_k += bestSED;
              newPointList.push_back({sum_k, pointID});
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
    
    return totalSum;
  }
  
  

  // Accelerated KMeans++ (TIE + Norm)
  double AcceleratedKMeansPlusPlusTIENorm(const Matrix& M, Matrix& centers, int nClusters, unsigned seed = 0) {

    int nSamples = M.rows();
    int nDimensions = M.cols();
    int nPartitions = nClusters << 1;
    
    // Random number generation setup
    mt19937 gen(seed);
    uniform_real_distribution<> u(0,1);
    uniform_int_distribution<> pick(0, nSamples - 1);

    centers = Matrix(nClusters, nDimensions);
    vector<ClusterN> partitions(nPartitions);
    vector<vector<PointN>> points(nPartitions);

    // Select first centroid randomly
    int newCenterID = pick(gen);
    centers.row(0) = M.row(newCenterID);
    double centerNorm = centers.row(0).norm();
  
    auto& points_0 = points[0];
    points_0.reserve(nSamples-1);
    auto& points_1 = points[1];
    points_1.reserve(nSamples-1);
  
    for (int i = 0; i < nSamples; ++i) {
      if (i == newCenterID) continue;
      const auto& point = M.row(i);
      double currentNorm = point.norm();
      double currentSED = (centers.row(0) - point).squaredNorm();
      if (currentNorm <= centerNorm) points_0.push_back({currentSED, currentNorm, i});  
      else points_1.push_back({currentSED, currentNorm, i});  
    }

    vector<double> cumSum(nPartitions, 0.0);
    double totalSum = Internal::updatePartition(partitions[0], points_0);
    cumSum[0] = totalSum;
    totalSum += Internal::updatePartition(partitions[1], points_1);
    cumSum[1] = totalSum;

    int pj = 2;
    for (int j = 1; j < nClusters; ++j) {
  
      // Select next center
      double randomValue = u(gen) * totalSum;
      auto it = lower_bound(cumSum.begin(), cumSum.begin() + pj, randomValue);
      int nextPartition = distance(cumSum.begin(), it);
      if (nextPartition > 0) randomValue -=  cumSum[nextPartition-1];
    
      const auto& selectedPoints = points[nextPartition];
      auto itp = lower_bound(selectedPoints.begin(), selectedPoints.end(), randomValue, 
                 [](const PointN& point, double value) {return point.ub < value;});
      newCenterID = itp->id;
      centerNorm = itp->norm;
      centers.row(j) = M.row(newCenterID);

      // Calculate Pairwise Distances between centers
      const auto squaredDistances = (centers.topRows(j).rowwise() - centers.row(j)).rowwise().squaredNorm()/4.0;
    
      auto& points_pj0 = points[pj];
      points_pj0.reserve(nSamples);
      auto& points_pj1 = points[pj+1];
      points_pj1.reserve(nSamples);
    
      int pk = 0;
      totalSum = 0.0;
      for (int k = 0;  k < j; ++k) {
    
        double dist = squaredDistances(k);
     
        // Lower Partition
        auto& partition_pk0 = partitions[pk];
      
        if  (partition_pk0.radius > dist  && centerNorm < partition_pk0.upperNorm && centerNorm > partition_pk0.lowerNorm) {
      
          auto& points_pk0 = points[pk];
          vector<PointN> newPointList;
          newPointList.reserve(points_pk0.size());
          double cumSED = 0.0;
          for (const auto& point : points_pk0) {    
            int pointID = point.id;
            double bestSED = point.ub - cumSED;
            cumSED = point.ub;          
            if (pointID == newCenterID) continue;         
            double currentNorm = point.norm;
            double currentDiff = currentNorm - centerNorm;   
            if (bestSED > dist && bestSED > currentDiff * currentDiff) {  
              double currentSED =  (centers.row(j) - M.row(pointID)).squaredNorm();                 
              if (currentSED < bestSED) {   
                if (currentNorm <= centerNorm) points_pj0.push_back({currentSED, currentNorm, pointID});  
                else points_pj1.push_back({currentSED, currentNorm, pointID});                                        
              }
              else newPointList.push_back({bestSED, currentNorm, pointID});     
            }
            else newPointList.push_back({bestSED, currentNorm, pointID});  
          }
          points_pk0 = move(newPointList);
          totalSum += Internal::updatePartition(partition_pk0, points_pk0);
        }
        else totalSum += partition_pk0.sum;
      
        cumSum[pk] = totalSum;
        pk++;
      
        // Upper Partition
        auto& partition_pk1 = partitions[pk];
      
        if  (partition_pk1.radius > dist  && centerNorm < partition_pk1.upperNorm && centerNorm > partition_pk1.lowerNorm) {
      
          auto& points_pk1 = points[pk];
          vector<PointN> newPointList;
          newPointList.reserve(points_pk1.size());
          double cumSED = 0.0;
          for (const auto& point : points_pk1) {    
            int pointID = point.id;
            double bestSED = point.ub - cumSED;
            cumSED = point.ub;          
            if (pointID == newCenterID) continue;         
            double currentNorm = point.norm; 
            double currentDiff = currentNorm - centerNorm; 
            if (bestSED > dist && bestSED > currentDiff * currentDiff) {                  
              double currentSED =  (centers.row(j) - M.row(pointID)).squaredNorm();                    
              if (currentSED < bestSED) {
                if (currentNorm <= centerNorm) points_pj0.push_back({currentSED, currentNorm, pointID});  
                else points_pj1.push_back({currentSED, currentNorm, pointID});                                        
              }
              else newPointList.push_back({bestSED, currentNorm, pointID});     
            }
            else newPointList.push_back({bestSED, currentNorm, pointID});  
          }
        
          points_pk1 = move(newPointList);
          totalSum += Internal::updatePartition(partition_pk1, points_pk1);
        }
        else totalSum += partition_pk1.sum;
        cumSum[pk] = totalSum;
        pk++;
      }
    
      totalSum += Internal::updatePartition(partitions[pj], points_pj0);
      cumSum[pj] = totalSum;
      pj++;
      totalSum += Internal::updatePartition(partitions[pj], points_pj1);
      cumSum[pj] = totalSum;
      pj++;
    }
    
    return totalSum;
  }

// General function interface
double run_kmeanspp(const Matrix& data, Matrix& centers, int k, unsigned seed = 0, int acceleration = 0) {
    switch (acceleration) {
        case 0:
            return KMeansPlusPlus(data, centers, k, seed); // Standard k-means++
        case 1:
            return AcceleratedKMeansPlusPlusTIE(data, centers, k, seed); // Accelerated with TIE
        case 2:
            return AcceleratedKMeansPlusPlusTIENorm(data, centers, k, seed); // Further accelerated with norm-based filter
        default:
            throw invalid_argument("Invalid acceleration mode. Use 0 (Standard), 1 (TIE), or 2 (TIE + Norm).");
    }
}

double run_kmeanspp(const string& filename, Matrix& centers,int k, unsigned seed = 0, int acceleration = 0) {
    Matrix M = readFile(filename);
    return run_kmeanspp(M, centers, k, seed, acceleration);
}

} // namespace FastKMeansPP

#endif // FASTKMEANSPP_HPP
