#ifndef _LATENCYLIB_H_INCLUDED
#define _LATENCYLIB_H_INCLUDED

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>
#ifdef GPU_CAPABLE 
#include <cuda_runtime.h>
#endif

#include "statistics.hpp"
#include "utils.hpp"


namespace MetricsCpp{
  /*
   * \Note: This class is not thread-safe!!!
   * */
  class TimeMeasurement{
    public:
      typedef std::chrono::microseconds ms;
      TimeMeasurement();
      TimeMeasurement(const std::string &name);
      ~TimeMeasurement();
      template <class T>
        int startMeasurement(const T &inDummyValue){
          firstTime = std::chrono::high_resolution_clock::now();
          DoNotOptimize(inDummyValue);  
          return 0;
        }
      template <class T>
        int stopMeasurement(const T &outDummyValue,std::string config={}){
          DoNotOptimize(outDummyValue);  
          const auto second = std::chrono::high_resolution_clock::now();
          const auto timeSpan = std::chrono::duration_cast<ms>(second- firstTime);  
          currentMeasurements.push_back(std::make_pair(std::move(config),std::move(timeSpan)));  
          return 0;
        }
#ifdef GPU_CAPABLE
      template <class T>
        int startGpuMeasurement(const T &inDummyValue){
          cudaEventRecord(start,0);
          DoNotOptimize(inDummyValue);  
          return 0;
        }
      template <class T>
        int stopGpuMeasurement(const T &outDummyValue, const std::string& config=""){
          DoNotOptimize(outDummyValue);  
          cudaEventRecord(stop,0);
          float elapsed;
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&elapsed, start, stop);
          //elapsed is in milliseconds
          std::chrono::microseconds timeSpan((static_cast<int>(elapsed*1000)));
          currentMeasurements.push_back({config,timeSpan});  
          return 0;
        }
#endif
      ms popLastValue(); 
      int addValue(ms value, const std::string& config="");
      int getSize();
      TimeStatistics aggregateResults();
      TimeStatistics aggregateResults(const std::string &type);
      friend std::ostream& operator<<(std::ostream& os, const TimeMeasurement& stats);
      std::vector<std::string> GetTimesWithConfig() const;
    private:
      const ms& getPercentile(const std::vector<std::pair<std::string,ms>>& values, double percentile) const;
      TimeStatistics aggregatedResults;
    private:
      const std::string name;
      std::chrono::high_resolution_clock::time_point firstTime;
      std::vector<std::pair<std::string,ms>> currentMeasurements; 
#ifdef GPU_CAPABLE
      cudaEvent_t start;
      cudaEvent_t stop;
#endif

  };
}

#endif //LATENCYLUB_H_INCLUDED_
