#ifndef _CPU_H_INCLUDED
#define _CPU_H_INCLUDED

#include <ctime>
#include <unordered_map>

#include "statistics.hpp"
#include "utils.hpp"

namespace MetricsCpp{
  class CpuMeasurement{
    public:
      CpuMeasurement() = default;
      template <class T>
        int startMeasurement(const T &inDummyValue){
          firstTime = std::clock();
          DoNotOptimize(inDummyValue);  
          return 0;
        }

      template <class T>
        int stopMeasurement(const T &outDummyValue,std::string config={}){
          DoNotOptimize(outDummyValue);  
          const auto second = std::clock();
          const auto timeSpan = (second- firstTime);  
          currentMeasurements[config].addNewElement(std::move(timeSpan));  
          return 0;
        }

     ClockStatistics getStatistics(const std::string &stats); 
      
      clock_t getTicksPerSecond() const; 

    private:
      clock_t firstTime;
      std::unordered_map<std::string,ClockStatistics> currentMeasurements; 
    private:

  };
}
#endif //CPU_H_INCLUDED

