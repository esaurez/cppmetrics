#include "cpu.h"
namespace MetricsCpp{
  clock_t CpuMeasurement::getTicksPerSecond() const{
    return CLOCKS_PER_SEC;
  }


  ClockStatistics CpuMeasurement::getStatistics(const std::string &statType){
    ClockStatistics stats;
    try{
      stats= currentMeasurements.at(statType);
    }
    catch(std::out_of_range& e){
      throw std::out_of_range("Statistic doesn't exist in measurement"); 
    }
    stats.aggregateResults();
    return stats;
  }
}
