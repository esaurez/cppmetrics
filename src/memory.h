#ifndef _MEMORY_H_INCLUDED
#define _MEMORY_H_INCLUDED

#include <chrono>
#include <thread>
#include <unordered_map>
#include <vector>

#include "statistics.hpp"
#include "utils.hpp"

namespace MetricsCpp{
  class MemoryMeasurement{
    public:
      enum MemoryMeasurementType {VIRTUAL_MEMORY,RESIDENT_SET};
      MemoryMeasurement();
      void saveMemoryCurrentMemoryMeasurement(std::string type="");
      void startSavingMemory(std::string type="",const std::chrono::milliseconds & sleep_duration=std::chrono::milliseconds(10), const std::chrono::milliseconds &waitBefore=std::chrono::milliseconds::zero());
      void stopSavingMemory();
      MemoryStatistics aggregateResults(MemoryMeasurementType memType, const std::string &type);
    private:
      std::atomic_bool measurementRunning;
      std::thread measurementThread;
      std::unordered_map<MemoryMeasurementType,std::unordered_map<std::string,MemoryStatistics>,EnumClassHash> currentMeasurements; 
    private:
      void getCurrentEstimatedMemoryUsage(double& vmUsage, double& residentSet);
      void memoryMeasurementThreadFunction(std::string type,const std::chrono::milliseconds &sleepDuration,const std::chrono::milliseconds &waitBefore);

  };
}
#endif

