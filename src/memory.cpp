#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <utility>

#include "memory.h"

namespace MetricsCpp{
  MemoryMeasurement::MemoryMeasurement(){
    this->currentMeasurements.emplace(std::make_pair(MemoryMeasurementType::VIRTUAL_MEMORY,std::unordered_map<std::string,MemoryStatistics>()));
    this->currentMeasurements.emplace(std::make_pair(MemoryMeasurementType::RESIDENT_SET,std::unordered_map<std::string,MemoryStatistics>()));
  }

  void MemoryMeasurement::getCurrentEstimatedMemoryUsage(double& vmUsage, double& residentSet){
    //Obtained from: [https://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-runtime-using-c]
    vmUsage     = 0.0;
    residentSet = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
      std::string ignore;
      std::ifstream ifs("/proc/self/stat", std::ios_base::in);
      ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
        >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
        >> ignore >> ignore >> vsize >> rss;
    }

    const long pagesSizeInKB= sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vmUsage = vsize / 1024.0;
    residentSet = rss * pagesSizeInKB;
  }

  void MemoryMeasurement::saveMemoryCurrentMemoryMeasurement(std::string type){
    double vmUsage,residentSet;
    getCurrentEstimatedMemoryUsage(vmUsage,residentSet);
    this->currentMeasurements.at(VIRTUAL_MEMORY)[type].addNewElement(vmUsage);
    this->currentMeasurements.at(RESIDENT_SET)[type].addNewElement(residentSet);
  }

  void MemoryMeasurement::memoryMeasurementThreadFunction(std::string type, const std::chrono::milliseconds &sleepDuration,const std::chrono::milliseconds &waitBefore){
    std::this_thread::sleep_for(waitBefore);
    while(measurementRunning){
      saveMemoryCurrentMemoryMeasurement(type);    
      std::this_thread::sleep_for(sleepDuration);
    }
  }

  void MemoryMeasurement::startSavingMemory(std::string type,const std::chrono::milliseconds & sleepDuration,const std::chrono::milliseconds &waitBefore){
    if(measurementThread.joinable()){
      throw std::runtime_error("Memory measurement class doesn't allow parallel measurements, there is one measurement already happening");
    }
    measurementRunning.store(true);
    measurementThread=std::thread(&MemoryMeasurement::memoryMeasurementThreadFunction,this,std::string(std::move(type)),sleepDuration,waitBefore);
  }

  void MemoryMeasurement::stopSavingMemory(){
    measurementRunning.store(false);
    if(!measurementThread.joinable()){
      throw std::runtime_error("Stop saving memory can only be called after calling start saving memory");
    }
    measurementThread.join();
  }

  
  MemoryStatistics MemoryMeasurement::aggregateResults(MemoryMeasurementType memType, const std::string &type){
    MemoryStatistics newStats;
    try{
      newStats = currentMeasurements.at(memType).at(type);
    }
    catch(std::out_of_range &e){
      throw std::out_of_range("There is no element of type ["+type+"] in the aggregated results"); 
    }
    newStats.aggregateResults();
    return newStats;

  }
}

