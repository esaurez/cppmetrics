#include <iostream> 

#include "cpu.h"
#include "latency.hpp"
#include "memory.h"

double functionToTest(double input){
  double result = 1.0;
  while(input>1.0){
    result*=input;
    input-=1.0;
  }
  return result;
}

double heavyMemory(double input){
  std::vector<double> partialResults;
  double result = 1.0;
  while(input>1.0){
    result*=input;
    input-=1.0;
    partialResults.push_back(result);
  }
  return result;
}



void measureLatency(){
  MetricsCpp::TimeMeasurement timeMeasurement;
  double input = 10000000;
  for(int i=0; i<20; ++i){
    timeMeasurement.startMeasurement(input);
    double output = functionToTest(input+i);
    timeMeasurement.stopMeasurement(output,"testlatency");
  }
  std::cout<< "LATENCY:"<<timeMeasurement.aggregateResults("testlatency")<<std::endl;;
}

void measureLongLatency(){
  MetricsCpp::TimeMeasurement timeMeasurement;
  double input = 10000000;
  for(int i=0; i<20; ++i){
    timeMeasurement.startMeasurement(input);
    double output = heavyMemory(input+i);
    timeMeasurement.stopMeasurement(output,"testlatency");
  }
  std::cout<< "LONG LATENCY:"<<timeMeasurement.aggregateResults("testlatency")<<std::endl;;
}



void measureMemory(){
  MetricsCpp::MemoryMeasurement memoryMeasurement;
  double input = 200000000;
  memoryMeasurement.startSavingMemory("testMemory",std::chrono::milliseconds(10),std::chrono::milliseconds(1200));
  double output = heavyMemory(input);
  memoryMeasurement.stopSavingMemory();
  const auto rss = memoryMeasurement.aggregateResults(MetricsCpp::MemoryMeasurement::MemoryMeasurementType::RESIDENT_SET,"testMemory");
  const auto vm = memoryMeasurement.aggregateResults(MetricsCpp::MemoryMeasurement::MemoryMeasurementType::VIRTUAL_MEMORY,"testMemory");
  std::cout << "RESIDENT SET: "<<rss <<std::endl;
  std::cout << "VIRTUAL MEMORY: "<<vm <<std::endl;
}

void measureCpu(){
  MetricsCpp::CpuMeasurement cpuMeasurement;
  double input = 10000000;
  for(int i=0; i<20; ++i){
    cpuMeasurement.startMeasurement(input);
    double output = functionToTest(input+i);
    cpuMeasurement.stopMeasurement(output,"testlatency");
  }
  const auto stats = cpuMeasurement.getStatistics("testlatency");
  std::cout<<"CPU MEASUREMENT: "<<stats<<std::endl;
}

void measureMemoryMeasurementLatency(){
  MetricsCpp::MemoryMeasurement memoryMeasurement;
  MetricsCpp::TimeMeasurement timeMeasurement;
  double input = 10000000;
  for(int i=0; i<20; ++i){
    std::string type = "type"+std::to_string(input+i);
    timeMeasurement.startMeasurement(type);
    memoryMeasurement.saveMemoryCurrentMemoryMeasurement(type);
    timeMeasurement.stopMeasurement(memoryMeasurement,"testlatency");
  }
  std::cout<< "MEMORY MEASUREMENT LATENCY:"<<timeMeasurement.aggregateResults("testlatency")<<std::endl;;
}


int main(int argc, char *argv[]){
  measureLatency();
  measureCpu();
  measureMemory();
  measureLongLatency();
  measureMemoryMeasurementLatency();
}
