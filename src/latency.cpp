#include "latency.hpp"
namespace MetricsCpp{

  TimeMeasurement::TimeMeasurement() : name("default"){
#ifdef GPU_CAPABLE
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif
  }

  TimeMeasurement::TimeMeasurement(const std::string &Name) : name(Name){
#ifdef GPU_CAPABLE
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif
  }

  TimeMeasurement::~TimeMeasurement(){
#ifdef GPU_CAPABLE
    // cudaEventDestroy(start);
    //  cudaEventDestroy(stop);
#endif
  }

  const TimeMeasurement::ms& TimeMeasurement::getPercentile(const std::vector<std::pair<std::string,ms>>& values, double percentile) const{
    const auto nth = values.cbegin() + (percentile*values.size())/100;
    return nth->second;
  }

  std::ostream& operator<<(std::ostream& os, const TimeMeasurement& stats){
    os<<stats.name<<","<<stats.aggregatedResults<<stats.currentMeasurements.size()<<",";
    return os;
  }

  TimeMeasurement::ms TimeMeasurement::popLastValue(){
    auto back = currentMeasurements.back(); 
    currentMeasurements.pop_back();
    return back.second;
  }

  int TimeMeasurement::addValue(TimeMeasurement::ms value, const std::string& config){
    this->currentMeasurements.push_back({config,std::move(value)});
    return 0;
  }

  int TimeMeasurement::getSize(){
    return this->currentMeasurements.size();
  }


  std::vector<std::string> TimeMeasurement::GetTimesWithConfig() const{
    std::vector<std::string> result;
    for(auto &&val:currentMeasurements){
      if(!val.first.empty()){
        result.push_back(val.first+","+std::to_string(val.second.count())); 
      }
    }
    return result;
  }

  TimeStatistics TimeMeasurement::aggregateResults(){
    if(!currentMeasurements.empty()){
      std::sort(currentMeasurements.begin(),currentMeasurements.end(),[](const std::pair<std::string,TimeStatistics::ms> &a,const std::pair<std::string,TimeStatistics::ms> &b){ return a.second < b.second; });

      TimeStatistics::ms totalFetch = std::accumulate(currentMeasurements.begin(),currentMeasurements.end(),ms::duration::zero(),
          [](ms current, const std::pair<std::string,TimeStatistics::ms> &a){ return current +  a.second; });
      TimeStatistics::ms average = totalFetch/currentMeasurements.size();
      TimeStatistics::ms min = currentMeasurements.begin()->second;
      TimeStatistics::ms max = currentMeasurements.rbegin()->second;
      TimeStatistics::ms percentile25 = getPercentile(currentMeasurements,25.0);
      TimeStatistics::ms median = getPercentile(currentMeasurements,50.0);
      TimeStatistics::ms percentile75= getPercentile(currentMeasurements,75.0);
      TimeStatistics::ms percentile90= getPercentile(currentMeasurements,90.0);
      TimeStatistics::ms percentile99= getPercentile(currentMeasurements,99.0);
      TimeStatistics stats(average, min, max, percentile25, median, percentile75, percentile90, percentile99);
      this->aggregatedResults=stats;
      return stats;
    }
    TimeStatistics stats;
    return stats;
  }

  TimeStatistics TimeMeasurement::aggregateResults(const std::string &type){
    if(!currentMeasurements.empty()){
      std::vector<std::pair<std::string,ms>> filtered;
      std::copy_if (currentMeasurements.begin(), currentMeasurements.end(), std::back_inserter(filtered), [&type](const std::pair<std::string,ms> &e){
          return e.first==type;
          });
      std::sort(filtered.begin(),filtered.end(),[](const std::pair<std::string,TimeStatistics::ms> &a,const std::pair<std::string,TimeStatistics::ms> &b){ return a.second < b.second; });

      TimeStatistics::ms totalFetch = std::accumulate(filtered.begin(),filtered.end(),ms::duration::zero(),
          [](ms current, const std::pair<std::string,TimeStatistics::ms> &a){ return current +  a.second; });
      TimeStatistics::ms average = totalFetch/filtered.size();
      TimeStatistics::ms min = filtered.cbegin()->second;
      TimeStatistics::ms max = filtered.crbegin()->second;
      TimeStatistics::ms percentile25 = getPercentile(filtered,25.0);
      TimeStatistics::ms median = getPercentile(filtered,50.0);
      TimeStatistics::ms percentile75= getPercentile(filtered,75.0);
      TimeStatistics::ms percentile90= getPercentile(filtered,90.0);
      TimeStatistics::ms percentile99= getPercentile(filtered,99.0);
      TimeStatistics stats(average, min, max, percentile25, median, percentile75, percentile90, percentile99);
      return stats;
    }
    TimeStatistics stats;
    return stats;
  }
}
