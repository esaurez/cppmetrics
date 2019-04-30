#include "statistics.hpp"

namespace MetricsCpp{
  TimeStatistics& TimeStatistics::operator=(const TimeStatistics& other){
    if(this!=&other){
      this->average = other.average;
      this->min = other.min;
      this->max = other.max;
      this->percentile25 = other.percentile25;
      this->median = other.median;
      this->percentile75 = other.percentile75;
      this->percentile90 = other.percentile90;
      this->percentile99 = other.percentile99;
      this->configuration = other.configuration;
    }
    return *this;
  }

  std::ostream& operator<<(std::ostream& os, const TimeStatistics& stats){
    os<<stats.average.count()<<",";
    os<<stats.min.count()<<",";
    os<<stats.max.count()<<",";
    os<<stats.percentile25.count()<<",";
    os<<stats.median.count()<<",";
    os<<stats.percentile75.count()<<",";
    os<<stats.percentile90.count()<<",";
    os<<stats.percentile99.count()<<",";
    return os;
  }
}

