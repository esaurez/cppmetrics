#ifndef _STATISTICS_H_INCLUDED
#define _STATISTICS_H_INCLUDED

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

namespace MetricsCpp{
  template <class DataType, class AverageType>
    class Statistics{
      public:
        AverageType average;
        DataType min,max,percentile25,median,percentile75,percentile90,percentile99,percentile999,percentile9999;
        int aggregateResults(){
          if(!elements.empty()){
            std::sort(elements.begin(),elements.end());
            average = std::accumulate(elements.begin(),elements.end(),DataType(0))/elements.size();
            min = *elements.cbegin();
            max = *elements.crbegin();
            percentile25 = getPercentile(25.0);
            median = getPercentile(50.0);
            percentile75= getPercentile(75.0);
            percentile90= getPercentile(90.0);
            percentile99= getPercentile(99.0);
            percentile999= getPercentile(99.9);
            percentile9999= getPercentile(99.99);
          }
          return 0;
        }

        friend std::ostream& operator<<(std::ostream& os, const Statistics& ss) {
          os<<ss.average<<",";
          os<<ss.min<<",";
          os<<ss.max<<",";
          os<<ss.percentile25<<",";
          os<<ss.median<<",";
          os<<ss.percentile75<<",";
          os<<ss.percentile90<<",";
          os<<ss.percentile99<<",";
          os<<ss.percentile999<<",";
          os<<ss.percentile9999<<",";
          os<<ss.elements.size()<<",";
          return os;
        } 

        int addNewElement(DataType element){
          elements.emplace_back(std::move(element));
          return 0;
        }
        Statistics& operator=(const Statistics& other){
          if (this != &other) { // self-assignment check expected
            std::copy(other.elements.begin(),other.elements.end(), std::back_inserter(elements));
          }
          return *this;
        }
      private:
        std::vector<DataType> elements;
          DataType getPercentile(double percentile) const{
          const int position = (percentile*elements.size())/100;
          return elements.at(position);
        }
    };

  template <class DataType, class AverageType>
    class DurationStatistics{
      public:
        AverageType average;
        DataType min,max,percentile25,median,percentile75,percentile90,percentile99;
        int aggregateResults(){
          if(!elements.empty()){
            std::sort(elements.begin(),elements.end());
            average = std::accumulate(elements.begin(),elements.end(),DataType(0))/DataType(elements.size());
            min = *elements.cbegin();
            max = *elements.crbegin();
            percentile25 = getPercentile(25.0);
            median = getPercentile(50.0);
            percentile75= getPercentile(75.0);
            percentile90= getPercentile(90.0);
            percentile99= getPercentile(99.0);
          }
          return 0;
        }

        friend std::ostream& operator<<(std::ostream& os, const DurationStatistics& ss) {
          if(ss.elements.size()==0){
            os<<"0,0,0,0,0,0,0,0,0,";
          }
          else{
            os<<ss.average<<",";
            os<<ss.min.count()<<",";
            os<<ss.max.count()<<",";
            os<<ss.percentile25.count()<<",";
            os<<ss.median.count()<<",";
            os<<ss.percentile75.count()<<",";
            os<<ss.percentile90.count()<<",";
            os<<ss.percentile99.count()<<",";
            os<<ss.elements.size()<<",";
          }
          return os;
        } 

        int addNewDuration(DataType size){
          elements.push_back(size);
          return 0;
        }
      private:
        std::vector<DataType> elements;
        DataType getPercentile(double percentile) const{
          const int position = (percentile*elements.size())/100;
          return elements.at(position);
        }
    };

  class TimeStatistics{
    public:
      typedef std::chrono::microseconds ms;
      TimeStatistics() : average(0),min(0),max(0),percentile25(0),median(0),percentile75(0),percentile90(0),percentile99(0){}; 
      TimeStatistics(const ms &avg, const ms &mini, const ms &maxi, const ms &per25, const ms &med, const ms &per75, const ms &per90, const ms &per99, const std::string &config="") : average(avg), min(mini), max(maxi), percentile25(per25), median(med), percentile75(per75), percentile90(per90), percentile99(per99),configuration(config) {};
      TimeStatistics& operator=(const TimeStatistics& other); 
      ms average;
      ms min;
      ms max;
      ms percentile25;
      ms median;
      ms percentile75;
      ms percentile90;
      ms percentile99;
      std::string configuration;
      friend std::ostream& operator<<(std::ostream& os, const TimeStatistics& dt);  
  };

  using SizeStatistics = Statistics<int,double>;
  using MemoryStatistics = Statistics<double,double>;
  using ClockStatistics = Statistics<clock_t,clock_t>;
}
#endif
