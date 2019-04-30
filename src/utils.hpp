#ifndef _UTILS_H_INCLUDED
#define _UTILS_H_INCLUDED
namespace MetricsCpp{
  //Taken from: https://stackoverflow.com/questions/37786547/enforcing-statement-order-in-c
  template <class T>
    __attribute__((always_inline)) inline void DoNotOptimize(const T &value) {
      asm volatile("" : "+m"(const_cast<T &>(value)));
    }

  //Taken from https://stackoverflow.com/questions/18837857/cant-use-enum-class-as-unordered-map-key
  struct EnumClassHash
  {
    template <typename T>
      std::size_t operator()(T t) const
      {
        return static_cast<std::size_t>(t);
      }
  };
}
#endif

