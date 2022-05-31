#include <ubu/coordinate/point.hpp>
#include <ubu/coordinate/size.hpp>
#include <tuple>
#include <utility>

namespace ns = ubu;


struct has_static_size_member_function
{
  constexpr static std::size_t size()
  {
    return 1;
  }
};


struct has_size_member_function
{
  constexpr static std::size_t size()
  {
    return 1;
  }
};

struct has_size_free_function {};

constexpr std::size_t size(const has_size_free_function&)
{
  return 1;
}

void test_size()
{
  static_assert(1 == ns::size_v<has_static_size_member_function>);

  static_assert(1 == ns::size(has_size_member_function{}));
  static_assert(1 == ns::size(has_size_free_function{}));

  static_assert(1 == ns::size_v<int>);
  static_assert(1 == ns::size(13));
  static_assert(1 == ns::size_v<unsigned int>);
  static_assert(1 == ns::size(13u));
  static_assert(1 == ns::size_v<std::size_t>);
  static_assert(1 == ns::size(std::size_t{13}));
  static_assert(1 == ns::size_v<std::tuple<char>>);
  static_assert(1 == ns::size(std::make_tuple('a')));
  static_assert(1 == ns::size_v<ns::point<int,1>>);
  static_assert(1 == ns::size(ns::point<int,1>{13}));

  static_assert(1 == ns::size_v<int&>);
  static_assert(1 == ns::size_v<const unsigned int&>);
  static_assert(1 == ns::size_v<std::size_t&&>);
  static_assert(1 == ns::size_v<const std::tuple<char>&>);

  static_assert(1 == ns::size_v<float>);
  static_assert(1 == ns::size(3.14f));
  static_assert(1 == ns::size_v<double>);
  static_assert(1 == ns::size(3.14));
  static_assert(1 == ns::size_v<std::tuple<float>>);
  static_assert(1 == ns::size(std::make_tuple(3.14f)));
  static_assert(1 == ns::size_v<std::tuple<double>>);
  static_assert(1 == ns::size(std::make_tuple(3.14)));

  static_assert(2 == ns::size_v<std::pair<int,int>>);
  static_assert(2 == ns::size(std::make_pair(13,7)));
  static_assert(2 == ns::size_v<std::tuple<int,unsigned int>>);
  static_assert(2 == ns::size(std::make_tuple(13,7u)));
  static_assert(3 == ns::size_v<std::tuple<int,unsigned int,std::size_t>>);
  static_assert(3 == ns::size(std::make_tuple(13,7u,std::size_t{42})));

  static_assert(2 == ns::size_v<std::pair<int&,int&>>);
  static_assert(2 == ns::size_v<std::tuple<const int &,unsigned int &>>);
  static_assert(3 == ns::size_v<std::tuple<const int &,unsigned int, const std::size_t&>>);
  static_assert(3 == ns::size_v<const std::tuple<const int &,unsigned int, const std::size_t&>&>);

  int value1{};
  unsigned int value2{};
  static_assert(2 == ns::size(std::tie(value1,value2)));

  using int2 = std::pair<int,int>;
  using uint3 = std::tuple<unsigned int, unsigned int, unsigned int>;
  static_assert(2 == ns::size_v<std::pair<int2,uint3>>);
  static_assert(3 == ns::size_v<std::tuple<int2,uint3,std::size_t>>);

  using uint2x3 = std::pair<uint3,uint3>;
  static_assert(2 == ns::size_v<uint2x3>);
  static_assert(2 == ns::size(uint2x3{}));
}

