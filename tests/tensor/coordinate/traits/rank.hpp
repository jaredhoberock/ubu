#include <ubu/tensor/coordinate/point.hpp>
#include <ubu/tensor/coordinate/traits/rank.hpp>
#include <tuple>
#include <utility>

namespace ns = ubu;


struct has_static_rank_member_variable
{
  constexpr static std::size_t rank = 1;
};


struct has_static_rank_member_function
{
  constexpr static std::size_t rank()
  {
    return 1;
  }
};


struct has_rank_member_function
{
  constexpr std::size_t rank()
  {
    return 1;
  }
};

struct has_rank_free_function {};

constexpr std::size_t rank(const has_rank_free_function&)
{
  return 1;
}

void test_rank()
{
  static_assert(1 == ns::rank_v<has_static_rank_member_function>);

  static_assert(1 == ns::rank(has_rank_member_function{}));
  static_assert(1 == ns::rank(has_rank_free_function{}));

  static_assert(1 == ns::rank_v<int>);
  static_assert(1 == ns::rank(13));
  static_assert(1 == ns::rank_v<unsigned int>);
  static_assert(1 == ns::rank(13u));
  static_assert(1 == ns::rank_v<std::size_t>);
  static_assert(1 == ns::rank(std::size_t{13}));
  static_assert(1 == ns::rank_v<std::tuple<char>>);
  static_assert(1 == ns::rank(std::make_tuple('a')));
  static_assert(1 == ns::rank_v<ns::point<int,1>>);
  static_assert(1 == ns::rank(ns::point<int,1>{13}));

  static_assert(1 == ns::rank_v<int&>);
  static_assert(1 == ns::rank_v<const unsigned int&>);
  static_assert(1 == ns::rank_v<std::size_t&&>);
  static_assert(1 == ns::rank_v<const std::tuple<char>&>);

  static_assert(2 == ns::rank_v<std::pair<int,int>>);
  static_assert(2 == ns::rank(std::make_pair(13,7)));
  static_assert(2 == ns::rank_v<std::tuple<int,unsigned int>>);
  static_assert(2 == ns::rank(std::make_tuple(13,7u)));
  static_assert(3 == ns::rank_v<std::tuple<int,unsigned int,std::size_t>>);
  static_assert(3 == ns::rank(std::make_tuple(13,7u,std::size_t{42})));

  static_assert(2 == ns::rank_v<std::pair<int&,int&>>);
  static_assert(2 == ns::rank_v<std::tuple<const int &,unsigned int &>>);
  static_assert(3 == ns::rank_v<std::tuple<const int &,unsigned int, const std::size_t&>>);
  static_assert(3 == ns::rank_v<const std::tuple<const int &,unsigned int, const std::size_t&>&>);

  int value1{};
  unsigned int value2{};
  static_assert(2 == ns::rank(std::tie(value1,value2)));

  using int2 = std::pair<int,int>;
  using uint3 = std::tuple<unsigned int, unsigned int, unsigned int>;
  static_assert(2 == ns::rank_v<std::pair<int2,uint3>>);
  static_assert(3 == ns::rank_v<std::tuple<int2,uint3,std::size_t>>);

  using uint2x3 = std::pair<uint3,uint3>;
  static_assert(2 == ns::rank_v<uint2x3>);
  static_assert(2 == ns::rank(uint2x3{}));
}

