#include <ubu/tensor/coordinate/concepts/ranked.hpp>
#include <ubu/tensor/coordinate/point.hpp>
#include <utility>
#include <tuple>
#include <array>


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


void test_ranked()
{
  {
    // test some ranked types
    static_assert(ns::ranked<int>);
    static_assert(ns::ranked<unsigned int>);
    static_assert(ns::ranked<std::size_t>);
    static_assert(ns::ranked<std::tuple<char>>);
    static_assert(ns::ranked<ns::point<int,1>>);
    static_assert(ns::ranked<std::tuple<int&>>);

    using int2 = std::pair<int,int>;
    using uint3 = std::tuple<unsigned int, unsigned int, unsigned int>;

    static_assert(ns::ranked<int2>);
    static_assert(ns::ranked<ns::int2>);
    static_assert(ns::ranked<uint3>);
    static_assert(ns::ranked<ns::uint3>);
    static_assert(ns::ranked<std::tuple<int>>);
    static_assert(ns::ranked<std::tuple<int,unsigned int>>);
    static_assert(ns::ranked<std::tuple<int,unsigned int,std::size_t>>);
    static_assert(ns::ranked<std::array<std::size_t,3>>);

    static_assert(ns::ranked<std::pair<int2,uint3>>);
    static_assert(ns::ranked<std::tuple<int2,uint3,std::size_t>>);

    using uint2x3 = std::pair<uint3,uint3>;
    static_assert(ns::ranked<uint2x3>);

    static_assert(ns::ranked<has_static_rank_member_variable>);
    static_assert(ns::ranked<has_static_rank_member_function>);
  }


  {
    // test some types that are not ranked

    static_assert(not ns::ranked<float>);
    static_assert(not ns::ranked<void>);
    static_assert(not ns::ranked<std::pair<float,int>>);
    static_assert(not ns::ranked<ns::float2>);
    static_assert(not ns::ranked<int*>);

    using float2 = std::pair<float,float>;
    using double3 = std::tuple<double, double, double>;

    static_assert(not ns::ranked<float2>);
    static_assert(not ns::ranked<double3>);

    static_assert(not ns::ranked<has_rank_member_function>);
    static_assert(not ns::ranked<has_rank_free_function>);
  }
}

