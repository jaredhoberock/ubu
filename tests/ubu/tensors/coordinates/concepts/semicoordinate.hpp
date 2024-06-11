#include <ubu/tensors/coordinates/concepts/semicoordinate.hpp>
#include <ubu/tensors/coordinates/point.hpp>
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


void test_semicoordinate()
{
  {
    // test some types that are semicoordinates

    static_assert(ns::semicoordinate<int>);
    static_assert(ns::semicoordinate<unsigned int>);
    static_assert(ns::semicoordinate<std::size_t>);
    static_assert(ns::semicoordinate<std::tuple<char>>);
    static_assert(ns::semicoordinate<ns::point<int,1>>);
    static_assert(ns::semicoordinate<std::tuple<int&>>);

    using int2 = std::pair<int,int>;
    using uint3 = std::tuple<unsigned int, unsigned int, unsigned int>;

    static_assert(ns::semicoordinate<int2>);
    static_assert(ns::semicoordinate<ns::int2>);
    static_assert(ns::semicoordinate<uint3>);
    static_assert(ns::semicoordinate<ns::uint3>);
    static_assert(ns::semicoordinate<std::tuple<int>>);
    static_assert(ns::semicoordinate<std::tuple<int,unsigned int>>);
    static_assert(ns::semicoordinate<std::tuple<int,unsigned int,std::size_t>>);
    static_assert(ns::semicoordinate<std::array<std::size_t,3>>);

    static_assert(ns::semicoordinate<std::pair<int2,uint3>>);
    static_assert(ns::semicoordinate<std::tuple<int2,uint3,std::size_t>>);

    using uint2x3 = std::pair<uint3,uint3>;
    static_assert(ns::semicoordinate<uint2x3>);

    static_assert(ns::semicoordinate<has_static_rank_member_variable>);
    static_assert(ns::semicoordinate<has_static_rank_member_function>);
  }


  {
    // test some types that are not semicoordinates

    static_assert(not ns::semicoordinate<float>);
    static_assert(not ns::semicoordinate<void>);
    static_assert(not ns::semicoordinate<std::pair<float,int>>);
    static_assert(not ns::semicoordinate<ns::float2>);
    static_assert(not ns::semicoordinate<int*>);

    using float2 = std::pair<float,float>;
    using double3 = std::tuple<double, double, double>;

    static_assert(not ns::semicoordinate<float2>);
    static_assert(not ns::semicoordinate<double3>);

    static_assert(not ns::semicoordinate<has_rank_member_function>);
    static_assert(not ns::semicoordinate<has_rank_free_function>);
  }
}

