#include <ubu/tensor/coordinates/concepts/coordinate.hpp>
#include <ubu/tensor/coordinates/point.hpp>

namespace ns = ubu;

void test_coordinate()
{
  static_assert(ns::coordinate<ns::point<char,1>>);
  static_assert(ns::coordinate<ns::point<char,2>>);
  static_assert(ns::coordinate<ns::point<char,3>>);
  static_assert(ns::coordinate<ns::point<char,4>>);
  static_assert(ns::coordinate<ns::point<char,5>>);

  static_assert(ns::coordinate<ns::point<unsigned char, 1>>);
  static_assert(ns::coordinate<ns::point<unsigned char, 2>>);
  static_assert(ns::coordinate<ns::point<unsigned char, 3>>);
  static_assert(ns::coordinate<ns::point<unsigned char, 4>>);
  static_assert(ns::coordinate<ns::point<unsigned char, 5>>);

  static_assert(ns::coordinate<ns::int1>);
  static_assert(ns::coordinate<ns::int2>);
  static_assert(ns::coordinate<ns::int3>);
  static_assert(ns::coordinate<ns::int4>);
  static_assert(ns::coordinate<ns::int5>);

  static_assert(ns::coordinate<ns::uint1>);
  static_assert(ns::coordinate<ns::uint2>);
  static_assert(ns::coordinate<ns::uint3>);
  static_assert(ns::coordinate<ns::uint4>);
  static_assert(ns::coordinate<ns::uint5>);

  static_assert(ns::coordinate<ns::size1>);
  static_assert(ns::coordinate<ns::size2>);
  static_assert(ns::coordinate<ns::size3>);
  static_assert(ns::coordinate<ns::size4>);
  static_assert(ns::coordinate<ns::size5>);

  static_assert(not ns::coordinate<ns::float1>);
  static_assert(not ns::coordinate<ns::float2>);
  static_assert(not ns::coordinate<ns::float3>);
  static_assert(not ns::coordinate<ns::float4>);
  static_assert(not ns::coordinate<ns::float5>);

  static_assert(not ns::coordinate<ns::double1>);
  static_assert(not ns::coordinate<ns::double2>);
  static_assert(not ns::coordinate<ns::double3>);
  static_assert(not ns::coordinate<ns::double4>);
  static_assert(not ns::coordinate<ns::double5>);
}

