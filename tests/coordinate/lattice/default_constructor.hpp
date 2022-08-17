#include <ubu/coordinate/lattice.hpp>
#include <ubu/coordinate/point.hpp>

#undef NDEBUG
#include <cassert>

void test_default_constructor()
{
  namespace ns = ubu;

  assert(ns::lattice<int>{}.empty());
  assert(ns::lattice<ns::int1>{}.empty());
  assert(ns::lattice<ns::int2>{}.empty());
  assert(ns::lattice<ns::int3>{}.empty());
  assert(ns::lattice<ns::int4>{}.empty());
  assert(ns::lattice<ns::int5>{}.empty());
  assert(ns::lattice<ns::int6>{}.empty());
  assert(ns::lattice<ns::int7>{}.empty());
  assert(ns::lattice<ns::int8>{}.empty());
  assert(ns::lattice<ns::int9>{}.empty());
  assert(ns::lattice<ns::int10>{}.empty());

  assert(ns::lattice<unsigned int>{}.empty());
  assert(ns::lattice<ns::uint1>{}.empty());
  assert(ns::lattice<ns::uint2>{}.empty());
  assert(ns::lattice<ns::uint3>{}.empty());
  assert(ns::lattice<ns::uint4>{}.empty());
  assert(ns::lattice<ns::uint5>{}.empty());
  assert(ns::lattice<ns::uint6>{}.empty());
  assert(ns::lattice<ns::uint7>{}.empty());
  assert(ns::lattice<ns::uint8>{}.empty());
  assert(ns::lattice<ns::uint9>{}.empty());
  assert(ns::lattice<ns::uint10>{}.empty());
}

