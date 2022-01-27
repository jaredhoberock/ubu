#include <aspera/coordinate/lattice.hpp>
#include <aspera/coordinate/point.hpp>

#undef NDEBUG
#include <cassert>

void test_default_constructor()
{
  namespace ns = aspera;

  assert(ns::lattice<int>{}.empty());
  assert(ns::lattice<ns::int2>{}.empty());
  assert(ns::lattice<ns::int3>{}.empty());
  assert(ns::lattice<ns::int4>{}.empty());

  assert(ns::lattice<unsigned int>{}.empty());
  assert(ns::lattice<ns::uint2>{}.empty());
  assert(ns::lattice<ns::uint3>{}.empty());
  assert(ns::lattice<ns::uint4>{}.empty());
}

