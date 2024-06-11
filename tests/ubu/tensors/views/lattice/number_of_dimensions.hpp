#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/views/lattice.hpp>

#undef NDEBUG
#include <cassert>

void test_number_of_dimensions()
{
  namespace ns = ubu;

  assert(1 == ns::lattice<int>::number_of_dimensions());
  assert(2 == ns::lattice<ns::int2>::number_of_dimensions());
  assert(3 == ns::lattice<ns::int3>::number_of_dimensions());
  assert(4 == ns::lattice<ns::int4>::number_of_dimensions());

  assert(1 == ns::lattice<unsigned int>::number_of_dimensions());
  assert(2 == ns::lattice<ns::uint2>::number_of_dimensions());
  assert(3 == ns::lattice<ns::uint3>::number_of_dimensions());
  assert(4 == ns::lattice<ns::uint4>::number_of_dimensions());
}

