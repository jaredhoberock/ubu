#include <ubu/coordinate/lattice.hpp>
#include <ubu/coordinate/point.hpp>

#undef NDEBUG
#include <cassert>

void test_number_of_dimensions()
{
  namespace ns = ubu;

  std::cout << "number_of_dimensions: " << ns::lattice<int>::number_of_dimensions() << std::endl;
  std::cout << "ns::size_v<int>: " << ns::size_v<int> << std::endl;
  std::cout << "ns::size(13): " << ns::size(13) << std::endl;

  assert(1 == ns::lattice<int>::number_of_dimensions());
  assert(2 == ns::lattice<ns::int2>::number_of_dimensions());
  assert(3 == ns::lattice<ns::int3>::number_of_dimensions());
  assert(4 == ns::lattice<ns::int4>::number_of_dimensions());

  assert(1 == ns::lattice<unsigned int>::number_of_dimensions());
  assert(2 == ns::lattice<ns::uint2>::number_of_dimensions());
  assert(3 == ns::lattice<ns::uint3>::number_of_dimensions());
  assert(4 == ns::lattice<ns::uint4>::number_of_dimensions());
}

