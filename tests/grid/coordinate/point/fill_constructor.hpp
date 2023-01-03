#include <algorithm>
#include <ubu/grid/coordinate/point.hpp>

#undef NDEBUG
#include <cassert>

namespace ns = ubu;


template<std::ptrdiff_t num_elements>
void test()
{
  ns::point<int, num_elements> p{13};

  assert(std::count(p.begin(), p.end(), 13) == num_elements);
}

void test_fill_constructor()
{
  test<1>();
  test<2>();
  test<3>();
}

