#include <algorithm>
#include <cassert>
#include <ubu/tensor/coordinates/comparisons.hpp>
#include <ubu/tensor/coordinates/point.hpp>
#include <ubu/tensor/shape/convert_shape.hpp>
#include <ubu/tensor/shape/shape_size.hpp>
#include <ubu/tensor/views/layout/lifting_layout.hpp>

namespace ns = ubu;

template<ns::coordinate S>
void test(S shape)
{
  int n = ns::shape_size(shape);

  ns::lifting_layout<int, S> layout(n, shape);
  std::vector<S> result(n);

  for(int i = 0; i < n; ++i)
  {
    result[i] = layout[i];
  }

  assert(std::is_sorted(result.begin(), result.end(), ns::colex_less));
}

void test_lifting_layout()
{
  using namespace std;

  size_t n = 12345;

  test(n);
  test(ns::convert_shape<ns::int2>(n));
  test(ns::convert_shape<ns::int3>(n));
  test(ns::convert_shape<pair<int, ns::int2>>(n));
  test(ns::convert_shape<pair<ns::int3, ns::int2>>(n));
}

