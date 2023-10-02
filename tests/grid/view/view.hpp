#include <algorithm>
#include <cassert>
#include <ubu/grid/coordinate/compare.hpp>
#include <ubu/grid/domain.hpp>
#include <ubu/grid/lattice.hpp>
#include <ubu/grid/layout/column_major.hpp>
#include <ubu/grid/layout/row_major.hpp>
#include <ubu/grid/view.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace ns = ubu;

template<ns::coordinate S>
void test(S shape)
{
  using namespace ns;

  {
    // column major view of a lattice

    lattice grid(shape);
    column_major layout(shape);

    view v(grid, layout);

    for(auto c : domain(v))
    {
      auto result = v[c];
      auto expected = grid[c];
      assert(expected == result);
    }

    assert(std::is_sorted(domain(v).begin(), domain(v).end(), ns::colex_less));
  }

  {
    // row major view of a lattice

    lattice grid(shape);
    row_major layout(shape);

    view v(grid, layout);

    for(auto c : domain(v))
    {
      auto result = v[c];
      auto expected = grid[ns::apply_stride(c, ns::compact_row_major_stride(shape))];
      assert(expected == result);
    }
  }
}

void test_view()
{
  using namespace std;

  size_t n = 12345;

  test(n);
  test(ns::convert_shape<ns::int2>(n));
  test(ns::convert_shape<ns::int3>(n));
  test(ns::convert_shape<pair<int, ns::int2>>(n));
  test(ns::convert_shape<pair<ns::int3, ns::int2>>(n));
}

