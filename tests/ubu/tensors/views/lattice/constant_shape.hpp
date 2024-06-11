#include <algorithm>
#include <cassert>
#include <tuple>
#include <utility>
#include <ubu/miscellaneous/constant.hpp>
#include <ubu/tensors/coordinates/comparisons.hpp>
#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/views/lattice.hpp>

namespace ns = ubu;

void test_constant_shape()
{
  using namespace ubu;

  {
    int origin = 13;
    auto shape = 7_c;

    ns::lattice l(origin, shape);

    int linear_idx = 0;
    for(auto i = l.begin(); i != l.end(); ++i, ++linear_idx)
    {
      auto result = *i;
      auto expected = l[linear_idx];
      assert(expected == result);
    }

    assert(std::is_sorted(l.begin(), l.end(), ns::colex_less));
  }

  {
    ns::int2 origin = {13,7};
    std::pair shape = {3_c,2_c};

    ns::lattice l(origin, shape);

    int linear_idx = 0;
    for(auto i = l.begin(); i != l.end(); ++i, ++linear_idx)
    {
      auto result = *i;
      auto expected = l[linear_idx];
      if(expected != result)
      {
        std::cout << "linear_idx: " << linear_idx << std::endl;
        std::cout << "expected: " << expected << std::endl;
        std::cout << "result: " << result << std::endl;
      }
      assert(expected == result);
    }

    assert(std::is_sorted(l.begin(), l.end(), ns::colex_less));
  }

  {
    ns::int3 origin = {13,7,42};
    std::tuple shape = {3_c,2_c,1_c};

    ns::lattice l(origin, shape);

    int linear_idx = 0;
    for(auto i = l.begin(); i != l.end(); ++i, ++linear_idx)
    {
      auto result = *i;
      auto expected = l[linear_idx];
      assert(expected == result);
    }

    assert(std::is_sorted(l.begin(), l.end(), ns::colex_less));
  }
}

