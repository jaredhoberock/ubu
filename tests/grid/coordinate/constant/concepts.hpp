#include <ubu/grid/coordinate/constant.hpp>
#include <ubu/grid/coordinate/concepts/congruent.hpp>
#include <ubu/grid/coordinate/concepts/coordinate.hpp>
#include <utility>

namespace ns = ubu;

void test_concepts()
{
  using namespace ns;

  static_assert(std::integral<constant<1>>);
  static_assert(coordinate<constant<1>>);
  static_assert(congruent<constant<1>, int>);
  static_assert(weakly_congruent<constant<1>, std::pair<int,int>>);
}

