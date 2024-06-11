#include <ubu/miscellaneous/constant.hpp>
#include <ubu/miscellaneous/constant_valued.hpp>
#include <ubu/miscellaneous/integral/integral_like.hpp>
#include <ubu/tensors/coordinates/concepts/congruent.hpp>
#include <ubu/tensors/coordinates/concepts/coordinate.hpp>
#include <utility>

namespace ns = ubu;

void test_concepts()
{
  using namespace ns;

  static_assert(integral_like<constant<1>>);
  static_assert(coordinate<constant<1>>);
  static_assert(congruent<constant<1>, int>);
  static_assert(weakly_congruent<constant<1>, std::pair<int,int>>);
  static_assert(constant_valued<constant<1>>);
  static_assert(constant_valued<std::pair<constant<1>, constant<1>>>);
  static_assert(not constant_valued<std::pair<constant<1>, int>>);
}

