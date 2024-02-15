#include <ubu/tensor/coordinate/constant.hpp>
#include <ubu/tensor/coordinate/concepts/congruent.hpp>
#include <ubu/tensor/coordinate/concepts/coordinate.hpp>
#include <ubu/tensor/coordinate/concepts/integral_like.hpp>
#include <utility>

namespace ns = ubu;

void test_concepts()
{
  using namespace ns;

  static_assert(integral_like<constant<1>>);
  static_assert(coordinate<constant<1>>);
  static_assert(congruent<constant<1>, int>);
  static_assert(weakly_congruent<constant<1>, std::pair<int,int>>);
}
