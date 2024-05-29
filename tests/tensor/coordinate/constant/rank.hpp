#include <ubu/miscellaneous/constant.hpp>
#include <ubu/tensor/coordinate/traits/rank.hpp>
#include <utility>

namespace ns = ubu;

void test_rank()
{
  using namespace ns;

  static_assert(1 == ns::rank_v<constant<1>>);
  static_assert(2 == ns::rank_v<std::pair<constant<1>, int>>);
}
