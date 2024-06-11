#include <ubu/miscellaneous/constant.hpp>
#include <ubu/tensors/coordinates/iterators/lexicographical_iterator.hpp>
#include <ubu/tensors/coordinates/point.hpp>
#include <utility>

namespace ns = ubu;

void test_lexicographical_iterator()
{
  using namespace ns;

  static_assert(std::random_access_iterator<lexicographical_iterator<int>>);
  static_assert(std::random_access_iterator<lexicographical_iterator<int,int>>);
  static_assert(std::random_access_iterator<lexicographical_iterator<int,int,int>>);

  static_assert(std::random_access_iterator<lexicographical_iterator<ns::int2>>);
  static_assert(std::random_access_iterator<lexicographical_iterator<ns::int2, std::pair<constant<7>, constant<3>>>>);
  static_assert(std::random_access_iterator<lexicographical_iterator<ns::int3>>);
  static_assert(std::random_access_iterator<lexicographical_iterator<ns::uint3, ns::int3, ns::size3>>);
}

