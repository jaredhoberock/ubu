#include <ubu/tensor/coordinate/constant.hpp>
#include <ubu/tensor/coordinate/iterator/colexicographical_iterator.hpp>
#include <ubu/tensor/coordinate/point.hpp>
#include <utility>

namespace ns = ubu;

void test_colexicographical_iterator()
{
  using namespace ns;

  static_assert(std::random_access_iterator<colexicographical_iterator<int>>);
  static_assert(std::random_access_iterator<colexicographical_iterator<int,int>>);
  static_assert(std::random_access_iterator<colexicographical_iterator<int,int,int>>);

  static_assert(std::random_access_iterator<colexicographical_iterator<ns::int2>>);
  static_assert(std::random_access_iterator<colexicographical_iterator<ns::int2, std::pair<constant<7>, constant<3>>>>);
  static_assert(std::random_access_iterator<colexicographical_iterator<ns::int3>>);
  static_assert(std::random_access_iterator<colexicographical_iterator<ns::uint3, ns::int3, ns::size3>>);
}

