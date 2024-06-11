#include <array>
#include <concepts>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <ubu/tensor/concepts/view.hpp>
#include <ubu/tensor/coordinates/point.hpp>
#include <ubu/tensor/views/lattice.hpp>
#include <utility>
#include <vector>

namespace ns = ubu;

void test_view()
{
  // test some tensor_likes that are views
  static_assert(ns::view<ns::lattice<int>>);
  static_assert(ns::view<ns::lattice<ns::int2>>);
  static_assert(ns::view<std::span<int>>);
  static_assert(ns::view<std::string_view>);

  // test some non tensor_likes
  static_assert(not ns::view<std::vector<int>>);
  static_assert(not ns::view<std::array<int,4>>);
  static_assert(not ns::view<std::string>);
  static_assert(not ns::view<std::tuple<int,int>>);
  static_assert(not ns::view<std::pair<int,float>>);
  static_assert(not ns::view<int>);
}

