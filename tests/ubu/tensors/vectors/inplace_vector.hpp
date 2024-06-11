#include <type_traits>
#include <ubu/tensors/vectors/inplace_vector.hpp>

void test_inplace_vector()
{
  using namespace ubu;

  static_assert(std::is_trivially_copy_constructible_v<inplace_vector<int,10>>);
  static_assert(std::is_trivially_move_constructible_v<inplace_vector<int,10>>);
  static_assert(std::is_trivially_destructible_v<inplace_vector<int,10>>);
}

