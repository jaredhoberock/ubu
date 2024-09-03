#include <ubu/tensors/concepts/tensor.hpp>
#include <ubu/tensors/concepts/view.hpp>
#include <ubu/tensors/views/layouts/concepts/layout.hpp>
#include <ubu/tensors/views/layouts/strided_layout.hpp>

void test_concepts()
{
  using namespace ubu;

  using type = strided_layout<int,int,int>;

  static_assert(tensor<type>);
  static_assert(view<type>);
  static_assert(layout<type>);
}


