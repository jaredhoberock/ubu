#include <ubu/tensors/concepts/tensor_like.hpp>
#include <ubu/tensors/concepts/view.hpp>
#include <ubu/tensors/views/layouts/concepts/layout.hpp>
#include <ubu/tensors/views/layouts/strided_layout.hpp>

void test_concepts()
{
  using namespace ubu;

  static_assert(tensor_like<strided_layout<int,int,int>>);
  static_assert(view<strided_layout<int,int,int>>);
  static_assert(layout<strided_layout<int,int,int>>);
}


