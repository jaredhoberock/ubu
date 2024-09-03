#include <array>
#include <concepts>
#include <string>
#include <tuple>
#include <ubu/utilities/constant.hpp>
#include <ubu/tensors/concepts/tensor.hpp>
#include <ubu/tensors/concepts/tensor_of.hpp>
#include <ubu/tensors/coordinates/point.hpp>
#include <ubu/tensors/traits/tensor_element.hpp>
#include <ubu/tensors/traits/tensor_shape.hpp>
#include <ubu/tensors/views/lattice.hpp>
#include <utility>
#include <vector>

namespace ns = ubu;

template<class Tensor, class Element, class Shape>
void test_should_be_a_tensor()
{
  using namespace ns;

  static_assert(tensor<Tensor>);
  static_assert(tensor<Tensor&>);
  static_assert(tensor<const Tensor&>);

  static_assert(std::same_as<Element, tensor_element_t<Tensor>>);
  static_assert(std::same_as<Element, tensor_element_t<Tensor&>>);
  static_assert(std::same_as<Element, tensor_element_t<const Tensor&>>);

  static_assert(std::same_as<Shape, tensor_shape_t<Tensor>>);
  static_assert(std::same_as<Shape, tensor_shape_t<Tensor&>>);
  static_assert(std::same_as<Shape, tensor_shape_t<const Tensor&>>);

  static_assert(tensor_of<Tensor, Element>);
  static_assert(tensor_of<Tensor&, Element>);
  static_assert(tensor_of<const Tensor&, Element>);
}

template<class T>
void test_non_tensor()
{
  static_assert(not ns::tensor<T>);
}

void test_tensor()
{
  // test some tensors
  test_should_be_a_tensor<std::vector<int>, int, std::size_t>();
  test_should_be_a_tensor<std::vector<float>, float, std::size_t>();
  test_should_be_a_tensor<std::array<int, 4>, int, ns::constant<4>>();
  test_should_be_a_tensor<std::string, char, std::size_t>();
  test_should_be_a_tensor<ns::lattice<int>, int, int>();
  test_should_be_a_tensor<ns::lattice<ns::int2>, ns::int2, ns::int2>();

  // test some non tensors
  test_non_tensor<std::tuple<int,int>>();
  test_non_tensor<std::pair<int,float>>();
  test_non_tensor<int>();
}

