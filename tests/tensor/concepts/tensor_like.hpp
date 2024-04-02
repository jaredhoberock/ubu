#include <array>
#include <concepts>
#include <string>
#include <tuple>
#include <ubu/tensor/concepts/tensor_like.hpp>
#include <ubu/tensor/concepts/tensor_like_of.hpp>
#include <ubu/tensor/coordinate/point.hpp>
#include <ubu/tensor/lattice.hpp>
#include <ubu/tensor/traits/tensor_element.hpp>
#include <ubu/tensor/traits/tensor_shape.hpp>
#include <utility>
#include <vector>

namespace ns = ubu;

template<class Tensor, class Element, class Shape>
void test_should_be_a_tensor_like()
{
  using namespace ns;

  static_assert(tensor_like<Tensor>);
  static_assert(tensor_like<Tensor&>);
  static_assert(tensor_like<const Tensor&>);

  static_assert(std::same_as<Element, tensor_element_t<Tensor>>);
  static_assert(std::same_as<Element, tensor_element_t<Tensor&>>);
  static_assert(std::same_as<Element, tensor_element_t<const Tensor&>>);

  static_assert(std::same_as<Shape, tensor_shape_t<Tensor>>);
  static_assert(std::same_as<Shape, tensor_shape_t<Tensor&>>);
  static_assert(std::same_as<Shape, tensor_shape_t<const Tensor&>>);

  static_assert(tensor_like_of<Tensor, Element>);
  static_assert(tensor_like_of<Tensor&, Element>);
  static_assert(tensor_like_of<const Tensor&, Element>);
}

template<class T>
void test_non_tensor_like()
{
  static_assert(not ns::tensor_like<T>);
}

void test_tensor_like()
{
  // test some tensor_likes
  test_should_be_a_tensor_like<std::vector<int>, int, std::size_t>();
  test_should_be_a_tensor_like<std::vector<float>, float, std::size_t>();
  test_should_be_a_tensor_like<std::array<int, 4>, int, std::size_t>();
  test_should_be_a_tensor_like<std::string, char, std::size_t>();
  test_should_be_a_tensor_like<ns::lattice<int>, int, int>();
  test_should_be_a_tensor_like<ns::lattice<ns::int2>, ns::int2, ns::int2>();

  // test some non tensor_likes
  test_non_tensor_like<std::tuple<int,int>>();
  test_non_tensor_like<std::pair<int,float>>();
  test_non_tensor_like<int>();
}

