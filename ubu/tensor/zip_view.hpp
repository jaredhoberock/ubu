#pragma once

#include "../detail/prologue.hpp"

#include "coordinate/concepts/congruent.hpp"
#include "coordinate/concepts/coordinate.hpp"
#include "concepts/sized_tensor_like.hpp"
#include "concepts/tensor_like.hpp"
#include "element_exists.hpp"
#include "iterator.hpp"
#include "shape/shape.hpp"
#include "slice/slice.hpp"
#include "slice/slicer.hpp"
#include "traits/tensor_coordinate.hpp"
#include "traits/tensor_reference.hpp"
#include "traits/tensor_shape.hpp"
#include <ranges>
#include <tuple>


namespace ubu
{
namespace detail
{

// zip_view and zip have a cyclic dependency and can't use each other directly
// declare detail::invoke_zip for zip_view's use
template<class... Args>
constexpr auto invoke_zip(Args&&... args);


} // end detail


template<tensor_like T, tensor_like... Ts>
  requires (... and congruent<tensor_shape_t<T>, tensor_shape_t<Ts>>)
class zip_view
{
  public:
    using coordinate_type = tensor_coordinate_t<T>;

    constexpr zip_view(T tensor, Ts... tensors)
      : tuple_of_tensors_(tensor, tensors...)
    {}

    zip_view(const zip_view&) = default;

    // XXX instead of simply returning the first tensor's shape, we should prefer
    //     to return any modes of the tensors' shapes which happen to be constant
    constexpr auto shape() const
    {
      return ubu::shape(get<0>(tuple_of_tensors_));
    }

    template<class T_ = T>
      requires (sized_tensor_like<T_> and (... and sized_tensor_like<Ts>))
    constexpr auto size() const
    {
      return std::ranges::size(get<0>(tuple_of_tensors_));
    }

    template<coordinate_for<T> C>
    constexpr std::tuple<tensor_reference_t<T>, tensor_reference_t<Ts>...> operator[](const C& coord) const
    {
      return std::apply([=](const auto&... tensors)
      {
        return std::forward_as_tuple(ubu::element(tensors, coord)...);
      }, tuple_of_tensors_);
    }

    template<coordinate_for<T> C>
    constexpr bool element_exists(const C& coord) const
    {
      return std::apply([=](const auto&... tensors)
      {
        return (... and ubu::element_exists(tensors, coord));
      }, tuple_of_tensors_);
    }

    template<class Self = zip_view>
    constexpr tensor_iterator<Self> begin() const
    {
      return {*this};
    }

    constexpr tensor_sentinel end() const
    {
      return {};
    }

    template<slicer_for<coordinate_type> K>
    constexpr tensor_like auto slice(const K& katana) const
    {
      return std::apply([=](const auto&... tensors)
      {
        return detail::invoke_zip(ubu::slice(tensors, katana)...);
      }, tuple_of_tensors_);
    }

  private:
    std::tuple<T,Ts...> tuple_of_tensors_;
};


namespace detail
{

// zip_view and zip have a cyclic dependency and can't use each other directly
// define detail::make_zip_view as soon as zip_view's definition is available
template<tensor_like T, tensor_like... Ts>
  requires (... and congruent<tensor_shape_t<T>, tensor_shape_t<Ts>>)
constexpr tensor_like auto make_zip_view(T tensor, Ts... tensors)
{
  return zip_view<T,Ts...>(tensor, tensors...);
}

} // end detail


} // end ubu

#include "../detail/epilogue.hpp"

