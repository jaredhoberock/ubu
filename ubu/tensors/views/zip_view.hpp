#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/integrals/size.hpp"
#include "../coordinates/concepts/congruent.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "../concepts/sized_tensor.hpp"
#include "../concepts/view.hpp"
#include "../element_exists.hpp"
#include "../shapes/shape.hpp"
#include "../traits/tensor_coordinate.hpp"
#include "../traits/tensor_reference.hpp"
#include "../traits/tensor_shape.hpp"
#include "slices/slice.hpp"
#include "slices/slicer.hpp"
#include "view_base.hpp"
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


template<view T, view... Ts>
  requires (... and congruent<tensor_shape_t<T>, tensor_shape_t<Ts>>)
class zip_view : public view_base
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
      requires (sized_tensor<T_> and (... and sized_tensor<Ts>))
    constexpr auto size() const
    {
      return ubu::size(get<0>(tuple_of_tensors_));
    }

    template<coordinate_for<T> C>
    constexpr std::tuple<tensor_reference_t<T>, tensor_reference_t<Ts>...> operator[](const C& coord) const
    {
      // carefully define the result of the lambda below
      // to avoid returning a tuple of dangling references
      using result_type = std::tuple<tensor_reference_t<T>, tensor_reference_t<Ts>...>;

      return std::apply([=](const auto&... tensors)
      {
        return result_type(ubu::element(tensors, coord)...);
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

    template<slicer_for<coordinate_type> K>
    constexpr view auto slice(const K& katana) const
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
template<view T, view... Ts>
  requires (... and congruent<tensor_shape_t<T>, tensor_shape_t<Ts>>)
constexpr view auto make_zip_view(T tensor, Ts... tensors)
{
  return zip_view<T,Ts...>(tensor, tensors...);
}

} // end detail


} // end ubu

#include "../../detail/epilogue.hpp"

