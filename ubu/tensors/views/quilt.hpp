#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinates/concepts/coordinate.hpp"
#include "../coordinates/concepts/congruent.hpp"
#include "../coordinates/coordinate_cat.hpp"
#include "../concepts/nested_tensor.hpp"
#include "../concepts/view.hpp"
#include "../concepts/viewable_tensor.hpp"
#include "../traits/inner_tensor_shape.hpp"
#include "../traits/tensor_shape.hpp"
#include "compose.hpp"
#include "layouts/identity_layout.hpp"
#include <concepts>
#include <utility>


namespace ubu
{
namespace detail
{


template<class T, class S>
concept quiltable_with =
  nested_tensor<T>
  and coordinate<S>
  and congruent<inner_tensor_shape_t<T&&>,S>
;


template<class R, class T, class S>
concept quilted_view_of = 
  view<R>
  and quiltable_with<T,S>
  and congruent<tensor_shape_t<R>, coordinate_cat_result_t<tensor_shape_t<T>,inner_tensor_shape_t<T>>>
  and std::same_as<tensor_reference_t<R>, tensor_reference_t<inner_tensor_t<T>>>
;


template<class T, class S>
concept has_quilt_member_function = requires(T t, S s)
{
  { std::forward<T>(t).quilt(std::forward<S>(s)) } -> view;
};

template<class T, class S>
concept has_quilt_free_function = requires(T t, S s)
{
  { quilt(std::forward<T>(t), std::forward<S>(s)) } -> view;
};

template<class T, class S>
concept has_quilt_customization = has_quilt_member_function<T,S> or has_quilt_free_function<T,S>;


struct dispatch_quilt
{
  template<class T, class S>
    requires has_quilt_customization<T&&,S&&>
  constexpr quilted_view_of<T&&,S&&> auto operator()(T&& t, S&& s) const
  {
    if constexpr (has_quilt_member_function<T&&,S&&>)
    {
      return std::forward<T>(t).quilt(std::forward<S>(s));
    }
    else
    {
      return quilt(std::forward<T>(t), std::forward<S>(s));
    }
  }

  template<viewable_tensor T, coordinate S>
    requires (not has_quilt_customization<T&&,S> and quiltable_with<T&&,S>)
  constexpr quilted_view_of<T&&,S> auto operator()(T&& t, S s) const
  {
    auto new_shape = coordinate_cat(s, shape(t));
    return compose(std::forward<T>(t), identity_layout(new_shape));
  }
};


} // end detail


inline constexpr detail::dispatch_quilt quilt;


} // end ubu

#include "../../detail/epilogue.hpp"

