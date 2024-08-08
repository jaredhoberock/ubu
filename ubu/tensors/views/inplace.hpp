#pragma once

#include "../../detail/prologue.hpp"

#include "../concepts/tensor_like.hpp"
#include "../concepts/view.hpp"
#include "../coordinates/concepts/bounded_coordinate.hpp"
#include "../traits/tensor_element.hpp"
#include "../traits/tensor_shape.hpp"
#include "inplaced_view.hpp"
#include <concepts>
#include <utility>

namespace ubu
{
namespace detail
{

template<class T>
concept has_inplace_member_function = requires(T tensor)
{
  { std::forward<T>(tensor).inplace() } -> view;
  
  // XXX need to require that the result has the same shape and element type as T
};

template<class T>
concept has_inplace_free_function = requires(T tensor)
{
  { inplace(std::forward<T>(tensor)) } -> view;

  // XXX need to require that the result has the same shape and element type as T
}

template<class T>
concept has_inplace_customization = has_inplace_member_function<T> or has_inplace_free_function<T>;

template<class T>
concept inplacable =
  tensor_like<T>
  and bounded_coordinate<tensor_shape_t<T>>
  and std::is_trivially_copy_constructible_v<tensor_element_t<T>>
;

template<class T>
concept can_inplace = has_inplace_customization<T> or inplacable<T>;


struct dispatch_inplace
{
  template<can_inplace T>
  constexpr view auto operator()(T&& t) const
  {
    if constexpr (has_inplace_member_function<T&&>)
    {
      return std::forward<T>(t).inplace();
    }
    else if constexpr (has_inplace_free_function<T&&>)
    {
      return inplace(std::forward<T>(t));
    }
    else
    {
      return inplaced_view(all(std::forward<T>(t));
    }
  }
};

} // end detail


inline constexpr detail::dispatch_inplace inplace;


} // end ubu

#include "../../detail/epilogue.hpp"

