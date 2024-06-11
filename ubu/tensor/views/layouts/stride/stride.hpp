#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../miscellaneous/constant_valued.hpp"
#include "../../../coordinates/concepts/coordinate.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

template<class T>
concept has_stride_member_function = requires(T arg)
{
  { arg.stride() } -> coordinate;
};

template<class T>
concept has_stride_free_function = requires(T arg)
{
  { stride(arg) } -> coordinate;
};

template<class T>
concept has_stride_customization = has_stride_member_function<T> or has_stride_free_function<T>;

struct dispatch_stride
{
  template<class T>
    requires has_stride_customization<T&&>
  constexpr coordinate auto operator()(T&& arg) const
  {
    if constexpr (has_stride_member_function<T&&>)
    {
      return std::forward<T&&>(arg).stride();
    }
    else if constexpr (has_stride_free_function<T&&>)
    {
      return stride(std::forward<T&&>(arg));
    }
  }

  template<class T>
    requires (not has_stride_customization<T&&>)
  constexpr coordinate auto operator()(T&&) const = delete;
};

} // end detail


inline constexpr detail::dispatch_stride stride;


template<class T>
using stride_t = decltype(stride(std::declval<T>()));


template<class T>
concept strided =
  requires(T arg)
  {
    stride(arg);
  }
;


// note that the type of stride_v<T> is different from stride_t<T>
// because, for example, this definition "unwraps" a constant<13> to 13
// this choice is consistent with the behavior of shape_v<T>
// if we regret this choice for shape_v<T>, then we should update
// the behavior of stride_v<T> accordingly
template<strided T>
  requires constant_valued<stride_t<T>>
constexpr inline auto stride_v = constant_value_v<stride_t<T>>;

template<class T>
concept constant_strided =
  requires()
  {
    // stride_v must exist
    stride_v<T>;
  }
;


} // end ubu

#include "../../../../detail/epilogue.hpp"

