#pragma once

#include "../detail/prologue.hpp"

#include "../tensor/coordinate/concepts/integral_like.hpp"
#include "detail/tag_invoke.hpp"
#include <concepts>
#include <ranges>
#include <utility>

namespace ubu
{
namespace detail
{

template<class T>
concept has_size_member_function = requires(T arg)
{
  { std::forward<T>(arg).size() } -> integral_like;
};

template<class T>
concept has_size_free_function = requires(T arg)
{
  { size(std::forward<T>(arg)) } -> integral_like;
};

template<class T, class CPO>
concept has_size_customization =
  tag_invocable<CPO, T>
  or has_size_member_function<T>
  or has_size_free_function<T>
  or std::ranges::sized_range<T>
;

struct dispatch_size
{
  template<has_size_customization<dispatch_size> T>
  constexpr integral_like auto operator()(T&& arg) const
  {
    if constexpr(tag_invocable<dispatch_size,T&&>)
    {
      return tag_invoke(*this, std::forward<T>(arg));
    }
    else if constexpr(has_size_member_function<T&&>)
    {
      return std::forward<T>(arg).size();
    }
    else if constexpr(has_size_free_function<T&&>)
    {
      return size(std::forward<T>(arg));
    }
    else
    {
      return std::ranges::size(std::forward<T>(arg));
    }
  }
};

} // end detail

inline constexpr detail::dispatch_size size;

template<class T>
using size_result_t = decltype(size(std::declval<T>()));

template<class T>
concept sized =
  requires(T arg)
  {
    size(arg);
  }
;

} // end ubu

#include "../detail/epilogue.hpp"

