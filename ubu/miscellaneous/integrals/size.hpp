#pragma once

#include "../../detail/prologue.hpp"

#include "../constant.hpp"
#include "../detail/tag_invoke.hpp"
#include "integral_like.hpp"
#include <array>
#include <concepts>
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
    else
    {
      return size(std::forward<T>(arg));
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


// customize ubu::size for std::array
namespace std
{

template<class T, std::size_t N>
constexpr auto tag_invoke(decltype(ubu::size), const std::array<T,N>&)
{
  // note that we return a constant int, not a constant std::size_t
  // this is simply for consistency with what the _c literal suffix returns
  return ubu::constant<int(N)>{};
}

} // end std


#include "../../detail/epilogue.hpp"

