#pragma once

#include "../../../detail/prologue.hpp"
#include "pointer_like.hpp"
#include <utility>

namespace ubu
{
namespace detail
{

template<class T, class P>
concept has_reinterpret_pointer_member_function = requires(P ptr)
{
  { std::forward<P>(ptr).template reinterpret_pointer<T>() } -> pointer_like;

  // XXX should also check that the pointee is T
};

template<class T, class P>
concept has_reinterpret_pointer_free_function = requires(P ptr)
{
  { reinterpret_pointer<T>(std::forward<P>(ptr)) } -> pointer_like;

  // XXX should also check that the pointee is T
};


template<class T>
struct dispatch_reinterpret_pointer
{
  template<class P>
    requires has_reinterpret_pointer_member_function<T,P>
  constexpr pointer_like auto operator()(P&& ptr) const
  {
    return std::forward<P>(ptr).template reinterpret_pointer<T>();
  }

  template<class P>
    requires (not has_reinterpret_pointer_member_function<T,P>
              and has_reinterpret_pointer_free_function<T,P>)
  constexpr pointer_like auto operator()(P&& ptr) const
  {
    return reinterpret_pointer<T>(std::forward<P>(ptr));
  }

  // this dispatch path handles raw pointers
  template<class U>
  constexpr pointer_like auto operator()(U* ptr) const
  {
    return reinterpret_cast<T*>(ptr);
  }

  // XXX TODO: i don't know of a general way to reinterpret fancy pointers
  //           which do not implement reinterpret_pointer
  //           std::pointer_traits isn't much help
};

} // end detail


template<class T>
inline constexpr detail::dispatch_reinterpret_pointer<T> reinterpret_pointer;

template<class T, pointer_like P>
using reinterpret_pointer_result_t = decltype(reinterpret_pointer<T>(std::declval<P>()));


} // end ubu

#include "../../../detail/epilogue.hpp"

