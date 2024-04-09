#pragma once

#include "../detail/prologue.hpp"
#include "concepts/tensor_like.hpp"
#include <type_traits>
#include <utility>

namespace ubu
{
namespace detail
{

template<class T>
concept trivially_copy_constructible = std::is_trivially_copy_constructible_v<T>;

template<class T>
concept has_all_member_function = requires(T t)
{
  { t.all() } -> tensor_like;
  { t.all() } -> trivially_copy_constructible;
};

template<class T>
concept has_all_free_function = requires(T t)
{
  { all(t) } -> tensor_like;
  { all(t) } -> trivially_copy_constructible;
};

struct dispatch_all
{
  template<class T>
    requires has_all_member_function<T>
  constexpr tensor_like auto operator()(T&& t) const
  {
    return std::forward<T>(t).all();
  }

  template<class T>
    requires (not has_all_member_function<T>
              and has_all_free_function<T>)
  constexpr tensor_like auto operator()(T&& t) const
  {
    return all(std::forward<T>(t));
  }

  template<tensor_like T>
    requires (not has_all_member_function<T>
              and not has_all_free_function<T>
              and trivially_copy_constructible<T>)
  constexpr T operator()(T t) const
  {
    // T is already a view, so this operation is the identity function
    return t;
  }
};

} // end detail

constexpr inline detail::dispatch_all all;

template<class T>
using all_t = decltype(all(std::declval<T>()));

} // end ubu

#include "../detail/epilogue.hpp"
