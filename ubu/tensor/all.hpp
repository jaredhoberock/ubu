#pragma once

#include "../detail/prologue.hpp"
#include "../miscellaneous/integral/size.hpp"
#include "concepts/tensor_like.hpp"
#include "concepts/view.hpp"
#include "fancy_span.hpp"
#include "vector/span_like.hpp"
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
  { std::forward<T>(t).all() } -> view;
};

template<class T>
concept has_all_free_function = requires(T t)
{
  { all(std::forward<T>(t)) } -> view;
};

struct dispatch_all
{
  template<class T>
    requires has_all_member_function<T>
  constexpr view auto operator()(T&& t) const
  {
    return std::forward<T>(t).all();
  }

  template<class T>
    requires (not has_all_member_function<T>
              and has_all_free_function<T>)
  constexpr view auto operator()(T&& t) const
  {
    return all(std::forward<T>(t));
  }

  template<view T>
    requires (not has_all_member_function<T>
              and not has_all_free_function<T>)
  constexpr T operator()(T t) const
  {
    // T is already a view, so this operation is the identity function
    return t;
  }

  // XXX are there span_like types that are not also views?
  template<span_like S>
    requires (not has_all_member_function<S&&>
              and not has_all_free_function<S&&>
              and not view<std::remove_cvref_t<S&&>>)
  constexpr span_like auto operator()(S&& s) const
  {
    return fancy_span(std::ranges::data(std::forward<S>(s)), size(std::forward<S>(s)));
  }

  // XXX we could have another default path here for std::ranges::random_access_range
};

} // end detail

constexpr inline detail::dispatch_all all;

template<class T>
using all_t = decltype(all(std::declval<T>()));

} // end ubu

#include "../detail/epilogue.hpp"

