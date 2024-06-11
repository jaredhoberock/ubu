#pragma once

#include "../../detail/prologue.hpp"
#include "../../miscellaneous/integral/size.hpp"
#include "../concepts/tensor_like.hpp"
#include "../concepts/view.hpp"
#include "../vectors/contiguous_vector_like.hpp"
#include "../vectors/fancy_span.hpp"
#include "../vectors/span_like.hpp"
#include <iterator>
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
    requires has_all_member_function<T&&>
  constexpr view auto operator()(T&& t) const
  {
    return std::forward<T>(t).all();
  }

  template<class T>
    requires (not has_all_member_function<T&&>
              and has_all_free_function<T&&>)
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

  template<contiguous_vector_like V>
    requires (not has_all_member_function<V&&>
              and not has_all_free_function<V&&>
              and not view<std::remove_cvref_t<V&&>>)
  constexpr view auto operator()(V&& vec) const
  {
    return fancy_span(std::data(std::forward<V>(vec)), size(std::forward<V>(vec)));
  }

  // XXX consider eliminating this function since a span_like is already a view
  template<span_like S>
    requires (not has_all_member_function<S>
              and not has_all_free_function<S>
              and not view<std::remove_cvref_t<S>>)
  constexpr span_like auto operator()(S s) const
  {
    // XXX I think this should be std::data instead of std::ranges::data
    return fancy_span(std::ranges::data(s), size(s));
  }

  // XXX we could have another default path here for std::ranges::random_access_range
};

} // end detail

constexpr inline detail::dispatch_all all;

template<class T>
using all_t = decltype(all(std::declval<T>()));

} // end ubu

#include "../../detail/epilogue.hpp"

