#pragma once

#include "../../detail/prologue.hpp"

#include "../concepts/view.hpp"
#include "all.hpp"
#include "domain.hpp"
#include "zip.hpp"
#include <utility>

namespace ubu
{
namespace detail
{


template<class T>
concept has_enumerate_member_function = requires(T t)
{
  { std::forward<T>(t).enumerate() } -> view;
};

template<class T>
concept has_enumerate_free_function = requires(T t)
{
  { enumerate(std::forward<T>(t)) } -> view;
};


template<class T>
concept has_enumerate_customization =
  has_enumerate_member_function<T>
  or has_enumerate_free_function<T>
;


struct dispatch_enumerate
{
  template<has_enumerate_customization T>
  constexpr view auto operator()(T&& t) const
  {
    if constexpr (has_enumerate_member_function<T&&>)
    {
      return std::forward<T>(t).enumerate();
    }
    else
    {
      return enumerate(std::forward<T>(t));
    }
  }

  template<tensor_like T>
    requires (not has_enumerate_customization<T>)
  constexpr view auto operator()(T&& t) const
  {
    return zip(domain(std::forward<T>(t)), all(std::forward<T>(t)));
  }
};


} // end detail


inline constexpr detail::dispatch_enumerate enumerate;


} // end ubu

#include "../../detail/epilogue.hpp"

