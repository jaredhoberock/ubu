#pragma once

#include "../../detail/prologue.hpp"

#include "../concepts/tensor.hpp"
#include "../concepts/view.hpp"
#include "all.hpp"
#include "nestled_view.hpp"
#include <utility>

namespace ubu
{
namespace detail
{


template<class T>
concept has_nestle_member_function = requires(T t)
{
  { std::forward<T>(t).nestle() } -> view;

  // XXX need to require that the result has rank<T> - 1
  //     need to require that the tensor_element of the result has rank 1
};

template<class T>
concept has_nestle_free_function = requires(T t)
{
  { nestle(std::forward<T>(t)) } -> view;

  // XXX need to require that the result has rank<T> - 1
  //     need to require that the tensor_element of the result has rank 1
};

template<class T>
concept has_nestle_customization = (has_nestle_member_function<T> or has_nestle_free_function<T>);


struct dispatch_nestle
{
  template<has_nestle_customization T>
  constexpr view auto operator()(T&& t) const
  {
    if constexpr (has_nestle_member_function<T&&>)
    {
      return std::forward<T>(t).nestle();
    }
    else
    {
      return nestle(std::forward<T>(t));
    }
  }

  template<tensor T>
    requires (not has_nestle_customization<T&&>)
  constexpr view auto operator()(T&& tensor) const
  {
    return nestled_view(all(std::forward<T>(tensor)));
  }
};


} // end detail

inline constexpr detail::dispatch_nestle nestle;


template<class T>
using nestle_result_t = decltype(nestle(std::declval<T>()));

} // end ubu

#include "../../detail/epilogue.hpp"

