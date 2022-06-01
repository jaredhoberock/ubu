#pragma once

#include "prologue.hpp"

#include <functional>
#include <tuple>


namespace ubu::detail
{


template<class... Args>
constexpr void ignore_arguments(Args&&...){}


template<class F, class Tuple>
constexpr void for_each_tuple_element(F&& f, Tuple&& t)
{
  std::apply([&](auto&&... elements)
  {
    // call std::invoke on f with each elements and unpack as parameters into ignore_arguments
    // use the comma operator to avoid problems with f returning void
    detail::ignore_arguments((std::invoke(std::forward<F>(f), elements), 0)...);
  }, std::forward<Tuple>(t));
}


} // end ubu::detail


#include "epilogue.hpp"

