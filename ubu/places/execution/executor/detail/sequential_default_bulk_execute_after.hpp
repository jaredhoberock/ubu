#pragma once

#include "../../../../detail/prologue.hpp"

#include "../../../../tensor/coordinates/concepts/coordinate.hpp"
#include "../../../../tensor/views/lattice.hpp"
#include "../../../causality/happening.hpp"
#include "../concepts/executor.hpp"
#include "../execute_after.hpp"
#include "../traits/executor_happening.hpp"
#include <concepts>
#include <functional>
#include <vector>
#include <utility>

namespace ubu::detail
{


template<coordinate S, std::invocable<S> F>
constexpr std::invocable auto make_sequential_default_bulk_execute_after_invocable(const S& shape, F&& function)
{
  return [=, function = std::forward<F>(function)]
  {
    for(auto coord : lattice(shape))
    {
      std::invoke(function, coord);
    }
  };
}


template<coordinate S, std::invocable<S> F>
using sequential_default_bulk_execute_after_invocable_t = decltype(make_sequential_default_bulk_execute_after_invocable(std::declval<S>(), std::declval<F>()));


// this default implementation of bulk_execute_after calls execute_after and puts the user function in a for loop
template<class E, happening H, coordinate S, std::invocable<S> F>
  requires dependent_executor_of<E&&, H&&, sequential_default_bulk_execute_after_invocable_t<S,F>>
executor_happening_t<E> sequential_default_bulk_execute_after(E&& ex, H&& before, const S& shape, F&& function)
{
  // create an invocable to represent the kernel
  std::invocable auto kernel = make_sequential_default_bulk_execute_after_invocable(shape,std::forward<F>(function));

  // asynchronously execute the kernel
  return execute_after(ex, std::forward<H>(before), std::move(kernel));
}


template<class E, class B, class S, class F>
concept has_sequential_default_bulk_execute_after = requires(E ex, B before, S shape, F f)
{
  { sequential_default_bulk_execute_after(std::forward<E>(ex), std::forward<B>(before), std::forward<S>(shape), std::forward<F>(f)) } -> happening;
};


} // end ubu::detail

#include "../../../../detail/epilogue.hpp"

