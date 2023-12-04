#pragma once

#include "../../../detail/prologue.hpp"

#include "../../../causality/happening.hpp"
#include "../../../grid/coordinate/coordinate.hpp"
#include "../../../grid/lattice.hpp"
#include "../concepts/executor.hpp"
#include "../execute_after.hpp"
#include "../traits/executor_happening.hpp"
#include <concepts>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>
#include <utility>

namespace ubu::detail
{


// XXX for now, this only works for "flat" groups of threads defined by workspace_shape = std::size_t, until we can come up with a generalization
//     that makes sense
//     for example we could require that shape and workspace_size are congruent
//     and then iterate through a lattice, creating a workspace for each mode of the lattice
template<coordinate S, std::regular_invocable<S, std::span<std::byte>> F>
constexpr std::regular_invocable auto make_default_new_bulk_execute_after_invocable(const S& shape, std::size_t workspace_size, F&& function)
{
  return [=, function = std::forward<F>(function)]
  {
    std::vector<std::byte> buffer(workspace_size);
    std::span workspace(buffer.data(), buffer.size());

    for(auto coord : lattice(shape))
    {
      std::invoke(function, coord, workspace);
    }
  };
}


template<coordinate S, std::regular_invocable<S, std::span<std::byte>> F>
using default_new_bulk_execute_after_invocable_t = decltype(make_default_new_bulk_execute_after_invocable(std::declval<S>(), std::declval<std::size_t>(), std::declval<F>()));


template<class E, happening H, coordinate S, std::regular_invocable<S, std::span<std::byte>> F>
  requires dependent_executor_of<E&&, H&&, default_new_bulk_execute_after_invocable_t<S,F>>
executor_happening_t<E> default_new_bulk_execute_after(E&& ex, H&& before, const S& shape, std::size_t workspace_size, F&& function)
{
  // create an invocable to represent the kernel
  std::regular_invocable auto kernel = make_default_new_bulk_execute_after_invocable(shape, workspace_size, std::forward<F>(function));

  // asynchronously execute the kernel
  return execute_after(ex, std::forward<H>(before), std::move(kernel));
}


template<class E, class B, class S, class W, class F>
concept has_default_new_bulk_execute_after = requires(E ex, B before, S shape, W workspace_shape, F f)
{
  { default_new_bulk_execute_after(std::forward<E>(ex), std::forward<B>(before), std::forward<S>(shape), std::forward<W>(workspace_shape), std::forward<F>(f)) } -> happening;
};


} // end ubu::detail

#include "../../../detail/epilogue.hpp"

