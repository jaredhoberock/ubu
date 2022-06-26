#pragma once

#include "../detail/prologue.hpp"

#include "../causality/happening.hpp"
#include "../execution/executor.hpp"
#include "../memory/allocator.hpp"
#include "../memory/pointer/construct_at.hpp"
#include "intrusive_future.hpp"
#include <concepts>
#include <utility>
#include <tuple>


namespace ubu
{


template<executor E, happening H, class... Args, asynchronous_allocator... As, std::invocable<Args&&...> F,
         class R = std::invoke_result_t<F,Args&&...>,
         asynchronous_allocator_of<R> A
        >
intrusive_future<R,A> invoke_after(const E& ex, const A& alloc, H&& before, F&& f, intrusive_future<Args,As>&&... future_args)
{
  // allocate storage for the result after before is ready
  auto [result_allocation_ready, ptr_to_result] = first_allocate<R>(alloc, 1);

  // create a happening dependent on before, the allocation, and future_args
  auto inputs_ready = dependent_on(ex, std::move(result_allocation_ready), std::forward<H>(before), future_args.ready()...);

  try
  {
    // when everything is ready, invoke f and construct the result
    auto result_ready = execute_after(ex, std::move(inputs_ready), [ptr_to_result = ptr_to_result, f = std::move(f), ... ptrs_to_args = future_args.data()]
    {
      construct_at(ptr_to_result, std::invoke(f, std::move(*ptrs_to_args)...));
    });

    // schedule the deletion of the future_args after the result is ready
    detail::for_each_arg([&](auto&& future_arg)
    {
      auto [alloc, _, ptr] = std::move(future_arg).release();
      finally_delete_after(alloc, result_ready, ptr, 1);
    }, std::move(future_args)...);

    // return a new future
    return {std::move(result_ready), ptr_to_result, alloc};
  }
  catch(...)
  {
    // XXX return an exceptional future
    throw std::runtime_error("invoke_after: execute_after failed");
  }

  // XXX until we can handle exceptions, just return this to make everything compile
  return {std::move(result_allocation_ready), ptr_to_result, alloc};
}


} // end ubu

#include "../detail/epilogue.hpp"

