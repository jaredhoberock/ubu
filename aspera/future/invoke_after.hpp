#pragma once

#include "../detail/prologue.hpp"

#include "../event/event.hpp"
#include "../execution/executor.hpp"
#include "../memory/allocator.hpp"
#include "../memory/construct_at.hpp"
#include "intrusive_future.hpp"
#include <concepts>
#include <utility>
#include <tuple>


ASPERA_NAMESPACE_OPEN_BRACE


template<executor Ex, asynchronous_allocator A, event Ev, class... Args, asynchronous_allocator... As, std::invocable<Args&&...> F,
         class R = std::invoke_result_t<F,Args&&...>
        >
intrusive_future<R,rebind_allocator_result_t<R,A>> invoke_after(const Ex& ex, const A& alloc, Ev&& before, F&& f, intrusive_future<Args,As>&&... future_args)
{
  // rebind the allocator to the type of the result
  auto rebound_alloc = rebind_allocator<R>(alloc);

  // allocate storage for the result after before is ready
  auto [result_allocation_ready, ptr_to_result] = first_allocate<R>(rebound_alloc, 1);

  // create an event dependent on before, the allocation, and future_args
  auto inputs_ready = dependent_on(ex, std::move(result_allocation_ready), std::forward<Ev>(before), future_args.ready()...);

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
    return {std::move(result_ready), ptr_to_result, std::move(rebound_alloc)};
  }
  catch(...)
  {
    // XXX return an exceptional future
    throw std::runtime_error("invoke_after: execute_after failed");
  }

  // XXX until we can handle exceptions, just return this to make everything compile
  return {std::remove_cvref_t<Ev>{}, nullptr, std::move(rebound_alloc)};
}


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

