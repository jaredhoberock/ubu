#pragma once

#include "../detail/prologue.hpp"

#include "../coordinate/grid_coordinate.hpp"
#include "../detail/for_each_arg.hpp"
#include "../execution/executor/associated_executor.hpp"
#include "../execution/executor/bulk_execute_after.hpp"
#include "../event/event.hpp"
#include "../memory/allocator/allocator_delete.hpp"
#include "../memory/allocator/asynchronous_allocator.hpp"
#include "../memory/allocator/finally_delete_after.hpp"
#include "../memory/allocator/first_allocate.hpp"
#include "../memory/allocator/rebind_allocator.hpp"
#include "../memory/allocator/traits.hpp"
#include "../memory/construct_at.hpp"
#include <cassert>
#include <concepts>
#include <future>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>

ASPERA_NAMESPACE_OPEN_BRACE


template<class T, asynchronous_allocator A, event E = allocator_event_t<A>>
class intrusive_future
{
  public:
    using value_type = T;
    using pointer = allocator_pointer_t<rebind_allocator_result_t<T,A>>;
    using event_type = E;
    using allocator_type = A;

    intrusive_future(std::tuple<allocator_type, event_type, pointer>&& resources)
      : resources_{std::move(resources)}
    {}

    intrusive_future(event_type&& ready, pointer ptr_to_value, const A& allocator) noexcept
      : intrusive_future{std::forward_as_tuple(allocator, std::move(ready), ptr_to_value)}
    {}

    intrusive_future(intrusive_future&&) = default;

    ~intrusive_future()
    {
      if(data())
      {
        auto [alloc, ready, ptr] = std::move(*this).release();
        finally_delete_after(alloc, std::move(ready), ptr, 1);
      }
    }

    const event_type& ready() const
    {
      return std::get<event_type>(resources_);
    }

    pointer data() const
    {
      return std::get<pointer>(resources_);
    }

    allocator_type allocator() const
    {
      return std::get<allocator_type>(resources_);
    }

    void wait() const
    {
      // either the allocation or the result needs to exist for this future to be valid
      if(!data() and maybe_result_.empty())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      // wait on the event
      std::get<event_type>(resources_).wait();

      // get the result if we haven't already
      if(maybe_result_.empty())
      {
        maybe_result_.emplace(std::move(*data()));

        // release resources
        auto [alloc, ready, ptr] = std::move(*this).release();
        allocator_delete(alloc, ptr);
      }
    }

    value_type get()
    {
      wait();

      value_type result = std::move(maybe_result_).value();
      maybe_result_.reset();
      return result;
    }

    std::tuple<allocator_type,event_type,pointer> release() &&
    {
      auto result = std::move(resources_);

      // ensure data() returns null
      std::get<pointer>(resources_) = nullptr;

      return result;
    }

    // invocable<S, T>
    // requires copyable<T>
    // XXX we should pass f a (raw) reference to the value rather than a pointer
    template<executor Ex, grid_coordinate S, std::regular_invocable<S,pointer> F,
             class Self = intrusive_future
            >
    intrusive_future then_bulk_execute(const Ex& ex, S grid_shape, F function) &&
    {
      auto [alloc, ready, ptr] = std::move(*this).release();

      try
      {
        auto after_f = bulk_execute_after(ex, std::move(ready), grid_shape, [f = function, ptr = ptr](const S& coord)
        {
          std::invoke(f, coord, ptr);
        });

        // return a new future
        return {std::forward_as_tuple(std::move(after_f), ptr, alloc)};
      }
      catch(...)
      {
        // XXX return an exceptional future
        throw std::runtime_error("future::then_bulk_execute: bulk_execute_after failed");
      }

      // XXX until we can handle exceptions, just return this to make everything compile
      return {std::forward_as_tuple(std::move(ready), ptr, alloc)};
    }


    template<executor Ex, asynchronous_allocator OtherA, event Ev, class... Args, class... Ds, std::invocable<value_type&&,Args&&...> F,
             class R = std::invoke_result_t<F,value_type&&,Args&&...>
            >
    intrusive_future<R, rebind_allocator_result_t<R,OtherA>>
      then_after(const Ex& ex, const OtherA& alloc, Ev&& before, F&& f, intrusive_future<Args,Ds>&&... future_args) &&
    {
      // XXX we can't simply call invoke_after because it #includes this header file
      //return invoke_after(ex, alloc, std::forward<Ev>(before), std::forward<F>(f), std::move(*this), std::move(future_args)...);
      
      // rebind the allocator to the type of the result
      auto rebound_alloc = rebind_allocator<R>(alloc);

      // allocate storage for the result after before is ready
      auto [result_allocation_ready, ptr_to_result] = first_allocate<R>(rebound_alloc, 1);

      // create an event contingent on before, the allocation, and future_args
      auto inputs_ready = contingent_on(ex, std::move(result_allocation_ready), std::forward<Ev>(before), this->ready(), future_args.ready()...);

      try
      {
        // when everything is ready, invoke f and construct the result
        auto result_ready = execute_after(ex, std::move(inputs_ready), [ptr_to_result = ptr_to_result, f = std::move(f), ptr_to_arg1 = this->data(), ... ptrs_to_args = future_args.data()]
        {
          construct_at(ptr_to_result, std::invoke(f, std::move(*ptr_to_arg1), std::move(*ptrs_to_args)...));
        });

        // schedule the deletion of the future_args after the result is ready
        detail::for_each_arg([&](auto&& future_arg)
        {
          auto [alloc, _, ptr] = std::move(future_arg).release();
          finally_delete_after(alloc, result_ready, ptr, 1);
        }, std::move(*this), std::move(future_args)...);

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


    template<executor Ex, asynchronous_allocator OtherA, class... Args, class... As, std::invocable<value_type&&,Args&&...> F,
             class R = std::invoke_result_t<F,value_type&&,Args&&...>
            >
    intrusive_future<R, rebind_allocator_result_t<R,OtherA>>
      then(const Ex& ex, const OtherA& alloc, F&& f, intrusive_future<Args,As>&&... future_args) &&
    {
      // forward to then_after using our ready() event as the before event
      return std::move(*this).then_after(ex, alloc, ready(), std::forward<F>(f), std::move(future_args)...);
    }


    template<executor Ex, class... Args, class... As, std::invocable<value_type&&,Args&&...> F,
             class R = std::invoke_result_t<F,value_type&&,Args&&...>
            >
    auto then(const Ex& ex, F&& f, intrusive_future<Args,As>&&... future_args) &&
    {
      // forward with our allocator as a parameter
      return std::move(*this).then(ex, allocator(), std::forward<F>(f), std::move(future_args)...);
    }


    template<class... Args, class... As, std::invocable<value_type&&,Args&&...> F,
             class R = std::invoke_result_t<F,value_type&&,Args&&...>
            >
      requires executor_associate<A>
    auto then(F&& f, intrusive_future<Args,As>&&... future_args) &&
    {
      // get an executor from our allocator and forward to the more primitive method
      return std::move(*this).then(associated_executor(allocator()), std::forward<F>(f), std::move(future_args)...);
    }

  private:
    std::tuple<allocator_type, event_type, pointer> resources_;
    std::optional<value_type> maybe_result_;
};


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

