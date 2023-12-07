#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/for_each_arg.hpp"
#include "../../execution/executor/bulk_execute_after.hpp"
#include "../../execution/executor/concepts/executor.hpp"
#include "../../execution/executor/traits/executor_workspace_shape.hpp"
#include "../../grid/coordinate/coordinate.hpp"
#include "../../grid/coordinate/zeros.hpp"
#include "../../memory/allocator/allocator_delete.hpp"
#include "../../memory/allocator/concepts/asynchronous_allocator.hpp"
#include "../../memory/allocator/finally_delete_after.hpp"
#include "../../memory/allocator/first_allocate.hpp"
#include "../../memory/allocator/traits.hpp"
#include "../../memory/pointer/construct_at.hpp"
#include "../happening.hpp"
#include <cassert>
#include <concepts>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>


namespace ubu
{


// XXX rename this to something else
template<class T, asynchronous_allocator_of<T> A, executor E, happening H = allocator_happening_t<A>>
class intrusive_future
{
  public:
    using value_type = T;
    using pointer = allocator_pointer_t<A,T>;
    using happening_type = H;
    using allocator_type = A;
    using executor_type = E;

    intrusive_future(std::tuple<allocator_type, executor_type, happening_type, pointer>&& resources)
      : resources_{std::move(resources)}
    {}

    intrusive_future(happening_type&& ready, pointer ptr_to_value, const A& allocator, const E& executor) noexcept
      : intrusive_future{std::forward_as_tuple(allocator, executor, std::move(ready), ptr_to_value)}
    {}

    intrusive_future(intrusive_future&& other) noexcept
      : resources_{std::move(other.resources_)},
        maybe_result_{std::move(other.maybe_result_)}
    {
      std::get<pointer>(other.resources_) = nullptr;
    }

    ~intrusive_future()
    {
      if(data())
      {
        auto [alloc, exec, ready, ptr] = std::move(*this).release();
        finally_delete_after(alloc, exec, std::move(ready), ptr, 1);
      }
    }

    const happening_type& ready() const
    {
      return std::get<happening_type>(resources_);
    }

    pointer data() const
    {
      return std::get<pointer>(resources_);
    }

    allocator_type allocator() const
    {
      return std::get<allocator_type>(resources_);
    }

    executor_type executor() const
    {
      return std::get<executor_type>(resources_);
    }

    void wait()
    {
      // either the allocation or the result needs to exist for this future to be valid
      if(!data() and !maybe_result_.has_value())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      // wait on the readiness of the result happening
      std::get<happening_type>(resources_).wait();

      // get the result if we haven't already
      if(!maybe_result_.has_value())
      {
        value_type result = std::move(*data());
        maybe_result_.emplace(std::move(result));

        // release resources
        auto [alloc, ex, ready, ptr] = std::move(*this).release();
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

    std::tuple<allocator_type,executor_type,happening_type,pointer> release() &&
    {
      auto result = std::move(resources_);

      // ensure data() returns null
      std::get<pointer>(resources_) = nullptr;

      return result;
    }

    // invocable<S, T>
    // requires copyable<T>
    // XXX we should pass f a (raw) reference to the value rather than a pointer
    template<ubu::executor Ex, coordinate S, std::regular_invocable<S,pointer> F,
             class Self = intrusive_future
            >
    intrusive_future then_bulk_execute(const Ex& ex, S grid_shape, F function) &&
    {
      auto [alloc, ready, ptr] = std::move(*this).release();

      try
      {
        // we don't need a workspace
        auto workspace_shape = zeros<executor_workspace_shape_t<Ex>>;

        auto after_f = old_bulk_execute_after(ex, std::move(ready), grid_shape, workspace_shape, [f = function, ptr = ptr](const S& coord, auto)
        {
          std::invoke(f, coord, ptr);
        });

        // return a new future
        return {std::forward_as_tuple(std::move(after_f), ptr, alloc)};
      }
      catch(...)
      {
        // XXX return an exceptional future
        throw std::runtime_error("future::then_bulk_execute: old_bulk_execute_after failed");
      }

      // XXX until we can handle exceptions, just return this to make everything compile
      return {std::forward_as_tuple(std::move(ready), ptr, alloc)};
    }


    template<ubu::executor OtherE, happening OtherH, class... Args, class... As, class... Es,
             std::invocable<value_type&&,Args&&...> F,
             class R = std::invoke_result_t<F,value_type&&,Args&&...>,
             asynchronous_allocator_of<R> OtherA
            >
    intrusive_future<R, OtherA, OtherE>
      then_after(const OtherE& ex, const OtherA& alloc, OtherH&& before, F&& f, intrusive_future<Args,As,Es>&&... future_args) &&
    {
      // XXX we can't simply call invoke_after because it #includes this header file
      //return invoke_after(ex, alloc, std::forward<H>(before), std::forward<F>(f), std::move(*this), std::move(future_args)...);

      // allocate storage for the result after before is ready
      auto [result_allocation_ready, ptr_to_result] = first_allocate<R>(alloc, 1);

      // create a happening dependent on before, the allocation, and future_args
      auto inputs_ready = dependent_on(ex, std::move(result_allocation_ready), std::forward<OtherH>(before), this->ready(), future_args.ready()...);

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
          auto [alloc, exec, _, ptr] = std::move(future_arg).release();
          finally_delete_after(alloc, exec, result_ready, ptr, 1);
        }, std::move(*this), std::move(future_args)...);

        // return a new future
        return {std::move(result_ready), ptr_to_result, alloc, ex};
      }
      catch(...)
      {
        // XXX return an exceptional future
        throw std::runtime_error("invoke_after: execute_after failed");
      }

      // XXX until we can handle exceptions, just return this to make everything compile
      return {std::move(result_allocation_ready), ptr_to_result, alloc, ex};
    }


    template<ubu::executor OtherE, class... Args, class... As, class... Es,
             std::invocable<value_type&&,Args&&...> F,
             class R = std::invoke_result_t<F,value_type&&,Args&&...>,
             asynchronous_allocator_of<R> OtherA
            >
    intrusive_future<R, OtherA, OtherE>
      then(const OtherE& ex, const OtherA& alloc, F&& f, intrusive_future<Args,As,Es>&&... future_args) &&
    {
      // forward to then_after using our ready() happening as the before happening
      return std::move(*this).then_after(ex, alloc, ready(), std::forward<F>(f), std::move(future_args)...);
    }


    template<ubu::executor OtherE, class... Args, class... As, class... Es,
             std::invocable<value_type&&,Args&&...> F,
             class R = std::invoke_result_t<F,value_type&&,Args&&...>
            >
    auto then(const OtherE& ex, F&& f, intrusive_future<Args,As,Es>&&... future_args) &&
    {
      // forward with our allocator as a parameter
      return std::move(*this).then(ex, allocator(), std::forward<F>(f), std::move(future_args)...);
    }


    template<class... Args, class... As, class... Es,
             std::invocable<value_type&&,Args&&...> F,
             class R = std::invoke_result_t<F,value_type&&,Args&&...>
            >
    auto then(F&& f, intrusive_future<Args,As,Es>&&... future_args) &&
    {
      // use our allocator and forward to the more primitive method
      return std::move(*this).then(executor(), std::forward<F>(f), std::move(future_args)...);
    }

  private:
    std::tuple<allocator_type, executor_type, happening_type, pointer> resources_;
    std::optional<value_type> maybe_result_;
};


} // end ubu


#include "../../detail/epilogue.hpp"

