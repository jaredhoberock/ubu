#pragma once

#include "../detail/prologue.hpp"

#include "../coordinate/grid_coordinate.hpp"
#include "../detail/for_each_arg.hpp"
#include "../execution/executor/associated_executor.hpp"
#include "../execution/executor/bulk_execute.hpp"
#include "../execution/executor/execute_after.hpp"
#include "../event/event.hpp"
#include "../event/make_complete_event.hpp"
#include "../memory/allocator/asynchronous_allocator.hpp"
#include "../memory/allocator/asynchronous_deallocator.hpp"
#include "../memory/allocator/finally_deallocate_after.hpp"
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

ASPERA_NAMESPACE_OPEN_BRACE


template<class T, asynchronous_deallocator D>
class basic_future;


namespace detail
{


template<class T, template<class...> class Template>
struct is_instance_of : std::false_type {};

template<template<class...> class Template, class... Args>
struct is_instance_of<Template<Args...>, Template> : std::true_type {};

template<class T, template<class...> class Template>
concept instance_of = is_instance_of<T,Template>::value;

template<class F>
concept allocation_future = requires
{
  requires instance_of<F, basic_future>;
  requires std::same_as<typename F::value_type,typename F::pointer>;
};


} // end detail


template<class T, asynchronous_deallocator D>
class basic_future
{
  public:
    // XXX this should be a requires clause, but putting it on all the various declarations of basic_future causes problems with clang
    static_assert(std::same_as<T,allocator_value_t<D>> or std::same_as<T,allocator_pointer_t<D>>);

    using value_type = T;
    using pointer = allocator_pointer_t<D>;
    using event_type = allocator_event_t<D>;
    using size_type = allocator_size_t<D>;

    basic_future(event_type&& ready, std::pair<pointer,size_type> allocation, const D& deallocator) noexcept
      : basic_future{std::forward_as_tuple(std::move(ready), allocation.first, allocation.second), deallocator}
    {}

    basic_future(basic_future&& other) noexcept
      : basic_future{std::move(other).release(), std::move(other.deallocator_)}
    {}

    basic_future& operator=(basic_future&& other) noexcept
    {
      resources_ = std::move(other).release();
      deallocator_ = std::move(other.deallocator_);
      return *this;
    }

    ~basic_future()
    {
      if(data())
      {
        schedule_delete_after(std::move(ready()));
      }
    }

    void wait() const
    {
      ready().wait();
    }

    value_type get()
    {
      if(!data())
      {
        throw std::future_error(std::future_errc::no_state);
      }

      wait();

      value_type result = std::move(*data());

      schedule_delete_after(std::move(ready()));

      return result;
    }

    std::tuple<event_type,pointer,size_type> release() &&
    {
      auto result = std::move(resources_);

      // ensure data() returns null
      std::get<1>(resources_) = nullptr;

      return result;
    }

    event_type release_event() &&
    {
      if(data())
      {
        schedule_delete_after(ready());
      }

      return std::get<0>(std::move(*this).release());
    }

    template<executor E, std::invocable F,
             class R = std::invoke_result_t<F>,
             class Self = basic_future>
      requires detail::allocation_future<Self> and std::constructible_from<allocator_value_t<D>, R>
    basic_future<allocator_value_t<D>,D> then_construct(const E& ex, F&& f) &&
    {
      auto [ready, ptr, n] = std::move(*this).release();

      try
      {
        auto after_f = ex.execute_after(std::move(ready), [f = std::move(f), ptr = ptr]
        {
          construct_at(ptr, std::invoke(f));
        });

        // return a new future
        return {std::forward_as_tuple(std::move(after_f), ptr, n), std::move(deallocator_)};
      }
      catch(...)
      {
        // XXX return an exceptional future
        throw std::runtime_error("future::then_construct: execute_after failed");
      }

      // XXX until we can handle exceptions, just return this to make everything compile
      return {std::forward_as_tuple(std::move(ready), ptr, n), std::move(deallocator_)};
    }


    // invocable<S, T>
    // requires copyable<T>
    template<executor E, grid_coordinate S, std::regular_invocable<S,pointer> F,
             class Self = basic_future
            >
      requires detail::allocation_future<Self>
    basic_future then_bulk_execute(const E& ex, S grid_shape, F function) &&
    {
      auto [ready, ptr, n] = std::move(*this).release();

      try
      {
        // XXX should we pass n as well?
        //     should we pass a const reference to the value or a pointer to the value?
        auto after_f = bulk_execute(ex, std::move(ready), grid_shape, [f = function, ptr = ptr](const S& coord)
        {
          std::invoke(f, coord, ptr);
        });

        // return a new future
        return {std::forward_as_tuple(std::move(after_f), ptr, n), std::move(deallocator_)};
      }
      catch(...)
      {
        // XXX return an exceptional future
        throw std::runtime_error("future::then_bulk_execute: bulk_execute failed");
      }

      // XXX until we can handle exceptions, just return this to make everything compile
      return {std::forward_as_tuple(std::move(ready), ptr, n), std::move(deallocator_)};
    }


    template<std::invocable F,
             class R = std::invoke_result_t<F>,
             class Self = basic_future>
      requires detail::allocation_future<Self> and std::constructible_from<allocator_value_t<D>, R> and executor_associate<D>
    basic_future<allocator_value_t<D>,D> then_construct(F&& f) &&
    {
      // get an executor from our deallocator and forward to the more primitive method
      return std::move(*this).then_construct(associated_executor(deallocator_), std::forward<F>(f));
    }


    template<executor Ex, event Ev, std::invocable F,
             class R = std::invoke_result_t<F>,
             class Self = basic_future>
      requires detail::allocation_future<Self> and std::constructible_from<allocator_value_t<D>, R>
    basic_future<allocator_value_t<D>,D> then_construct_after(const Ex& ex, Ev&& before, F&& f) &&
    {
      // extend our lifetime beyond the before event
      ready() = contingent_on(ex, std::forward<Ev>(before), std::move(ready()));

      // forward to the more primitive method
      return std::move(*this).then_construct(ex, std::forward<F>(f));
    }


    template<executor Ex, event Ev, class... Args, class... Ds, std::invocable<Args&&...> F,
             class R = std::invoke_result_t<F,Args&&...>,
             class Self = basic_future
            >
      requires detail::allocation_future<Self> and std::constructible_from<allocator_value_t<D>, R>
    basic_future<allocator_value_t<D>,D> then_construct_after(const Ex& ex, Ev&& before, F&& f, basic_future<Args,Ds>&&... future_args) &&
    {
      // create an event contingent on before and future_args
      auto all_ready = contingent_on(ex, std::forward<Ev>(before), future_args.ready()...);

      // invoke when everything is ready
      basic_future<R,D> result = std::move(*this).then_construct_after(ex, std::move(all_ready), [f = std::move(f), ... ptrs_to_args = future_args.data()]
      {
        return std::invoke(f, std::move(*ptrs_to_args)...);
      });

      // tell the futures to schedule their deletion after f's result is ready
      detail::for_each_arg([&](auto&& future_arg)
      {
        std::move(future_arg).schedule_delete_after(result.ready());
      }, std::move(future_args)...);

      return result;
    }


    template<executor Ex, asynchronous_allocator A, event Ev, class... Args, class... Ds, std::invocable<value_type&&,Args&&...> F,
             class R = std::invoke_result_t<F,value_type&&,Args&&...>
            >
    basic_future<R, rebind_allocator_result_t<R,A>>
      then_after(const Ex& ex, const A& alloc, Ev&& before, F&& f, basic_future<Args,Ds>&&... future_args) &&
    {
      // 0. asynchronously allocate the output
      auto future_ptr = first_allocate<R>(alloc, 1);

      // 1. schedule f after the before event
      //    pass this future as the second argument of this lambda, which becomes the argument of f
      return std::move(future_ptr).then_construct_after(ex, std::forward<Ev>(before), [f](auto&&... args)
      {
        return std::invoke(f, std::move(args)...);
      }, std::move(*this), std::move(future_args)...);
    }


    template<executor Ex, asynchronous_allocator A, class... Args, class... Ds, std::invocable<value_type&&,Args&&...> F,
             class R = std::invoke_result_t<F,value_type&&,Args&&...>
            >
    basic_future<R, rebind_allocator_result_t<R,A>>
      then(const Ex& ex, const A& alloc, F&& f, basic_future<Args,Ds>&&... future_args) &&
    {
      // forward to then_after using our ready() event as the before event
      return std::move(*this).then_after(ex, alloc, ready(), std::forward<F>(f), std::move(future_args)...);
    }


    template<executor Ex, class... Args, class... Ds, std::invocable<value_type&&,Args&&...> F,
             class R = std::invoke_result_t<F,value_type&&,Args&&...>,
             class A = D
            >
      requires asynchronous_allocator<A>
    auto then(const Ex& ex, F&& f, basic_future<Args,Ds>&&... future_args) &&
    {
      // forward with our deallocator as a parameter
      return std::move(*this).then(ex, deallocator_, std::forward<F>(f), std::move(future_args)...);
    }


    template<class... Args, class... Ds, std::invocable<value_type&&,Args&&...> F,
             class R = std::invoke_result_t<F,value_type&&,Args&&...>
            >
      requires executor_associate<D>
    auto then(F&& f, basic_future<Args,Ds>&&... future_args) &&
    {
      // get an executor from our deallocator and forward to the more primitive method
      return std::move(*this).then(associated_executor(deallocator_), std::forward<F>(f), std::move(future_args)...);
    }


  private:
    // give future access to schedule_delete_after
    template<class OtherT, asynchronous_deallocator OtherD>
    friend class basic_future;

    basic_future(std::tuple<event_type, pointer, size_type>&& resources, const D& deallocator)
      : resources_{std::move(resources)},
        deallocator_{deallocator}
    {}

    event_type& ready()
    {
      return std::get<0>(resources_);
    }

    const event_type& ready() const
    {
      return std::get<0>(resources_);
    }

    pointer data()
    {
      return std::get<1>(resources_);
    }

    template<event E>
    void schedule_delete_after(E&& before)
    {
      auto [_, ptr_to_allocation, allocation_size] = std::move(*this).release();

      // what we do here depends on whether or not data() refers to an object
      // or is simply an uninitialized memory allocation
      //
      // when data() refers to an object, we need to run the objects's destructor prior to memory deallocation
      //
      // data() onlys refer to an uninitialized memory allocation when T is the same type as pointer
      if constexpr(std::same_as<value_type, pointer>)
      {
        // data() is uninitialized memory, so just deallocate it
        finally_deallocate_after(deallocator_, std::forward<E>(before), ptr_to_allocation, allocation_size);
      }
      else
      {
        // data() is an object
        assert(1 == allocation_size);
        finally_delete_after(deallocator_, std::forward<E>(before), ptr_to_allocation, 1);
      }
    }

    std::tuple<event_type, pointer, size_type> resources_;
    D deallocator_;
};


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

