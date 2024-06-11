#pragma once

#include <atomic>
#include <concepts>
#include <ubu/ubu.hpp>

// XXX could use enable_plus_like, enable_minus_like, etc.

template<class F, class T>
concept plus_like =
  std::regular_invocable<F,T,T> and
  std::same_as<std::invoke_result_t<F,T,T>, T> and
  (std::same_as<F,std::plus<>> or std::same_as<F,std::plus<T>>)
;

template<class F, class T>
concept minus_like =
  std::regular_invocable<F,T,T> and
  std::same_as<std::invoke_result_t<F,T,T>, T> and
  (std::same_as<F,std::minus<>> or std::same_as<F,std::minus<T>>)
;

template<class F, class T>
concept max_like =
  std::regular_invocable<F,T,T> and
  std::same_as<std::invoke_result_t<F,T,T>, T> and
  std::same_as<F,decltype(ubu::larger)>
;

template<class F, class T>
concept min_like =
  std::regular_invocable<F,T,T> and
  std::same_as<std::invoke_result_t<F,T,T>, T> and
  std::same_as<F,decltype(ubu::smaller)>
;

template<class T>
concept indirectly_rw =
  std::indirectly_readable<T> and
  std::indirectly_writable<T,std::iter_value_t<T>>
;

template<class F, class I, class T>
concept indirectly_atomic_invocable =
  indirectly_rw<I> and
  std::is_reference_v<std::iter_reference_t<I>> and
  (std::integral<std::iter_value_t<I>> or std::floating_point<std::iter_value_t<I>>) and
  std::convertible_to<T,std::iter_value_t<I>> and
  (plus_like<F,std::iter_value_t<I>> or minus_like<F,std::iter_value_t<I>> or min_like<F,std::iter_value_t<I>> or max_like<F,std::iter_value_t<I>>)
;

template<indirectly_rw I, ubu::semicooperator C, std::regular_invocable<std::iter_value_t<I>,std::iter_value_t<I>> F>
class atomic_accumulator
{
  public:
    constexpr static int dynamic_size_in_bytes()
    {
      // we just need one boolean
      return sizeof(bool);
    }

    constexpr atomic_accumulator(C& self, I result_iter, F op)
      : self_{self},
        result_iter_{result_iter},
        locked_{reinterpret_cast<bool*>(ubu::coop_alloca(self_, sizeof(bool)))},
        op_{op}
    {}

    template<class T>
      requires std::regular_invocable<F, std::iter_reference_t<I>,T>
    constexpr void accumulate(const T& value)
    {
      if constexpr (indirectly_atomic_invocable<F,I,T>)
      {
        atomic_accumulate(value);
      }
      else
      {
        locking_accumulate(value);
      }
    }

  private:
    constexpr void lock()
    {
      bool expected = false;
      while(not std::atomic_ref(*locked_).compare_exchange_strong(expected, true, std::memory_order_acquire))
      {
        // busy wait, reset expected value
        expected = false;
      }
    }

    constexpr void unlock()
    {
      std::atomic_ref(*locked_).store(false, std::memory_order_release);
    }

    template<class T, std::regular_invocable<std::iter_reference_t<I>,T> F_ = F>
    constexpr void locking_accumulate(const T& value)
    {
      // in the general case, we need to lock & unlock the mutex to update the result

      lock();

      if constexpr (std::is_reference_v<std::iter_reference_t<I>>)
      {
        // if we are able to get an actual reference to the value being updated, then we can use atomic loads and stores
        auto old_result = std::atomic_ref(*result_iter_).load(std::memory_order_relaxed);
        auto new_result = op_(old_result, value);
        std::atomic_ref(*result_iter_).store(new_result, std::memory_order_relaxed);
      }
      else
      {
        // XXX circle build 201's old LLVM cannot handle this memory_order_acquire fence
        // std::atomic_thread_fence(std::memory_order_acquire);
        __threadfence();

        auto old_result = *result_iter_;
        auto new_result = op_(old_result, value);
        *result_iter_ = new_result;

        // XXX circle build 201's old LLVM cannot handle this memory_order_release fence
        // std::atomic_thread_fence(std::memory_order_release);
        __threadfence();
      }

      unlock();
    }

    template<class T, indirectly_atomic_invocable<I,T> F_ = F>
    constexpr void atomic_accumulate(const T& value)
    {
      using R = std::iter_value_t<I>;

      if constexpr (std::integral<R>)
      {
        if constexpr (plus_like<F,R>)
        {
          std::atomic_ref(*result_iter_).fetch_add(value);
        }
        else if constexpr (minus_like<F,R>)
        {
          std::atomic_ref(*result_iter_).fetch_sub(value);
        }
        else if constexpr (min_like<F,R>)
        {
          std::atomic_ref(*result_iter_).fetch_min(value);
        }
        else
        {
          std::atomic_ref(*result_iter_).fetch_max(value);
        }
      }
      else
      {
        if constexpr (plus_like<F,R>)
        {
          atomicAdd(*result_iter_, value);
        }
        else if constexpr (minus_like<F,R>)
        {
          atomicSub(*result_iter_, value);
        }
        else if constexpr (min_like<F,R>)
        {
          atomicMin(*result_iter_, value);
        }
        else
        {
          atomicMax(*result_iter_, value);
        }
      }
    }

    C& self_;
    I result_iter_;
    bool* locked_;
    [[no_unique_address]] F op_;
};

template<indirectly_rw I, class F>
constexpr std::size_t dynamic_size_in_bytes_of_atomic_accumulator(I result_iter, F op)
{
  using C = ubu::basic_cooperator<ubu::empty_buffer, int>;
  return atomic_accumulator<I,C,F>::dynamic_size_in_bytes();
}

