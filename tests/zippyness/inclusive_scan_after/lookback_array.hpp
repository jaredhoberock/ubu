#pragma once

#include "maybe_add.hpp"
#include <atomic>
#include <ubu/ubu.hpp>
#include <ubu/detail/atomic.hpp>

// lookback_array implements the decoupled look-back algorithm described here:
// https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

// lookback_storage is separate from lookback_array to accomodate an important storage optimization
template<class T, ubu::semicooperator C, ubu::integral_like Z = ubu::default_coordinate_t<ubu::cooperator_size_t<C>>>
class lookback_storage
{
  public:
    constexpr static ubu::integral_like auto dynamic_size_in_bytes(Z size)
    {
      return size * (2 * sizeof(T) + sizeof(status_t));
    }

    constexpr lookback_storage(C& self, Z size, std::optional<T> carry_in = std::nullopt)
      : self_{self},
        size_{size},
        carry_in_{carry_in},
        sums_{reinterpret_cast<T*>(ubu::coop_alloca(self_, sizeof(T) * size))},
        cumulative_sums_{reinterpret_cast<T*>(ubu::coop_alloca(self_, sizeof(T) * size))},
        statuses_{reinterpret_cast<T*>(ubu::coop_alloca(self_, sizeof(status_t) * size))}
    {}

    // XXX consider eliminating this destructor if C is not a cooperator
    //     i.e., can't synchronize
    constexpr ~lookback_storage()
    {
      ubu::coop_dealloca(self_, dynamic_size_in_bytes(size_));
    }

    // precondition: 0 <= idx and idx < size
    template<ubu::integral_like I>
    constexpr void store_sum(I idx, const T& value)
    {
      sums_[idx] = value;
      ubu::detail::store_release(statuses_ + idx, sum_available);
    }

    // precondition: 0 <= idx and idx < size
    template<ubu::integral_like I>
    constexpr void store_cumulative_sum(I idx, const T& value)
    {
      cumulative_sums_[idx] = value;
      ubu::detail::store_release(statuses_ + idx, cumulative_sum_available);
    }

    // precondition: idx < size()
    // when idx is < -1, (std::nullopt, cumulative_sum_available) is returned
    // when idx is == -1, (carry_in, cumulative_sum_available) is returned
    // otherwise, (value, status) is returned
    template<ubu::cooperator S, ubu::integral_like I>
    constexpr std::pair<std::optional<T>,bool> coop_wait_and_load(S self, I idx) const
    {
      using namespace ubu;

      status_t status = unavailable;
      do
      {
        // XXX delay here

        status = idx >= 0 ?
          detail::load_acquire(statuses_ + idx) :
          cumulative_sum_available
        ;
      }
      while(synchronize_and_any(self, status == unavailable));

      std::optional<T> result;

      if(idx == -1)
      {
        result = carry_in_;
      }
      else if(idx >= 0)
      {
        // choose which array to load from
        T* ptr = (status == sum_available) ?
          sums_ :
          cumulative_sums_
        ;

        result = ptr[idx];
      }
    
      return {result, status == cumulative_sum_available};
    }

  private:
    enum status_t
    {
      unavailable = 0,
      sum_available,
      cumulative_sum_available
    };

    C& self_;
    [[no_unique_address]] Z size_;
    std::optional<T> carry_in_;
    T* sums_;
    T* cumulative_sums_;
    status_t* statuses_;
};


// optimization for where T is trivial and value_and_status can fit into a word
template<class T, ubu::semicooperator C, ubu::integral_like Z>
  requires (std::is_trivial_v<T> and (sizeof(T) + sizeof(int) <= sizeof(std::uint64_t)))
class lookback_storage<T,C,Z>
{
  public:
    constexpr static ubu::integral_like auto dynamic_size_in_bytes(Z size)
    {
      return size * sizeof(value_and_status);
    }

    constexpr lookback_storage(C& self, Z size, std::optional<T> carry_in = std::nullopt)
      : self_{self},
        size_{size},
        carry_in_{carry_in},
        data_{reinterpret_cast<value_and_status*>(ubu::coop_alloca(self_, dynamic_size_in_bytes(size)))}
    {}

    constexpr ~lookback_storage()
    {
      ubu::coop_dealloca(self_, dynamic_size_in_bytes(size_));
    }

    // precondition: 0 <= idx and idx < size()
    template<ubu::integral_like I>
    constexpr void store_sum(I idx, const T& value)
    {
      value_and_status store_me{value, sum_available};
      ubu::detail::atomic_store(data_ + idx, store_me, std::memory_order_relaxed);
    }

    // precondition: 0 <= idx and idx < size()
    template<ubu::integral_like I>
    constexpr void store_cumulative_sum(I idx, const T& value)
    {
      value_and_status store_me{value, cumulative_sum_available};
      ubu::detail::atomic_store(data_ + idx, store_me, std::memory_order_relaxed);
    }

    // precondition: idx < size()
    // when idx is < -1, (std::nullopt, cumulative_sum_available) is returned
    // when idx is == -1, (carry_in, cumulative_sum_available) is returned
    // otherwise, (value, status) is returned
    // XXX should require that S be some subgroup of C
    template<ubu::cooperator S, ubu::integral_like I>
    constexpr std::pair<std::optional<T>,bool> coop_wait_and_load(S self, I idx) const
    {
      using namespace ubu;

      value_and_status result = checked_load(idx);

      while(synchronize_and_any(self, result.status == unavailable))
      {
        result = checked_load(idx);
      }

      // idx == -1 is special and receives the carry-in
      std::optional<T> opt = (idx < -1) ? std::nullopt : idx == -1 ? carry_in_ : result.value;

      return {opt, result.status == cumulative_sum_available};
    }

  private:
    enum status_t
    {
      unavailable = 0,
      sum_available,
      cumulative_sum_available
    };

    struct value_and_status
    {
      T value;
      status_t status;
    };

    // precondition: idx < size()
    // when idx is < 0, (T{}, cumulative_sum_available) is returned
    // otherwise, (value, status) is returned
    template<ubu::integral_like I>
    constexpr value_and_status checked_load(I idx) const
    {
      return idx >= 0 ?
        ubu::detail::atomic_load(data_ + idx, std::memory_order_relaxed) :
        value_and_status{{}, cumulative_sum_available}
      ;
    }

    C& self_;
    [[no_unique_address]] Z size_;
    std::optional<T> carry_in_;
    value_and_status* data_;
};


// XXX we need to require that C have a buffer
template<class T, ubu::semicooperator C, ubu::integral_like Z = ubu::default_coordinate_t<ubu::cooperator_size_t<C>>>
class lookback_array
{
  public:
    constexpr static ubu::integral_like auto dynamic_size_in_bytes(Z size)
    {
      return lookback_storage<T,C,Z>::dynamic_size_in_bytes(size);
    }

    constexpr lookback_array(C& self, Z size, std::optional<T> carry_in = std::nullopt)
      : storage_{self, size, carry_in}
    {}

    // Store the leader's value of sum; the value of sum in all other threads is ignored
    // precondition: The value of idx is uniform across the group of threads
    // The carry-in corresponding to idx is returned to thread 0, all other threads receive std::nullopt
    // XXX should require that S be some subgroup of C
    template<ubu::cooperator S, ubu::integral_like I, class Function>
    constexpr std::optional<T> coop_store_sum_and_load_carry_in(S self, I idx, std::optional<T> sum, Function op)
    {
      using namespace ubu;

      std::optional<T> carry_in;

      // first, store our idx's sum
      if(is_leader(self))
      {
        storage_.store_sum(idx, *sum);
      }

      // sum backwards through a group-wide window of indices, stopping
      // when we encounter a cumulative sum
      auto predecessor_idx = idx - size(self) + id(self);
      while(true)
      {
        // wait until the entire window of sums is available
        auto [value, value_is_cumulative] = storage_.coop_wait_and_load(self, predecessor_idx);

        // the result of this reduction is returned to the leader
        std::optional window_sum = coop_reduce_last_span(self, value, value_is_cumulative, op);

        // accumulate the sum of the window into the carry_in
        carry_in = maybe_add(window_sum, carry_in, op);

        if(synchronize_and_any(self, value_is_cumulative))
        {
          break;
        }

        predecessor_idx -= size(self);
      }

      // accumulate the sum of all values "to the left" of idx and store
      if(is_leader(self))
      {
        sum = maybe_add(carry_in, sum, op);
        storage_.store_cumulative_sum(idx, *sum);
      }

      synchronize(self);

      return is_leader(self) ? carry_in : std::nullopt;
    }

  private:
    lookback_storage<T,C,Z> storage_;
};

template<ubu::semicooperator C, ubu::integral_like Z, class T>
lookback_array(C,Z,T) -> lookback_array<T,C,Z>;


// this function is just a way to workaround the problem of
// needing a cooperator type to instantiate lookback_array
// this allows us to calculate the dynamic storage requirements
// of lookback_array outside of a kernel before we have a cooperator
template<class T, class Z>
constexpr std::size_t dynamic_size_in_bytes_of_lookback_array(Z size)
{
  using C = ubu::basic_cooperator<ubu::empty_buffer, int>;
  return lookback_array<T,C,Z>::dynamic_size_in_bytes(size);
}

