#pragma once

#include "../detail/prologue.hpp"
#include "../miscellaneous/integrals/ceil_div.hpp"
#include "../miscellaneous/integrals/integral_like.hpp"
#include "../tensors/coordinates/traits/default_coordinate.hpp"
#include "cooperators/concepts/allocating_cooperator.hpp"
#include "cooperators/coop_alloca.hpp"
#include "cooperators/coop_dealloca.hpp"
#include "cooperators/traits/cooperator_size.hpp"
#include <atomic>
#include <concepts>
#include <optional>

namespace ubu
{


// this is an optimized form of uninitialized_coop_array<std::optional<T>,C>
// it compresses each std::optional's valid bit into a bitset stored separately from the values
template<class T, ubu::allocating_cooperator C, ubu::integral_like S = ubu::default_coordinate_t<ubu::cooperator_size_t<C>>>
class uninitialized_coop_optional_array
{
  public:
    constexpr uninitialized_coop_optional_array(C& self, S size)
      : self_{self},
        size_{size},
        allocation_{reinterpret_cast<T*>(ubu::coop_alloca(self_, allocation_size(size)))}
    {}

    constexpr ~uninitialized_coop_optional_array()
    {
      ubu::coop_dealloca(self_, allocation_size(size_));
    }

    constexpr S size() const
    {
      return size_;
    }

    template<std::integral I>
    struct reference
    {
      uninitialized_coop_optional_array& array;
      I i;

      constexpr void operator=(const std::optional<T>& value) const
      {
        array.put(i, value);
      }

      constexpr operator std::optional<T> () const
      {
        return array.get(i);
      }
    };

    template<std::integral I>
    constexpr reference<I> operator[](I i) const
    {
      return {const_cast<uninitialized_coop_optional_array&>(*this), i};
    }

    template<std::integral I>
    constexpr std::optional<T> get(I i) const
    {
      std::optional<T> result;

      if(is_valid(i))
      {
        result = values()[i];
      }

      return result;
    }

    template<std::integral I>
    constexpr void put(I i, const std::optional<T>& value)
    {
      // XXX attempts at refactoring this to hide the bit manipulation
      //     significantly increase register usage

      // find the word containing i's valid bit
      std::atomic_ref word(valid_bits()[i / sizeof(std::uint32_t)]);

      // select the bit within the word
      int bit = 1 << (i % sizeof(std::uint32_t));

      if(value)
      {
        values()[i] = *value;

        // set_valid(i)
        word.fetch_or(bit, std::memory_order_relaxed);
      }
      else
      {
        // set_invalid(i)
        word.fetch_and(~bit, std::memory_order_relaxed);
      }
    }

  private:
    constexpr static std::size_t allocation_size(S size)
    {
      // XXX probably needs to account for T's alignment
      return size * sizeof(T) + ubu::ceil_div(size, sizeof(std::uint32_t));
    }

    constexpr T* values() const
    {
      return allocation_;
    }

    constexpr std::uint32_t* valid_bits() const
    {
      T* values_end = values() + size();

      // XXX probably needs to account for T's alignment
      return reinterpret_cast<std::uint32_t*>(values_end);
    }

    template<std::integral I>
    constexpr bool is_valid(I i) const
    {
      auto word = i / sizeof(std::uint32_t);
      auto bit = i % sizeof(std::uint32_t);
      return valid_bits()[word] & (1 << bit);
    }

    C& self_;
    [[no_unique_address]] S size_;
    T* allocation_;
};


} // end ubu

#include "../detail/epilogue.hpp"

