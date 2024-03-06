#pragma once

#include "../detail/prologue.hpp"
#include "../tensor/coordinate/concepts/integral_like.hpp"
#include "../tensor/coordinate/traits/default_coordinate.hpp"
#include "cooperator/concepts/allocating_cooperator.hpp"
#include "cooperator/coop_alloca.hpp"
#include "cooperator/coop_dealloca.hpp"
#include "cooperator/synchronize_and_count.hpp"
#include "cooperator/traits/cooperator_size.hpp"
#include <concepts>
#include <optional>

namespace ubu
{

template<class T, allocating_cooperator C, integral_like S = default_coordinate_t<cooperator_size_t<C>>>
class uninitialized_coop_array
{
  public:
    constexpr uninitialized_coop_array(C& self, S size)
      : self_{self},
        size_{size},
        data_{reinterpret_cast<T*>(coop_alloca(self_, size_ * sizeof(T)))}
    {}

    constexpr ~uninitialized_coop_array()
    {
      coop_dealloca(self_, size_ * sizeof(T));
    }

    constexpr S size() const
    {
      return size_;
    }

    template<std::integral I>
    constexpr T& operator[](I i)
    {
      return data_[i];
    }

    template<std::integral I>
    constexpr const T& operator[](I i) const
    {
      return data_[i];
    }

    template<std::integral I>
    constexpr S assign_and_count(C& self, I i, std::optional<T> value)
    {
      if(value)
      {
        operator[](i) = *value;
      }

      return synchronize_and_count(self, value.has_value());
    }

    template<std::integral I>
    constexpr S assign_if_valid_index_and_count(C& self, I i, std::optional<T> value)
    {
      return assign_and_count(self, i, i < size() ? value : std::nullopt);
    }

    template<std::integral I>
    constexpr std::optional<T> shuffle_if_valid_index(C& self, I dest_idx, std::optional<T> value)
    {
      std::optional<T> result;

      if(id(self) < assign_if_valid_index_and_count(self, dest_idx, value))
      {
        result = operator[](id(self));
      }

      return result;
    }

    constexpr T* data() const
    {
      return data_;
    }

  private:
    C& self_;
    [[no_unique_address]] S size_;
    T* data_;  // XXX This member should be a pointer_like related to the result of coop_alloca
};

} // end ubu

#include "../detail/epilogue.hpp"

