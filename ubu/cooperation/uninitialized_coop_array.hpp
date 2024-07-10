#pragma once

#include "../detail/prologue.hpp"
#include "../miscellaneous/integrals/integral_like.hpp"
#include "../tensors/coordinates/traits/default_coordinate.hpp"
#include "../tensors/traits/tensor_element.hpp"
#include "../tensors/traits/tensor_size.hpp"
#include "../tensors/vectors/fancy_span.hpp"
#include "../tensors/vectors/sized_vector_like.hpp"
#include "algorithms/coop_copy.hpp"
#include "primitives/concepts/allocating_cooperator.hpp"
#include "primitives/coop_alloca.hpp"
#include "primitives/coop_dealloca.hpp"
#include "primitives/synchronize_and_count.hpp"
#include "primitives/traits/cooperator_size.hpp"
#include <concepts>
#include <optional>
#include <memory>

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

    template<sized_vector_like V>
      requires std::constructible_from<T,tensor_element_t<V>>
    constexpr uninitialized_coop_array(C& self, V source)
      : uninitialized_coop_array(self, source.size())
    {
      coop_copy(self, source, all());
    }

    constexpr ~uninitialized_coop_array()
    {
      coop_dealloca(self_, size_ * sizeof(T));
    }

    constexpr fancy_span<T*,S> all() const
    {
      return {data(), size()};
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

    template<std::integral I, class... Args>
      requires std::constructible_from<T,Args&&...>
    constexpr T& construct_at(I i, Args&&... args)
    {
      return *std::construct_at(&data_[i], std::forward<Args>(args)...);
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

template<allocating_cooperator C, sized_vector_like V>
uninitialized_coop_array(C,V) -> uninitialized_coop_array<tensor_element_t<V>,C,tensor_size_t<V>>;

} // end ubu

#include "../detail/epilogue.hpp"

