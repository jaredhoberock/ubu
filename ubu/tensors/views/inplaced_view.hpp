#pragma once

#include "../../detail/prologue.hpp"
#include "../inplace_tensor.hpp"
#include "view_base.hpp"
#include <concepts>

namespace ubu
{

// inplaced_view is a way to coerce a container with a bounded shape (such as std::array or inplace_tensor)
// to be a view. The elements viewed by inplaced_view are contained within itself.
template<class T, bounded_coordinate S>
  requires std::is_trivially_copy_constructible_v<T>
class inplaced_view : public view_base
{
  public:
    template<view V>
      requires (std::constructible_from<T, tensor_element_t<V>> and std::same_as<S, tensor_shape_t<V>>)
    constexpr inplaced_view(V view)
      : elements_(from_tensor_like, view)
    {}

    constexpr inplaced_view(const inplaced_view&) = default;

    constexpr auto shape() const
    {
      return ubu::shape(elements_);
    }

    constexpr auto size() const
    {
      return elements_.size();
    }

    // note that the return type of operator[] is a value, not a reference
    template<congruent<S> C>
    constexpr auto operator[](C coord) const
    {
      return elements_[coord];
    }

  private:
    inplace_tensor<T,S> elements_;
};

template<view V>
  requires (std::is_trivially_copy_constructible_v<tensor_element_t<V>> and bounded_coordinate<tensor_shape_t<V>>)
inplaced_view(V) -> inplaced_view<tensor_element_t<V>, tensor_shape_t<V>>;

} // end ubu

#include "../../detail/epilogue.hpp"

