#pragma once

#include "../detail/prologue.hpp"

#include "../utilities/constant.hpp"
#include "../utilities/integrals/integral_like.hpp"
#include "concepts/tensor_like.hpp"
#include "concepts/view.hpp"
#include "coordinates/concepts/bounded_coordinate.hpp"
#include "coordinates/concepts/congruent.hpp"
#include "coordinates/traits/maximum.hpp"
#include "shapes/shape.hpp"
#include "shapes/shape_size.hpp"
#include "traits/tensor_element.hpp"
#include "traits/tensor_shape.hpp"
#include "vectors/inplace_vector.hpp"
#include "views/compose.hpp"
#include "views/layouts/compact_left_major_layout.hpp"
#include "views/layouts/concepts/layout.hpp"
#include "views/layouts/lifting_layout.hpp"
#include <type_traits>
#include <utility>

namespace ubu
{

struct from_tensor_like_t {};

constexpr inline from_tensor_like_t from_tensor_like;

template<class T, bounded_coordinate S>
class inplace_tensor
{
  public:
    using shape_type = S;

    inplace_tensor() = default;

    constexpr inplace_tensor(shape_type shape, const T& value)
      : shape_{shape}, elements_(size(), value)
    {}

    constexpr inplace_tensor(shape_type shape)
      : inplace_tensor(shape, T{})
    {}

    // view ctor
    // effects: for each coord in domain(other), constructs (*this)[coord] from element(other, coord)
    // postcondition: shape() == shape(other)
    template<tensor_like O>
      requires (view<O> and congruent<shape_t<O>, S>)
    constexpr inplace_tensor(from_tensor_like_t, O other)
      : shape_(ubu::shape(other)),
        elements_(from_vector_like, ubu::compose(other, inverse_layout()))
    {}

    inplace_tensor(const inplace_tensor&) = default;

    constexpr inplace_tensor(const inplace_tensor& other) requires(std::is_copy_constructible_v<T> and not std::is_trivially_copy_constructible_v<T>)
      : shape_(other.shape()),
        elements_(other.elements_)
    {}

    constexpr inplace_tensor(inplace_tensor&& other) requires(std::is_move_constructible_v<T> and not std::is_trivially_move_constructible_v<T>)
      : shape_(other.shape()),
        elements_(std::move(other.elements_))
    {}

    ~inplace_tensor() = default;

    constexpr auto begin() noexcept
    {
      return elements_.begin();
    }

    constexpr auto begin() const noexcept
    {
      return elements_.begin();
    }

    constexpr auto end() noexcept
    {
      return elements_.end();
    }

    constexpr auto end() const noexcept
    {
      return elements_.end();
    }

    constexpr S shape() const
    {
      return shape_;
    }

    constexpr integral_like auto size() const
    {
      return layout().size();
    }

    constexpr static ubu::integral_like auto max_size()
    {
      constexpr integral_like auto N = shape_size(maximum_v<S>);

      // convert the result to an int for constant
      return constant<int(N)>{};
    }

    // returns the layout : shape -> size
    constexpr compact_left_major_layout<S> layout() const
    {
      return {shape()};
    }

    // returns the layout : size -> shape
    constexpr ubu::layout auto inverse_layout() const
    {
      return lifting_layout(size(), shape());
    }

    template<congruent<S> C>
    constexpr const T& operator[](C coord) const
    {
      return elements_[layout()[coord]];
    }

    template<congruent<S> C>
    constexpr T& operator[](C coord)
    {
      return elements_[layout()[coord]];
    }

    constexpr view auto all()
    {
      return compose(elements_, layout());
    }

    constexpr view auto all() const
    {
      return compose(elements_, layout());
    }

    // XXX consider providing .data() because inplace_tensor's layout is compact

  private:
    S shape_;

    // XXX it's a little wasteful to use inplace_vector here because we
    //     don't need to store its size separately from our shape_
    inplace_vector<T, max_size()> elements_;
};

template<tensor_like T>
  requires shaped_and_bounded<T>
inplace_tensor(from_tensor_like_t, T&&) -> inplace_tensor<tensor_element_t<T&&>, tensor_shape_t<T&&>>; 

} // end ubu

#include "../detail/epilogue.hpp"

