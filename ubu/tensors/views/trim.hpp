#pragma once

#include "../../detail/prologue.hpp"

#include "../../places/memory/data.hpp"
#include "../concepts/tensor_like.hpp"
#include "../concepts/view.hpp"
#include "../coordinates/bound_coordinate.hpp"
#include "../coordinates/concepts/coordinate.hpp"
#include "../coordinates/element.hpp"
#include "../element_exists.hpp"
#include "../shapes/in_domain.hpp"
#include "../shapes/shape.hpp"
#include "../vectors/fancy_span.hpp"
#include "../vectors/span_like.hpp"
#include "all.hpp"
#include "slices/slice.hpp"
#include "view_base.hpp"
#include <span>
#include <utility>

namespace ubu
{


template<view T, congruent<tensor_shape_t<T>> S>
class trimmed_view : public view_base
{
  public:
    // precondition: is_below_or_equal(shape, ubu::shape(tensor))
    constexpr trimmed_view(T tensor, S shape)
      : tensor_(tensor), shape_(shape)
    {}

    trimmed_view(const trimmed_view&) = default;

    constexpr S shape() const
    {
      return shape_;
    }

    template<congruent<S> C>
    constexpr decltype(auto) operator[](C coord) const
    {
      return element(tensor_, coord);
    }

    template<congruent<S> C>
    constexpr bool element_exists(C coord) const
    {
      return is_below(coord, shape()) and ubu::element_exists(tensor_, coord);
    }

    template<slicer_for<S> K>
    constexpr view auto slice(K katana) const
    {
      return ubu::slice(tensor_, katana);
    }

    // precondition: is_below_or_equal(new_shape, this->shape())
    template<congruent<S> N>
    constexpr trimmed_view<T,N> trim(N new_shape) const
    {
      return {tensor_, new_shape};
    }

  private:
    T tensor_; // XXX we should probably EBO this
    S shape_;
};


namespace detail
{


template<class T, class S>
concept has_trim_member_function = requires(T tensor, S shape)
{
  // .trim() must return a view
  { std::forward<T>(tensor).trim(std::forward<S>(shape)) } -> view;

  // the shape of that view must be congruent with S
  { ubu::shape(std::forward<T>(tensor).trim(std::forward<S>(shape))) } -> congruent<S>;
};

template<class T, class S>
concept has_trim_free_function = requires(T tensor, S shape)
{
  // trim() must return a view
  { trim(std::forward<T>(tensor), std::forward<S>(shape)) } -> view;

  // the shape of that view must be congruent with S
  { ubu::shape(trim(std::forward<T>(tensor), std::forward<S>(shape))) } -> congruent<S>;
};


template<class T, class S>
concept has_trim_customization =
  has_trim_member_function<T,S>
  or has_trim_free_function<T,S>
;


struct dispatch_trim
{
  template<class T, class S>
    requires (has_trim_customization<T,S>
              or (tensor_like<T> and congruent<tensor_shape_t<T>, S>))
  constexpr view auto operator()(T&& tensor, S&& shape) const
  {
    if constexpr (has_trim_member_function<T&&,S&&>)
    {
      return std::forward<T>(tensor).trim(std::forward<S>(shape));
    }
    else if constexpr (has_trim_free_function<T&&,S&&>)
    {
      return trim(std::forward<T>(tensor), std::forward<S>(shape));
    }
    else
    {
      // a precondition on trim is is_below_or_equal(shape, ubu::shape(tensor))
      // this means we can use ubu::shape(tensor) as a bound for the shape of the trimmed result

      // XXX it might be more convenient if we had bound(tensor, shape) as an operation
      auto bounded_shape = bound_coordinate(shape, ubu::shape(tensor));

      if constexpr(span_like<T>)
      {
        // when tensor is span_like, we can simplify the result by returning a fancy_span
        
        // XXX it might be better instead to call a CPO first(span, bounded_shape) or take(span, bounded_shape)
        return fancy_span(data(std::forward<T>(tensor)), bounded_shape);
      }

      return trimmed_view(all(std::forward<T>(tensor)), bounded_shape);
    }
  }
};


} // end detail

constexpr inline detail::dispatch_trim trim;

} // end ubu

#include "../../detail/epilogue.hpp"

