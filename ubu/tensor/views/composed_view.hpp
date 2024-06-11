#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/integral/size.hpp"
#include "../concepts/sized_tensor_like.hpp"
#include "../concepts/tensor_like.hpp"
#include "../coordinates/element.hpp"
#include "../element_exists.hpp"
#include "../shapes/shape.hpp"
#include "../traits/tensor_shape.hpp"
#include "../traits/tensor_coordinate.hpp"
#include "../vectors/span_like.hpp"
#include "domain.hpp"
#include "layouts/layout.hpp"
#include "slices/slice.hpp"
#include "compose.hpp"
#include <ranges>

namespace ubu
{
namespace detail
{

// composed_view and compose have a cyclic dependency and can't use each other directly
// declare detail::invoke_compose for composed_view's use
template<class... Args>
constexpr auto invoke_compose(Args&&... args);

} // end detail


template<class A, layout_for<A> B>
  requires std::is_object_v<A>
class composed_view : public std::ranges::view_base
{
  public:
    using shape_type = tensor_shape_t<B>;
    using coordinate_type = tensor_coordinate_t<B>;

    constexpr composed_view(A a, B b)
      : a_{a}, b_{b}
    {}

    composed_view(const composed_view&) = default;

    constexpr shape_type shape() const
    {
      return ubu::shape(b_);
    }

    template<class A_ = A, class B_ = B>
      requires (not tensor_like<A_> and sized_tensor_like<B_>)
    constexpr auto size() const
    {
      return ubu::size(b_);
    }

    // precondition: element_exists(coord)
    template<coordinate_for<B> C>
    constexpr decltype(auto) operator[](const C& coord) const
    {
      return element(a_, element(b_,coord));
    }

    // precondition: in_domain(b(), coord)
    template<coordinate_for<B> C>
    constexpr bool element_exists(const C& coord) const
    {
      if (not ubu::element_exists(b_, coord)) return false;

      auto a_coord = element(b_,coord);

      // if A actually fulfills the requirements of tensor_like,
      // check the coordinate produced by the layout against a_
      // otherwise, we assume that the layout always perfectly covers a_
      if constexpr (tensor_like<A>)
      {
        if (not in_domain(a_, a_coord)) return false;
        if (not ubu::element_exists(a_, a_coord)) return false;
      }

      return true;
    }

    constexpr A a() const
    {
      return a_;
    }

    constexpr B b() const
    {
      return b_;
    }

    template<slicer_for<coordinate_type> K>
    constexpr tensor_like auto slice(const K& katana) const
    {
      return detail::invoke_compose(a_, ubu::slice(b_, katana));
    }

    // returns .a() if it is span_like
    template<span_like S = A>
    constexpr span_like auto span() const
    {
      return a();
    }

    // returns .b() if it is a layout
    template<ubu::layout L = B>
    constexpr ubu::layout auto layout() const
    {
      return b();
    }

  private:
    A a_;
    B b_;
};

namespace detail
{

// composed_view and compose have a cyclic dependency and can't use each other directly
// define detail::make_composed_view as soon as composed_view's definition is available
template<class A, layout_for<A> B>
  requires std::is_object_v<A>
constexpr auto make_composed_view(A a, B b)
{
  return composed_view<A,B>(a,b);
}


} // end detail


} // end ubu

#include "../../detail/epilogue.hpp"

