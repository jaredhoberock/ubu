#pragma once

#include "../../detail/prologue.hpp"

#include "../../utilities/integrals/size.hpp"
#include "../../utilities/tuples.hpp"
#include "../concepts/composable.hpp"
#include "../concepts/sized_tensor_like.hpp"
#include "../concepts/tensor_like.hpp"
#include "../concepts/view.hpp"
#include "../coordinates/element.hpp"
#include "../element_exists.hpp"
#include "../shapes/in_domain.hpp"
#include "../shapes/shape.hpp"
#include "../traits/tensor_shape.hpp"
#include "../traits/tensor_coordinate.hpp"
#include "../vectors/span_like.hpp"
#include "all.hpp"
#include "compose.hpp"
#include "slices/slice.hpp"
#include "slices/slicer.hpp"
#include "view_base.hpp"


namespace ubu
{
namespace detail
{

// composed_view and compose have a cyclic dependency and can't use each other directly
// declare detail::invoke_compose for composed_view's use
template<class... Args>
constexpr view auto invoke_compose(Args&&... args);

// composed_view and slice have a cyclic dependency and can't use each other directly
// declare detail::invoke_slice for composed_view's use
template<class... Args>
constexpr view auto invoke_slice(Args&&... args);

} // end detail


template<class A, view B>
  requires (std::is_trivially_copy_constructible_v<A> and composable<A,B> and (view<A> or not tensor_like<A>))
class composed_view : public view_base
{
  public:
    using shape_type = tensor_shape_t<B>;
    using coordinate_type = tensor_coordinate_t<B>;

    constexpr composed_view(A a, B b)
      : a_(a), b_(b)
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

    template<coordinate_for<B> C>
    constexpr bool element_exists(const C& coord) const
    {
      if (not ubu::element_exists(b_, coord)) return false;

      // avoid evaluating element(b_, coord) if we don't need to
      if constexpr (shaped<A>)
      {
        auto a_coord = element(b_, coord);

        return ubu::element_exists(a_, a_coord);
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

    // returns the pair (a(),b())
    constexpr std::pair<A,B> decompose() const
    {
      return {a(), b()};
    }

    // returns a() if it is span_like
    // XXX TODO eliminate this function
    template<span_like S = A>
    constexpr S span() const
    {
      return a();
    }

    template<slicer_for<shape_type> K>
    constexpr view auto slice(const K& katana) const
    {
      return detail::invoke_compose(a_, detail::invoke_slice(b_, katana));
    }

    template<class C>
      requires composable<composed_view,C&&>
    constexpr view auto compose(C&& c) const
    {
      return detail::invoke_compose(a_, detail::invoke_compose(b_, std::forward<C>(c)));
    }

  private:
    A a_;
    B b_;
};


namespace detail
{

// composed_view and compose have a cyclic dependency and can't use each other directly
// define make_composed_view for compose's use as soon as composed_view is available
template<class A, view B>
  requires (std::is_trivially_copy_constructible_v<A> and composable<A,B> and (view<A> or not tensor_like<A>))
constexpr view auto make_composed_view(A a, B b)
{
  return composed_view(a,b);
}

} // end detail

} // end ubu

#include "../../detail/epilogue.hpp"

