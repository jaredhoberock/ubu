#pragma once

#include "../detail/prologue.hpp"

#include "../utilities/integrals/size.hpp"
#include "../utilities/tuples.hpp"
#include "concepts/composable.hpp"
#include "concepts/sized_tensor_like.hpp"
#include "concepts/tensor_like.hpp"
#include "concepts/view.hpp"
#include "coordinates/element.hpp"
#include "element_exists.hpp"
#include "shapes/shape.hpp"
#include "traits/tensor_shape.hpp"
#include "traits/tensor_coordinate.hpp"
#include "vectors/span_like.hpp"
#include "views/all.hpp"
#include "views/compose.hpp"
#include "views/domain.hpp"
#include "views/layouts/layout.hpp"
#include "views/slices/slice.hpp"

namespace ubu
{
namespace detail
{

// composed_tensor and compose have a cyclic dependency and can't use each other directly
// declare detail::invoke_compose for composed_tensor's use
template<class... Args>
constexpr view auto invoke_compose(Args&&... args);

} // end detail


template<class A, view B>
  requires (std::is_object_v<A> and composable<A,B>)
class composed_tensor
{
  public:
    using shape_type = tensor_shape_t<B>;
    using coordinate_type = tensor_coordinate_t<B>;

    template<class OtherA>
      requires std::constructible_from<A,OtherA&&>
    constexpr composed_tensor(OtherA&& a, B b)
      : a_(std::forward<OtherA>(a)), b_(b)
    {}

    composed_tensor(const composed_tensor&) = default;

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

      // if A actually fulfills the requirements of tensor_like,
      // check the coordinate produced by b_ against a_
      // otherwise, we assume that b_ always perfectly covers a_
      if constexpr (tensor_like<A>)
      {
        auto a_coord = element(b_,coord);

        if (not in_domain(a_, a_coord)) return false;
        if (not ubu::element_exists(a_, a_coord)) return false;
      }

      return true;
    }

    constexpr auto a() const
    {
      if constexpr (not tensor_like<A>)
      {
        return a_;
      }
      else
      {
        return ubu::all(a_);
      }
    }

    constexpr B b() const
    {
      return b_;
    }

    constexpr view auto all() const
    {
      return detail::invoke_compose(a_, b_);
    }

    // returns the pair (a(),b())
    constexpr tuples::pair_like auto decompose() const &
    {
      return std::pair(a(), b());
    }

    // returns the pair (a_, b_)
    // this function consumes this composed_tensor
    constexpr std::pair<A,B> decompose() &&
    {
      return std::pair(std::move(a_), std::move(b_));
    }

    // returns a() if it is span_like
    // XXX TODO eliminate this function
    template<span_like S = A>
    constexpr S span() const
    {
      return a();
    }

    template<slicer_for<coordinate_type> K>
    constexpr view auto slice(const K& katana) const
    {
      return detail::invoke_compose(a_, ubu::slice(b_, katana));
    }

  private:
    A a_;
    B b_;
};

template<class A, view B>
  requires (std::is_object_v<A> and composable<A,B>)
composed_tensor(A&&,B) -> composed_tensor<std::remove_cvref_t<A>, B>;


} // end ubu

#include "../detail/epilogue.hpp"

