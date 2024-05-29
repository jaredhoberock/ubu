#pragma once

#include "../detail/prologue.hpp"

#include "../miscellaneous/integral/size.hpp"
#include "compose.hpp"
#include "concepts/sized_tensor_like.hpp"
#include "concepts/tensor_like.hpp"
#include "coordinate/element.hpp"
#include "domain.hpp"
#include "element_exists.hpp"
#include "iterator.hpp"
#include "layout/layout.hpp"
#include "shape/shape.hpp"
#include "slice/slice.hpp"
#include "traits/tensor_shape.hpp"
#include "traits/tensor_coordinate.hpp"
#include "vector/span_like.hpp"

namespace ubu
{
namespace detail
{

// view and compose have a cyclic dependency and can't use each other directly
// declare detail::invoke_compose for view's use
template<class... Args>
constexpr auto invoke_compose(Args&&... args);

} // end detail


// XXX consider requiring that Tensor and Layout be trivially relocatable
// i.e., Tensor can't be std::vector, but it could be a view of std::vector (e.g. a pointer into std::vector)

template<class Tensor, layout_for<Tensor> Layout>
class view
{
  public:
    using shape_type = tensor_shape_t<Layout>;
    using coordinate_type = tensor_coordinate_t<Layout>;

    constexpr view(Tensor tensor, Layout layout)
      : tensor_{tensor}, layout_{layout}
    {}

    view(const view&) = default;

    constexpr shape_type shape() const
    {
      return ubu::shape(layout_);
    }

    template<class T_ = Tensor, class L_ = Layout>
      requires (not tensor_like<T_> and sized_tensor_like<L_>)
    constexpr auto size() const
    {
      return ubu::size(layout_);
    }

    // precondition: element_exists(coord)
    template<coordinate_for<Layout> C>
    constexpr decltype(auto) operator[](const C& coord) const
    {
      return element(tensor_, element(layout_,coord));
    }

    // precondition: in_domain(layout(), coord)
    template<coordinate_for<Layout> C>
    constexpr bool element_exists(const C& coord) const
    {
      if (not ubu::element_exists(layout_, coord)) return false;

      auto to_coord = element(layout_,coord);

      // if Tensor actually fulfills the requirements of tensor_like,
      // check the coordinate produced by the layout against tensor_
      // otherwise, we assume that the layout always perfectly covers tensor_
      if constexpr (tensor_like<Tensor>)
      {
        if (not in_domain(tensor_, to_coord)) return false;
        if (not ubu::element_exists(tensor_, to_coord)) return false;
      }

      return true;
    }

    constexpr Tensor tensor() const
    {
      return tensor_;
    }

    constexpr Layout layout() const
    {
      return layout_;
    }

    // begin is a template because tensor_iterator requires its template
    // parameter to be a complete type
    template<class Self = view>
    constexpr tensor_iterator<Self> begin() const
    {
      return {*this};
    }
    
    constexpr tensor_sentinel end() const
    {
      return {};
    }

    template<slicer_for<coordinate_type> K>
    constexpr tensor_like auto slice(const K& katana) const
    {
      return detail::invoke_compose(tensor(), ubu::slice(layout(), katana));
    }

    // returns .tensor() if it is span_like
    // this is just provided to make code a bit more readable
    template<class T_ = Tensor>
      requires span_like<T_>
    constexpr span_like auto span() const
    {
      return tensor();
    }

  private:
    Tensor tensor_;
    Layout layout_;
};

namespace detail
{

// view and compose have a cyclic dependency and can't use each other directly
// define detail::make_view as soon as view's definition is available
template<class T, layout_for<T> L>
constexpr auto make_view(T t, L l)
{
  return view<T,L>(t,l);
}


} // end detail


} // end ubu

#include "../detail/epilogue.hpp"

