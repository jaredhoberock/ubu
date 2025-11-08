#pragma once

#include "../../detail/prologue.hpp"

#include "../concepts/tensor.hpp"
#include "../concepts/view.hpp"
#include "../coordinates/concepts/congruent.hpp"
#include "../coordinates/coordinate_modulo.hpp"
#include "../coordinates/element.hpp"
#include "../element_exists.hpp"
#include "../shapes/in_domain.hpp"
#include "../shapes/shape.hpp"
#include "../traits/tensor_shape.hpp"
#include "all.hpp"
#include "view_base.hpp"
#include <utility>


namespace ubu
{

template<view T, congruent<tensor_shape_t<T>> S>
class tiled_view : public view_base
{
  public:
    constexpr tiled_view(T tensor, S shape)
      : tensor_(tensor), shape_(shape)
    {}

    constexpr S shape() const
    {
      return shape_;
    }

    constexpr decltype(auto) operator[](congruent<S> auto coord) const
    {
      return ubu::element(tensor_, coordinate_modulo(coord, ubu::shape(tensor_)));
    }

    constexpr bool element_exists(congruent<S> auto coord) const
    {
      return is_strictly_inside(coord, shape()) and ubu::element_exists(tensor_, coordinate_modulo(coord, ubu::shape(tensor_)));
    }

  private:
    T tensor_; // XXX EBO this
    S shape_;
};


namespace detail
{


template<class T, class S>
concept has_tile_member_function = requires(T tensor, S shape)
{
  // .tile() must return a view
  { std::forward<T>(tensor).tile(std::forward<S>(shape)) } -> view;

  // the shape of that view must be congruent with S
  { shape(std::forward<T>(tensor).tile(std::forward<S>(shape))) } -> congruent<S>;
};

template<class T, class S>
concept has_tile_free_function = requires(T tensor, S shape)
{
  // tile() must return a view
  { tile(std::forward<T>(tensor), std::forward<S>(shape)) } -> view;

  // the shape of that view must be congruent with S
  { shape(tile(std::forward<T>(tensor), std::forward<S>(shape))) } -> congruent<S>;
};


template<class T, class S>
concept has_tile_customization =
  has_tile_member_function<T,S>
  or has_tile_free_function<T,S>
;


struct dispatch_tile
{
  template<class T, class S>
    requires (has_tile_customization<T,S>
              or (tensor<T> and congruent<tensor_shape_t<T>,S>))
  constexpr view auto operator()(T&& tensor, S&& shape) const
  {
    if constexpr (has_tile_member_function<T&&,S&&>)
    {
      return std::forward<T>(tensor).tile(std::forward<S>(shape));
    }
    else if constexpr (has_tile_free_function<T&&,S&&>)
    {
      return tile(std::forward<T>(tensor), std::forward<S>(shape));
    }
    else
    {
      return tiled_view(all(std::forward<T>(tensor)), std::forward<S>(shape));
    }
  }
};


} // end detail

constexpr inline detail::dispatch_tile tile;

} // end ubu

#include "../../detail/epilogue.hpp"

