#pragma once

#include "../../detail/prologue.hpp"

#include "../coordinates/element.hpp"
#include "../element_exists.hpp"
#include "../concepts/view.hpp"
#include "../traits/tensor_reference.hpp"
#include "all.hpp"
#include "view_base.hpp"
#include <utility>

namespace ubu
{


template<view V>
class enumerated_view : public view_base
{
  public:
    constexpr enumerated_view(V tensor)
      : tensor_(tensor)
    {}

    enumerated_view(const enumerated_view&) = default; 

    using shape_type = tensor_shape_t<V>;

    constexpr shape_type shape() const
    {
      return ubu::shape(tensor_);
    }

    // returns the pair (coord, tensor[coord])
    template<congruent<shape_type> C>
    constexpr std::pair<C, tensor_reference_t<V>> operator[](const C& coord) const
    {
      return {coord, ubu::element(tensor_, coord)};
    }

    template<congruent<shape_type> C>
    constexpr bool element_exists(const C& coord) const
    {
      return ubu::element_exists(tensor_, coord);
    }

    // XXX consider customizing slice

  private:
    V tensor_;
};


template<tensor_like T>
constexpr enumerated_view<all_t<T&&>> enumerate(T&& tensor)
{
  return {all(std::forward<T>(tensor))};
}


} // end ubu

#include "../../detail/epilogue.hpp"

