#pragma once

#include "../../../detail/prologue.hpp"
#include "../../../utilities/integrals/size.hpp"
#include "../../concepts/sized_tensor_like.hpp"
#include "../../concepts/view.hpp"
#include "../../element_exists.hpp"
#include "../../shapes/shape.hpp"
#include <ranges>
#include <type_traits>

namespace ubu
{

// XXX ideally, we want a general masked_view<tensor_like T, mask M>
//     where a mask is a tensor_like of booleans
template<view T>
class uniform_masked_view : public std::ranges::view_base
{
  public:
    constexpr uniform_masked_view(T tensor, bool mask)
      : tensor_{tensor}, mask_{mask}
    {}

    uniform_masked_view(const uniform_masked_view&) = default;

    constexpr auto shape() const
    {
      return ubu::shape(tensor_);
    }

    template<sized_tensor_like T_ = T>
    constexpr auto size() const
    {
      return ubu::size(tensor_);
    }

    // precondition: element_exists(coord)
    template<coordinate_for<T> C>
    constexpr decltype(auto) operator[](const C& coord) const
    {
      return ubu::element(tensor_, coord);
    }

    // precondition: in_domain(tensor(), coord)
    template<coordinate_for<T> C>
    constexpr bool element_exists(const C& coord) const
    {
      return mask_ and ubu::element_exists(tensor_, coord);
    }

    constexpr T tensor() const
    {
      return tensor_;
    }

    // XXX a customization of slice would return mask(slice(tensor_), mask_)

  private:
    T tensor_;
    bool mask_;
};

} // end ubu

#include "../../../detail/epilogue.hpp"

