#pragma once

#include "../../detail/prologue.hpp"
#include "../concepts/sized_tensor_like.hpp"
#include "../concepts/tensor_like.hpp"
#include "../element_exists.hpp"
#include "../iterator.hpp"
#include "../shape/shape.hpp"
#include <ranges>
#include <type_traits>

namespace ubu
{

// XXX ideally, we want a general masked_view<tensor_like T, mask M>
//     where a mask is a tensor_like of booleans
template<tensor_like T>
  requires std::is_trivially_copy_constructible_v<T>
class uniform_masked_view
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

    template<class T_ = T>
      requires sized_tensor_like<T>
    constexpr auto size() const
    {
      return std::ranges::size(tensor_);
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

    // begin is a template because tensor_iterator requires its template
    // parameter to be a complete type
    template<class Self = uniform_masked_view>
    constexpr tensor_iterator<Self> begin() const
    {
      return {*this};
    }
    
    constexpr tensor_sentinel end() const
    {
      return {};
    }

    // XXX a customization of slice would return mask(slice(tensor_), mask_)

  private:
    T tensor_;
    bool mask_;
};

} // end ubu

#include "../../detail/epilogue.hpp"
