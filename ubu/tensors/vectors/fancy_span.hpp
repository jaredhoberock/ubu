#pragma once

#include "../../detail/prologue.hpp"

#include "../../places/memory/pointers/pointer_like.hpp"
#include "../../miscellaneous/constant.hpp"
#include "../../miscellaneous/constant_valued.hpp"
#include "../../miscellaneous/dynamic_valued.hpp"
#include "../../miscellaneous/integrals/bounded_integral_like.hpp"
#include "../../miscellaneous/integrals/integral_like.hpp"
#include "../../miscellaneous/integrals/to_integral.hpp"
#include "../../miscellaneous/integrals/size.hpp"
#include "../traits/tensor_size.hpp"
#include "contiguous_vector_like.hpp"
#include <concepts>
#include <cstdint>
#include <iterator>
#include <memory>
#include <ranges>
#include <type_traits>


namespace ubu
{


// S defaults to std::size_t unless P is unusually fancy
template<pointer_like P, integral_like S = std::make_unsigned_t<typename std::pointer_traits<P>::difference_type>>
class fancy_span
{
  public:
    using element_type = typename std::pointer_traits<P>::element_type;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = S;
    using difference_type = std::ptrdiff_t;
    using pointer = P;
    using const_pointer = typename std::pointer_traits<pointer>::template rebind<const element_type>;
    using reference = std::iter_reference_t<pointer>;
    using const_reference = std::iter_reference_t<const_pointer>;
    using iterator = pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;

    static constexpr S extent = std::numeric_limits<S>::max();

    template<pointer_like OtherP>
      requires std::convertible_to<OtherP, P>
    explicit(constant_valued<size_type>)
    constexpr fancy_span(OtherP first, size_type count)
      : data_{first},
        size_{count}
    {}

    template<int = 0>
      requires (extent == 0 or dynamic_valued<size_type>)
    constexpr fancy_span() noexcept
      : fancy_span{P{nullptr},0}
    {}

    template<pointer_like OtherP>
      requires (std::convertible_to<OtherP, P> and !std::convertible_to<OtherP, size_type>)
    explicit(constant_valued<size_type>)
    constexpr fancy_span(OtherP first, OtherP last)
      : fancy_span(first, last - first)
    {}

    template<contiguous_vector_like V>
      requires (    std::convertible_to<decltype(std::data(std::declval<V&&>())), P>
                and std::convertible_to<tensor_size_t<V&&>, S>)
    explicit(constant_valued<size_type>)
    constexpr fancy_span(V&& vec)
      : fancy_span(std::data(std::forward<V&&>(vec)),
                   ubu::size(std::forward<V&&>(vec)))
    {}

    template<pointer_like OtherP, integral_like OtherS>
      requires ((dynamic_valued<size_type> or dynamic_valued<OtherS> or bool(fancy_span<OtherP,OtherS>::extent == extent))
                and std::convertible_to<OtherP,P>)
    explicit(constant_valued<size_type> and dynamic_valued<OtherS>)
    constexpr fancy_span(const fancy_span<OtherP,OtherS>& other) noexcept
      : fancy_span(other.data(), other.size())
    {}

    constexpr fancy_span(const fancy_span&) noexcept = default;

    constexpr fancy_span& operator=(const fancy_span& other) noexcept = default;

    constexpr iterator begin() const noexcept
    {
      return data();
    }

    constexpr iterator end() const noexcept
    {
      return begin() + size();
    }

    constexpr reverse_iterator rbegin() const noexcept
    {
      return std::make_reverse_iterator(end());
    }

    constexpr reverse_iterator rend() const noexcept
    {
      return std::make_reverse_iterator(begin());
    }

    constexpr reference front() const
    {
      return *begin();
    }

    constexpr reference back() const
    {
      return *(end() - 1);
    }

    template<integral_like I>
    constexpr reference operator[](I idx) const
    {
      return data()[idx];
    }

    constexpr pointer data() const noexcept
    {
      return data_;
    }

    // size is not static when size_type is dynamic valued
    template<int = 0>
      requires dynamic_valued<size_type>
    constexpr size_type size() const noexcept
    {
      return size_;
    }

    // size is static when size_type is constant valued
    template<int = 0>
      requires constant_valued<size_type>
    constexpr static size_type size() noexcept
    {
      return S{};
    }

    // size_bytes is not static when size_type is dynamic valued
    template<int = 0>
      requires dynamic_valued<size_type>
    constexpr size_type size_bytes() const noexcept
    {
      return size() * sizeof(element_type);
    }

    // size_bytes is static when size_type is constant valued
    template<int = 0>
      requires constant_valued<size_type>
    constexpr static auto size_bytes() noexcept
    {
      return size() * constant<sizeof(element_type)>();
    }

    // empty is not static when size_type is dynamic valued
    template<int = 0>
      requires dynamic_valued<size_type>
    [[nodiscard]] constexpr bool empty() const noexcept
    {
      return size() == 0;
    }

    // empty is static when size_type is constant valued
    template<int = 0>
      requires constant_valued<size_type>
    [[nodiscard]] constexpr static bool empty() noexcept
    {
      return size() == 0;
    }

    constexpr auto first(integral_like auto count) const
    {
      return subspan(0_c, count);
    }

    constexpr auto last(integral_like auto count) const
    {
      return subspan(size() - count, count);
    }

    constexpr auto subspan(integral_like auto offset, integral_like auto count) const
    {
      if constexpr(std::integral<S> and std::integral<decltype(count)>)
      {
        // if neither S nor the count are fancy,
        // just return the same type of span as this span
        return fancy_span(data() + offset, count);
      }
      else
      {
        return fancy_span<P,decltype(count)>(data() + offset, count);
      }
    }

    constexpr auto subspan(integral_like auto offset) const
    {
      return subspan(offset, size() - offset);
    }

    // tensor-like extensions

    // if size is dynamic but it has a constant bound,
    // enable shape() and return a constant
    template<int = 0>
      requires (dynamic_valued<size_type> and bounded_integral_like<size_type>)
    static constexpr constant<to_integral(extent)> shape() noexcept
    {
      return {};
    }

    // enable this function when shape() is enabled
    template<integral_like I>
      requires (dynamic_valued<size_type> and bounded_integral_like<size_type>)
    constexpr bool element_exists(I idx) const noexcept
    {
      return idx < size();
    }

  private:
    P data_;
    [[no_unique_address]] size_type size_;
};

template<pointer_like P, integral_like S>
fancy_span(P, S) -> fancy_span<P,S>;

template<contiguous_vector_like V>
fancy_span(V&&) -> fancy_span<decltype(std::data(std::declval<V&&>())), tensor_size_t<V&&>>;

} // end ubu


// std::ranges interop
template<ubu::pointer_like P, ubu::integral_like S>
inline constexpr bool std::ranges::enable_borrowed_range<ubu::fancy_span<P,S>> = true;

template<ubu::pointer_like P, ubu::integral_like S>
inline constexpr bool std::ranges::enable_view<ubu::fancy_span<P,S>> = true;


#include "../../detail/epilogue.hpp"

