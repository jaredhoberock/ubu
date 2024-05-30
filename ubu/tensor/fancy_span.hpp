#pragma once

#include "../detail/prologue.hpp"

#include "../memory/pointer/pointer_like.hpp"
#include "../miscellaneous/constant.hpp"
#include "../miscellaneous/constant_valued.hpp"
#include "../miscellaneous/integral/integral_like.hpp"
#include <concepts>
#include <cstdint>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
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

    // note that the type of extent may differ from S
    // for example, when S is bounded<b>, extent is constant<b>
    static constexpr integral_like auto extent = std::numeric_limits<S>::max();

    template<pointer_like OtherP>
      requires std::convertible_to<OtherP, P>
    explicit(constant_valued<size_type>)
    constexpr fancy_span(OtherP first, size_type count)
      : data_{first},
        size_{count}
    {}

    template<int = 0>
      requires (extent == 0 or not constant_valued<size_type>)
    constexpr fancy_span() noexcept
      : fancy_span{P{nullptr},0}
    {}

    template<pointer_like OtherP>
      requires (std::convertible_to<OtherP, P> and !std::convertible_to<OtherP, size_type>)
    explicit(constant_valued<size_type>)
    constexpr fancy_span(OtherP first, OtherP last)
      : fancy_span(first, last - first)
    {}

    // XXX i think using std::ranges::size demotes fancy integers
    //     we should use span_like here
    template<class R>
      requires (std::ranges::contiguous_range<R&&> and std::ranges::sized_range<R&&>
                and std::convertible_to<decltype(std::ranges::data(std::declval<R&&>())), P>
                and std::convertible_to<std::ranges::range_size_t<R&&>, S>)
    explicit(constant_valued<size_type>)
    constexpr fancy_span(R&& range)
      : fancy_span(std::ranges::data(std::forward<R&&>(range)),
                   std::ranges::size(std::forward<R&&>(range)))
    {}

    template<pointer_like OtherP, integral_like OtherS>
      requires ((extent == std::dynamic_extent or fancy_span<OtherP,OtherS>::extent == std::dynamic_extent or fancy_span<OtherP,OtherS>::extent == extent)
               and std::convertible_to<OtherP,P>)
    explicit(extent != std::dynamic_extent and fancy_span<OtherP,OtherS>::extent == std::dynamic_extent)
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

    template<int = 0>
      requires (not constant_valued<S>)
    constexpr size_type size() const noexcept
    {
      return size_;
    }

    template<int = 0>
      requires constant_valued<S>
    constexpr static size_type size() noexcept
    {
      return S{};
    }

    template<int = 0>
      requires (not constant_valued<S>)
    constexpr size_type size_bytes() const noexcept
    {
      return size() * sizeof(element_type);
    }

    template<int = 0>
      requires constant_valued<S>
    constexpr static auto size_bytes() noexcept
    {
      return size() * constant<sizeof(element_type)>();
    }

    // empty is not static when S is not constant valued
    template<int = 0>
      requires (not constant_valued<S>)
    [[nodiscard]] constexpr bool empty() const noexcept
    {
      return size() == 0;
    }

    // empty is static when S is constant valued
    template<int = 0>
      requires constant_valued<S>
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

    // enable this function for static cases
    template<int = 0>
      requires (extent != std::dynamic_extent)
    static constexpr integral_like auto shape() noexcept
    {
      return extent;
    }

    // enable this function for static cases
    template<integral_like I>
      requires (extent != std::dynamic_extent)
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

// XXX i think using std::ranges::range_size_t demotes fancy integers
//     we should use span_like here
template<class R>
  requires (std::ranges::contiguous_range<R&&> and std::ranges::sized_range<R&&>)
fancy_span(R&&) -> fancy_span<decltype(std::ranges::data(std::declval<R&&>())), std::ranges::range_size_t<R&&>>;

} // end ubu


// std::ranges interop
template<ubu::pointer_like P, ubu::integral_like S>
inline constexpr bool std::ranges::enable_borrowed_range<ubu::fancy_span<P,S>> = true;

template<ubu::pointer_like P, ubu::integral_like S>
inline constexpr bool std::ranges::enable_view<ubu::fancy_span<P,S>> = true;


#include "../detail/epilogue.hpp"

