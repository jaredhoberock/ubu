#pragma once

#include "../detail/prologue.hpp"

#include "../memory/pointer/pointer_like.hpp"
#include <concepts>
#include <cstdint>
#include <iterator>
#include <memory>
#include <span>


namespace ubu
{


template<pointer_like P, std::size_t Extent = std::dynamic_extent>
class fancy_span
{
  public:
    using element_type = typename std::pointer_traits<P>::element_type;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = P;
    using const_pointer = typename std::pointer_traits<pointer>::template rebind<const element_type>;
    using reference = std::iter_reference_t<pointer>;
    using const_reference = std::iter_reference_t<const_pointer>;
    using iterator = pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;

    static constexpr std::size_t extent = Extent;

    template<pointer_like OtherP>
      requires std::convertible_to<OtherP, P>
    explicit(extent != std::dynamic_extent)
    constexpr fancy_span(OtherP first, size_type count)
      : data_{first},
        size_{count}
    {}

    template<int = 0>
      requires (extent == 0 or extent == std::dynamic_extent)
    constexpr fancy_span() noexcept
      : fancy_span{P{nullptr},0}
    {}

    template<pointer_like OtherP>
      requires (std::convertible_to<OtherP, P> and !std::convertible_to<OtherP, size_type>)
    explicit(extent != std::dynamic_extent)
    constexpr fancy_span(OtherP first, OtherP last)
      : fancy_span(first, last - first)
    {}

    template<pointer_like OtherP, std::size_t N>
      requires ((extent == std::dynamic_extent or N == std::dynamic_extent or N == extent)
               and std::convertible_to<OtherP,P>)
    explicit(extent != std::dynamic_extent and N == std::dynamic_extent)
    constexpr fancy_span(const fancy_span<OtherP,N>& other) noexcept
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

    constexpr reference operator[](size_type idx) const
    {
      return data()[idx];
    }

    constexpr pointer data() const noexcept
    {
      return data_;
    }

    template<int = 0>
      requires (Extent == std::dynamic_extent)
    constexpr size_type size() const noexcept
    {
      return size_;
    }

    template<int = 0>
      requires (Extent != std::dynamic_extent)
    constexpr size_type size() const noexcept
    {
      return extent;
    }

    constexpr size_type size_bytes() const noexcept
    {
      return size() * sizeof(element_type);
    }

    [[nodiscard]] constexpr bool empty() const noexcept
    {
      return size() == 0;
    }

    template<std::size_t Count>
    constexpr fancy_span<P, Count> first() const
    {
      return {data(), Count};
    }

    constexpr fancy_span<P, std::dynamic_extent> first(size_type count) const
    {
      return {data(), count};
    }

    template<std::size_t Count>
    constexpr fancy_span<P, Count> last() const
    {
      return fancy_span<P,Count>{data() + (size() - Count), Count};
    }

    constexpr fancy_span<P, std::dynamic_extent> last(size_type count) const
    {
      return {data() + (size() - count), count};
    }

    template<std::size_t Offset,
             std::size_t Count = std::dynamic_extent>
    constexpr fancy_span<P,subspan_extent<Offset,Count>()> subspan() const
    {
      std::size_t count = (Count == std::dynamic_extent) ? size() - Offset : Count;

      return {data() + Offset, count};
    }

    constexpr fancy_span<P, std::dynamic_extent> subspan(size_type offset, size_type count) const
    {
      return {data() + offset, count};
    }

  private:

    template<std::size_t Offset, std::size_t Count>
    static constexpr std::size_t subspan_extent()
    {
      if(Count != std::dynamic_extent) return Count;

      if(Extent != std::dynamic_extent) return Extent - Offset;

      return std::dynamic_extent;
    }

    P data_;
    size_type size_;
};


template<pointer_like P, std::integral E>
fancy_span(P, E) -> fancy_span<P>;


} // end ubu


#include "../detail/epilogue.hpp"

