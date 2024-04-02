#pragma once

#include "../detail/prologue.hpp"

#include "../memory/pointer/pointer_like.hpp"
#include "../miscellaneous/constant_valued.hpp"
#include "../tensor/coordinate/concepts/integral_like.hpp"
#include "../tensor/coordinate/constant.hpp"
#include <concepts>
#include <cstdint>
#include <iterator>
#include <memory>
#include <span>


namespace ubu
{


template<pointer_like P, integral_like S = std::size_t>
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

  private:
    template<integral_like T>
    static constexpr std::size_t extent_impl()
    {
      if constexpr (constant_valued<T>)
      {
        return constant_value_v<T>;
      }
      else
      {
        return std::dynamic_extent;
      }
    }

  public:
    static constexpr std::size_t extent = extent_impl<S>();

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

    constexpr reference operator[](size_type idx) const
    {
      return data()[idx];
    }

    constexpr pointer data() const noexcept
    {
      return data_;
    }

    template<int = 0>
      requires (extent == std::dynamic_extent)
    constexpr size_type size() const noexcept
    {
      return size_;
    }

    template<int = 0>
      requires (extent != std::dynamic_extent)
    constexpr static size_type size() noexcept
    {
      return extent;
    }

    constexpr size_type size_bytes() const noexcept
    {
      return size() * sizeof(element_type);
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

    template<std::size_t Count>
    constexpr fancy_span<P, constant<Count>> first() const
    {
      return {data(), constant<Count>()};
    }

    constexpr fancy_span<P> first(size_type count) const
    {
      return {data(), count};
    }

    template<std::size_t Count>
    constexpr fancy_span<P, constant<Count>> last() const
    {
      return fancy_span<P, constant<Count>>{data() + (size() - Count), Count};
    }

    constexpr fancy_span<P> last(size_type count) const
    {
      return {data() + (size() - count), count};
    }

    // the return type of this is a fancy_span
    template<std::size_t Offset,
             std::size_t Count = std::dynamic_extent>
    constexpr auto subspan() const
    {
      if constexpr (Count == std::dynamic_extent)
      {
        return fancy_span(data() + Offset, size() - constant<Offset>());
      }
      else
      {
        return fancy_span(data() + Offset, constant<Count>());
      }
    }

    constexpr fancy_span<P> subspan(size_type offset, size_type count) const
    {
      return {data() + offset, count};
    }

  private:
    P data_;
    [[no_unique_address]] size_type size_;
};


template<pointer_like P, std::integral E>
fancy_span(P, E) -> fancy_span<P>;


} // end ubu


#include "../detail/epilogue.hpp"

