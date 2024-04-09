#pragma once

#include "../../detail/prologue.hpp"

#include "../../miscellaneous/bounded.hpp"
#include "../coordinate/constant.hpp"
#include "../element_exists.hpp"
#include "../fancy_span.hpp"
#include "vector_like.hpp"
#include <concepts>
#include <iterator>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

namespace ubu
{

template<class T, std::size_t N>
class inplace_vector
{
  public:
    using size_type = bounded<N>;
    
    // construct/copy/destroy
    inplace_vector() = default;

    template<std::input_iterator I, std::sentinel_for<I> S>
    constexpr inplace_vector(I begin, S end)
    {
      for(; begin != end; ++begin)
      {
        push_back(*begin);
      }
    }

    inplace_vector(const inplace_vector&) = default;

    constexpr ~inplace_vector()
    {
      clear();
    }

    // iterators
    constexpr T* begin() noexcept
    {
      return data();
    }

    constexpr const T* begin() const noexcept
    {
      return data();
    }

    constexpr T* end()
    {
      return begin() + size();
    }

    constexpr const T* end() const
    {
      return begin() + size();
    }

    // size/capacity
    [[nodiscard]] constexpr bool empty() const noexcept
    {
      return size() == 0;
    }

    constexpr size_type size() const noexcept
    {
      return size_;
    }

    static constexpr constant<N> max_size() noexcept
    {
      return {};
    }

    static constexpr constant<N> capacity() noexcept
    {
      return {};
    }

    // XXX TODO
    // constexpr void resize(size_type sz);
    // constexpr void resize(size_type sz, const T& c);
    // static constexpr void reserve(size_type n);
    // static constexpr void shrink_to_fit();

    // element access
    constexpr T& operator[](size_type n)
    {
      return data()[n];
    }

    constexpr const T& operator[](size_type n) const
    {
      return data()[n];
    }

    constexpr T& front()
    {
      return operator[](0);
    }

    constexpr const T& front() const
    {
      return operator[](0);
    }

    constexpr T& back()
    {
      return operator[](size() - 1);
    }

    constexpr const T& back() const
    {
      return operator[](size() - 1);
    }

    constexpr std::optional<T> maybe_back() const
    {
      return empty() ? std::nullopt : std::make_optional(back());
    }

    // data access
    constexpr T* data()
    {
      return std::launder(reinterpret_cast<T*>(data_));
    }

    constexpr const T* data() const
    {
      return std::launder(reinterpret_cast<const T*>(data_));
    }

    // modifiers
    template<class... Args>
      requires std::constructible_from<T,Args&&...>
    constexpr T& emplace_back(Args&&... args)
    {
      std::construct_at(data() + size_, std::forward<Args>(args)...);
      ++size_;
      return back();
    }

    constexpr T& push_back(const T& value)
    {
      return emplace_back(value);
    }

    constexpr T& push_back(T&& value)
    {
      return emplace_back(std::move(value));
    }

    constexpr void pop_back()
    {
      --size_;
      std::destroy_at(data() + size_);
    }

    constexpr void clear()
    {
      //while(size() > 0)
      //{
      //  pop_back();
      //}

      // XXX perhaps this is more likely to be unrolled?
      for(; size_ > 0; --size_)
      {
        std::destroy_at(&back());
      }
    }

    // tensor-like extensions
    static constexpr size_type shape() noexcept
    {
      return capacity();
    }

    constexpr bool element_exists(size_type i) const noexcept
    {
      return i < size();
    }

    template<vector_like V>
    constexpr void store(V dst) const
    {
      for(size_type i = 0; i < N; ++i)
      {
        if(i < N and ubu::element_exists(dst, i))
        {
          dst[i] = (*this)[i];
        }
      }
    }

    constexpr auto all()
    {
      return fancy_span(data(),size());
    }

    constexpr auto all() const
    {
      return fancy_span(data(),size());
    }

  private:
    std::aligned_storage_t<sizeof(T), alignof(T)> data_[N];
    size_type size_ = 0;
};

} // end ubu

#include "../../detail/epilogue.hpp"

