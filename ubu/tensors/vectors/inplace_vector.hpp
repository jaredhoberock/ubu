#pragma once

#include "../../detail/prologue.hpp"

#include "../../detail/reflection/is_host.hpp"
#include "../../utilities/constant.hpp"
#include "../../utilities/integrals/bounded.hpp"
#include "../element_exists.hpp"
#include "../shapes/shape.hpp"
#include "../traits/tensor_element.hpp"
#include "fancy_span.hpp"
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

struct from_vector_like_t {};

constexpr inline from_vector_like_t from_vector_like;

template<class T, std::size_t N>
class inplace_vector
{
  public:
    using size_type = bounded<N>;
    
    // construct/copy/destroy
    inplace_vector() = default;

    constexpr inplace_vector(size_type n, const T& value)
      : size_(n)
    {
      #pragma unroll
      for(int i = 0; i != capacity(); ++i)
      {
        if(i < size())
        {
          std::construct_at(data() + i, value);
        }
      }
    }

    constexpr explicit inplace_vector(size_type n)
      : inplace_vector(n, T{})
    {}

    template<std::input_iterator I, std::sentinel_for<I> S>
    constexpr inplace_vector(I begin, S end)
    {
      // XXX this is not gonna be unrolled
      for(; begin != end; ++begin)
      {
        push_back(*begin);
      }
    }

    template<vector_like V>
    constexpr inplace_vector(from_vector_like_t, V&& vec)
    {
      #pragma unroll
      for(int i = 0; i < ubu::shape(vec); ++i)
      {
        if(i < capacity() and ubu::element_exists(vec, i))
        {
          // XXX we may need to move vec[i] vec was moved into this ctor
          std::construct_at(data() + i, vec[i]);
          size_ = size_ + 1;
        }
      }
    }

    inplace_vector(const inplace_vector&) = default;

    constexpr inplace_vector(const inplace_vector& other) requires(std::is_copy_constructible_v<T> and not std::is_trivially_copy_constructible_v<T>)
      : inplace_vector(from_vector_like, other)
    {}

    inplace_vector(inplace_vector&&) = default;

    constexpr inplace_vector(inplace_vector&& other) requires(std::is_move_constructible_v<T> and not std::is_trivially_move_constructible_v<T>)
    {
      // XXX ideally, we would just forward to the from_vector_like ctor
      //     but we don't currently have a good way to create a moving view
      //     without as_rvalue
      #pragma unroll
      for(int i = 0; i < ubu::shape(other); ++i)
      {
        if(i < capacity() and ubu::element_exists(other, i))
        {
          std::construct_at(data() + i, std::move(other[i]));
        }
      }

      size_ = other.size_;
    }

    ~inplace_vector() = default;

    constexpr ~inplace_vector() requires(not std::is_trivially_destructible_v<T>)
    {
      clear();
    }

    inplace_vector& operator=(const inplace_vector&) = default;
    inplace_vector& operator=(inplace_vector&&) = default;

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
    constexpr T& operator[](int n)
    {
      return data()[n];
    }

    constexpr const T& operator[](int n) const
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
      if UBU_TARGET(detail::is_host())
      {
        return empty() ? std::nullopt : std::make_optional(back());
      }
      else
      {
        // the loop below seems to avoid generating
        // spills to local memory in device code

        std::optional<T> result;

        #pragma unroll
        for(int i = 0; i < capacity(); ++i)
        {
          if(i == size() - 1 and not result.has_value())
          {
            result = operator[](i);
          }
        }

        return result;
      }
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
      if constexpr (not std::is_trivially_destructible_v<T>)
      {
        #pragma unroll
        for(int i = 0; i < capacity(); ++i)
        {
          if(i < size())
          {
            std::destroy_at(data() + i);
          }
        }
      }

      size_ = 0;
    }

    // tensor-like extensions
    static constexpr size_type shape() noexcept
    {
      return capacity();
    }

    constexpr bool element_exists(int i) const noexcept
    {
      return i < size();
    }

    template<vector_like V>
      requires std::convertible_to<T,ubu::tensor_element_t<V>>
    constexpr void store(V dst) const
    {
      #pragma unroll
      for(int i = 0; i < capacity(); ++i)
      {
        if(i < size() and ubu::element_exists(dst, i))
        {
          dst[i] = (*this)[i];
        }
      }
    }

  private:
    std::aligned_storage_t<sizeof(T), alignof(T)> data_[N];
    size_type size_ = 0;
};

template<vector_like V>
  requires constant_shaped<V&&>
inplace_vector(from_vector_like_t, V&&) -> inplace_vector<tensor_element_t<V&&>, shape_v<V&&>>;


// this is just sugar for inplace_vector's ctor
template<vector_like V>
  requires constant_shaped<V&&>
constexpr inplace_vector<tensor_element_t<V&&>, shape_v<V&&>> load(V&& vec)
{
  return inplace_vector(from_vector_like, std::forward<V>(vec));
}

} // end ubu

#include "../../detail/epilogue.hpp"

