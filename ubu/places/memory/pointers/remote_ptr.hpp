#pragma once

#include "../../../detail/prologue.hpp"

#include "../loaders.hpp"
#include "pointer_like.hpp"
#include <concepts>
#include <iterator>
#include <type_traits>

namespace ubu
{


template<class T, loader L>
class remote_ptr;


namespace detail
{


template<class T, loader L>
  requires (!std::is_void_v<T>)
class remote_ref : private L
{
  private:
    // derive from loader for EBCO
    using super_t = L;
    using address_type = loader_address_t<L>;

    using element_type = T;
    using value_type = std::remove_cv_t<element_type>;

  public:
    remote_ref() = default;

    remote_ref(const remote_ref&) = default;

    constexpr remote_ref(const address_type& a, const L& loader)
      : super_t{loader}, address_{a}
    {}

    template<class = T>
      requires std::is_default_constructible_v<value_type>
    operator value_type () const
    {
      value_type result{};
      download(loader(), address_, sizeof(T), &result);
      return result;
    }

    // address-of operator returns remote_ptr
    constexpr remote_ptr<T,L> operator&() const
    {
      return {address_, loader()};
    }

    remote_ref operator=(const remote_ref& ref) const
    {
      // XXX we could optimize this with a single copy-like transaction 
      T value = ref;
      return *this = value;
    }

    template<class = T>
      requires std::is_assignable_v<element_type&, value_type>
    remote_ref operator=(const value_type& value) const
    {
      upload(loader(), &value, sizeof(T), address_);
      return *this;
    }

    // equality
    friend bool operator==(const remote_ref& self, const value_type& value)
    {
      return self.operator value_type () == value;
    }

    friend bool operator==(const value_type& value, const remote_ref& self)
    {
      return self.operator value_type () == value;
    }

    friend bool operator==(const remote_ref& lhs, const remote_ref& rhs)
    {
      return lhs.operator value_type () == rhs.operator value_type ();
    }

    // inequality
    friend bool operator!=(const remote_ref& self, const value_type& value)
    {
      return !(self == value);
    }

    friend bool operator!=(const value_type& value, const remote_ref& self)
    {
      return !(self == value);
    }

    friend bool operator!=(const remote_ref& lhs, const remote_ref& rhs)
    {
      return !(lhs == rhs);
    }

  private:
    constexpr const L& loader() const
    {
      return *this;
    }

    constexpr L& loader()
    {
      return *this;
    }

    address_type address_;
};


} // end detail


template<class T, loader L>
class remote_ptr : private L
{
  private:
    // derive from loader for EBCO
    using super_t = L;

  public:
    using element_type = T;
    using address_type = loader_address_t<L>;

    // iterator traits
    using difference_type = address_difference_result_t<address_type>;
    using value_type = std::remove_cv_t<element_type>;
    using pointer = remote_ptr;
    using reference = detail::remote_ref<T,L>;
    using iterator_category = std::random_access_iterator_tag;
    using iterator_concept = std::random_access_iterator_tag;

    remote_ptr() = default;

    constexpr remote_ptr(std::nullptr_t) noexcept
      : remote_ptr{make_null_address<address_type>()}
    {}

    remote_ptr(const remote_ptr&) = default;
    remote_ptr& operator=(const remote_ptr&) = default;

    constexpr remote_ptr(const address_type& a) noexcept
      : remote_ptr{a, L{}}
    {}

    constexpr remote_ptr(const address_type& a, const L& l) noexcept
      : super_t{l}, address_{a}
    {}

    constexpr remote_ptr(const address_type& a, L&& l) noexcept
      : super_t{std::move(l)}, address_{a}
    {}

    template<class... Args>
      requires std::constructible_from<L,Args&&...>
    constexpr remote_ptr(const address_type& a, Args&&... loader_args)
      : remote_ptr{a, L{std::forward<Args&&>(loader_args)...}}
    {}

    template<class U, loader OtherL>
      requires (std::convertible_to<U*,T*> and
                std::convertible_to<typename remote_ptr<U,OtherL>::address_type, address_type> and
                std::convertible_to<OtherL, L>)
    constexpr remote_ptr(const remote_ptr<U,OtherL>& other)
      : remote_ptr{other.to_address(), other.loader()}
    {}

    // returns the underlying address
    constexpr const address_type& to_address() const noexcept
    {
      return address_;
    }

    // returns the underlying address as a raw pointer to T
    // when the underlying address type is itself a raw pointer
    template<class A = address_type>
      requires pointer_like<A>
    constexpr T* to_raw_pointer() const noexcept
    {
      return reinterpret_cast<T*>(to_address());
    }

    // reinterprets this remote_ptr
    template<class U>
    constexpr remote_ptr<U,L> reinterpret_pointer() noexcept
    {
      return {to_address(), loader()};
    }

    // returns the loader
    const L& loader() const noexcept
    {
      return *this;
    }

    // customize construct_at
    template<class... Args>
      requires (std::is_trivially_constructible_v<T,Args...> and
                std::is_trivial_v<T>)
    void construct_at(Args&&... args) const
    {
      T copy(std::forward<Args>(args)...);
      **this = copy;
    }

    // customize destroy_at
    // this is customized for parity with construct_at
    template<class = void>
      requires std::is_trivially_destructible_v<T>
    void destroy_at() const
    {
      // no-op
    }

    // conversion to bool
    explicit operator bool() const noexcept
    {
      return to_address() != make_null_address<address_type>();
    }

    // dereference
    template<class = void>
      requires (!std::is_void_v<T>)
    reference operator*() const
    {
      return {to_address(), loader()};
    }

    // subscript
    template<class = void>
      requires (!std::is_void_v<T>)
    reference operator[](difference_type i) const
    {
      return *(*this + i);
    }

    // pre-increment
    template<class = void>
      requires (!std::is_void_v<T>)
    remote_ptr& operator++()
    {
      advance_address(address_, sizeof(T));
      return *this;
    }

    // pre-decrement
    template<class = void>
      requires (!std::is_void_v<T>)
    remote_ptr& operator--()
    {
      advance_address(address_, -sizeof(T));
      return *this;
    }

    // post-increment
    template<class = void>
      requires (!std::is_void_v<T>)
    remote_ptr operator++(int)
    {
      remote_ptr result = *this;
      operator++();
      return result;
    }

    // post-decrement
    template<class = void>
      requires (!std::is_void_v<T>)
    remote_ptr operator--(int)
    {
      remote_ptr result = *this;
      operator--();
      return result;
    }

    // plus
    template<class = void>
      requires (!std::is_void_v<T>)
    remote_ptr operator+(difference_type n) const
    {
      remote_ptr result = *this;
      result += n;
      return result;
    }

    template<class = void>
      requires (!std::is_void_v<T>)
    friend remote_ptr operator+(difference_type n, const remote_ptr& rhs)
    {
      return rhs + n;
    }

    // minus
    template<class = void>
      requires (!std::is_void_v<T>)
    remote_ptr operator-(difference_type n) const
    {
      remote_ptr result = *this;
      result -= n;
      return result;
    }

    // plus-equal
    template<class = void>
      requires (!std::is_void_v<T>)
    remote_ptr& operator+=(difference_type n)
    {
      advance_address(address_, n * sizeof(T));
      return *this;
    }

    // minus-equal
    template<class = void>
      requires (!std::is_void_v<T>)
    remote_ptr& operator-=(difference_type n)
    {
      return operator+=(-n);
    }

    // difference
    template<class = void>
      requires (!std::is_void_v<T>)
    difference_type operator-(const remote_ptr& other) const noexcept
    {
      return address_difference(to_address(), other.to_address()) / sizeof(T);
    }

    // equality
    bool operator==(const remote_ptr& other) const noexcept
    {
      return to_address() == other.to_address();
    }

    friend bool operator==(const remote_ptr& self, std::nullptr_t) noexcept
    {
      return self.to_address() == make_null_address<address_type>();
    }

    friend bool operator==(std::nullptr_t, const remote_ptr& self) noexcept
    {
      return make_null_address<address_type>() == self.to_address();
    }

    // inequality
    bool operator!=(const remote_ptr& other) const noexcept
    {
      return !operator==(other);
    }

    friend bool operator!=(const remote_ptr& self, std::nullptr_t) noexcept
    {
      return !(self == nullptr);
    }

    friend bool operator!=(std::nullptr_t, const remote_ptr& self) noexcept
    {
      return !(nullptr == self);
    }

    // less
    bool operator<(const remote_ptr& other) const noexcept
    {
      return to_address() < other.to_address();
    }

    // lequal
    bool operator<=(const remote_ptr& other) const noexcept
    {
      return to_address() <= other.to_address();
    }

    // greater
    bool operator>(const remote_ptr& other) const noexcept
    {
      return to_address() > other.to_address();
    }

    // gequal
    bool operator>=(const remote_ptr& other) const noexcept
    {
      return to_address() >= other.to_address();
    }

    // spaceship
    bool operator<=>(const remote_ptr& other) const noexcept
    {
      return to_address() <=> other.to_address();
    }

  private:
    address_type address_;
};


} // end ubu


namespace std
{

template<class T, class L>
struct iterator_traits<ubu::remote_ptr<T,L>>
{
  using difference_type = typename ubu::remote_ptr<T,L>::difference_type;
  using value_type = typename ubu::remote_ptr<T,L>::value_type;
  using pointer = typename ubu::remote_ptr<T,L>::pointer;
  using reference = typename ubu::remote_ptr<T,L>::reference;
  using iterator_category = typename ubu::remote_ptr<T,L>::iterator_category;
  using iterator_concept = typename ubu::remote_ptr<T,L>::iterator_concept;
};

} // end std

#include "../../../detail/epilogue.hpp"

