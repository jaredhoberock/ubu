#pragma once

#include "../detail/prologue.hpp"

#include "copier.hpp"
#include <concepts>
#include <iterator>
#include <type_traits>

namespace ubu
{


template<class T, copier_of<T> C>
class fancy_ptr;


namespace detail
{


template<class T, copier_of<T> C>
  requires (!std::is_void_v<T>)
class fancy_ref : private C
{
  private:
    // derive from copier for EBCO
    using super_t = C;
    using address_type = copier_address_t<C, T>;

    static_assert(std::same_as<T, address_element_t<address_type>>);
    using element_type = address_element_t<address_type>;

    using value_type = std::remove_cv_t<element_type>;

  public:
    fancy_ref() = default;

    fancy_ref(const fancy_ref&) = default;

    constexpr fancy_ref(const address_type& a, const C& copier)
      : super_t{copier}, address_{a}
    {}

    template<class = T>
      requires std::is_default_constructible_v<value_type>
    operator value_type () const
    {
      value_type result{};
      copy_n(copier(), address_, 1, &result);
      return result;
    }

    // address-of operator returns fancy_ptr
    constexpr fancy_ptr<T,C> operator&() const
    {
      return {address_, copier()};
    }

    fancy_ref operator=(const fancy_ref& ref) const
    {
      copy_n(copier(), ref.address_, 1, address_);
      return *this;
    }

    template<class = T>
      requires std::is_assignable_v<element_type&, value_type>
    fancy_ref operator=(const value_type& value) const
    {
      copy_n(copier(), &value, 1, address_);
      return *this;
    }

    // equality
    friend bool operator==(const fancy_ref& self, const value_type& value)
    {
      return self.operator value_type () == value;
    }

    friend bool operator==(const value_type& value, const fancy_ref& self)
    {
      return self.operator value_type () == value;
    }

    friend bool operator==(const fancy_ref& lhs, const fancy_ref& rhs)
    {
      return lhs.operator value_type () == rhs.operator value_type ();
    }

    // inequality
    friend bool operator!=(const fancy_ref& self, const value_type& value)
    {
      return !(self == value);
    }

    friend bool operator!=(const value_type& value, const fancy_ref& self)
    {
      return !(self == value);
    }

    friend bool operator!=(const fancy_ref& lhs, const fancy_ref& rhs)
    {
      return !(lhs == rhs);
    }

  private:
    constexpr const C& copier() const
    {
      return *this;
    }

    constexpr C& copier()
    {
      return *this;
    }

    address_type address_;
};


} // end detail


template<class T, copier_of<T> C>
class fancy_ptr : private C
{
  private:
    // derive from copier for EBCO
    using super_t = C;

  public:
    using address_type = copier_address_t<C, T>;
    static_assert(std::same_as<T, address_element_t<address_type>>);

    using element_type = T;

    // iterator traits
    using difference_type = address_difference_result_t<address_type>;
    using value_type = std::remove_cv_t<element_type>;
    using pointer = fancy_ptr;
    using reference = detail::fancy_ref<T,C>;
    using iterator_category = std::random_access_iterator_tag;
    using iterator_concept = std::random_access_iterator_tag;

    fancy_ptr() = default;

    constexpr fancy_ptr(std::nullptr_t) noexcept
      : fancy_ptr{make_null_address<address_type>()}
    {}

    fancy_ptr(const fancy_ptr&) = default;
    fancy_ptr& operator=(const fancy_ptr&) = default;

    constexpr fancy_ptr(const address_type& a) noexcept
      : fancy_ptr{a, C{}}
    {}

    constexpr fancy_ptr(const address_type& a, const C& c) noexcept
      : super_t{c}, address_{a}
    {}

    constexpr fancy_ptr(const address_type& a, C&& c) noexcept
      : super_t{std::move(c)}, address_{a}
    {}

    template<class... Args>
      requires std::constructible_from<C,Args&&...>
    constexpr fancy_ptr(const address_type& a, Args&&... copier_args)
      : fancy_ptr{a, C{std::forward<Args&&>(copier_args)...}}
    {}

    template<class U, copier_of<U> OtherC>
      requires (std::convertible_to<U*,T*> and
                std::convertible_to<typename fancy_ptr<U,OtherC>::address_type, address_type> and
                std::convertible_to<OtherC, C>)
    constexpr fancy_ptr(const fancy_ptr<U,OtherC>& other)
      : fancy_ptr{other.to_address(), other.copier()}
    {}

    // returns the underlying address
    const address_type& to_address() const noexcept
    {
      return address_;
    }

    // returns the copier
    C& copier() noexcept
    {
      return *this;
    }

    // returns the copier
    const C& copier() const noexcept
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
      return {to_address(), copier()};
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
    fancy_ptr& operator++()
    {
      advance_address(address_, 1);
      return *this;
    }

    // pre-decrement
    template<class = void>
      requires (!std::is_void_v<T>)
    fancy_ptr& operator--()
    {
      advance_address(address_, -1);
      return *this;
    }

    // post-increment
    template<class = void>
      requires (!std::is_void_v<T>)
    fancy_ptr operator++(int)
    {
      fancy_ptr result = *this;
      operator++();
      return result;
    }

    // post-decrement
    template<class = void>
      requires (!std::is_void_v<T>)
    fancy_ptr operator--(int)
    {
      fancy_ptr result = *this;
      operator--();
      return result;
    }

    // plus
    template<class = void>
      requires (!std::is_void_v<T>)
    fancy_ptr operator+(difference_type n) const
    {
      fancy_ptr result = *this;
      result += n;
      return result;
    }

    template<class = void>
      requires (!std::is_void_v<T>)
    friend fancy_ptr operator+(difference_type n, const fancy_ptr& rhs)
    {
      return rhs + n;
    }

    // minus
    template<class = void>
      requires (!std::is_void_v<T>)
    fancy_ptr operator-(difference_type n) const
    {
      fancy_ptr result = *this;
      result -= n;
      return result;
    }

    // plus-equal
    template<class = void>
      requires (!std::is_void_v<T>)
    fancy_ptr& operator+=(difference_type n)
    {
      advance_address(address_, n);
      return *this;
    }

    // minus-equal
    template<class = void>
      requires (!std::is_void_v<T>)
    fancy_ptr& operator-=(difference_type n)
    {
      return operator+=(-n);
    }

    // difference
    template<class = void>
      requires (!std::is_void_v<T>)
    difference_type operator-(const fancy_ptr& other) const noexcept
    {
      return address_difference(to_address(), other.to_address());
    }

    // equality
    bool operator==(const fancy_ptr& other) const noexcept
    {
      return to_address() == other.to_address();
    }

    friend bool operator==(const fancy_ptr& self, std::nullptr_t) noexcept
    {
      return self.to_address() == make_null_address<address_type>();
    }

    friend bool operator==(std::nullptr_t, const fancy_ptr& self) noexcept
    {
      return make_null_address<address_type>() == self.to_address();
    }

    // inequality
    bool operator!=(const fancy_ptr& other) const noexcept
    {
      return !operator==(other);
    }

    friend bool operator!=(const fancy_ptr& self, std::nullptr_t) noexcept
    {
      return !(self == nullptr);
    }

    friend bool operator!=(std::nullptr_t, const fancy_ptr& self) noexcept
    {
      return !(nullptr == self);
    }

    // less
    bool operator<(const fancy_ptr& other) const noexcept
    {
      return to_address() < other.to_address();
    }

    // lequal
    bool operator<=(const fancy_ptr& other) const noexcept
    {
      return to_address() <= other.to_address();
    }

    // greater
    bool operator>(const fancy_ptr& other) const noexcept
    {
      return to_address() > other.to_address();
    }

    // gequal
    bool operator>=(const fancy_ptr& other) const noexcept
    {
      return to_address() >= other.to_address();
    }

    // spaceship
    bool operator<=>(const fancy_ptr& other) const noexcept
    {
      return to_address() <=> other.to_address();
    }

  private:
    address_type address_;
};


} // end ubu


namespace std
{

template<class T, class C>
struct iterator_traits<ubu::fancy_ptr<T,C>>
{
  using difference_type = typename ubu::fancy_ptr<T,C>::difference_type;
  using value_type = typename ubu::fancy_ptr<T,C>::value_type;
  using pointer = typename ubu::fancy_ptr<T,C>::pointer;
  using reference = typename ubu::fancy_ptr<T,C>::reference;
  using iterator_category = typename ubu::fancy_ptr<T,C>::iterator_category;
  using iterator_concept = typename ubu::fancy_ptr<T,C>::iterator_concept;
};

} // end std

#include "../detail/epilogue.hpp"

