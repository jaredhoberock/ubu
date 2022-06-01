#pragma once

#include "../detail/prologue.hpp"

#include <concepts>
#include <iterator>
#include <type_traits>

namespace ubu
{

namespace detail
{


template<class C, class H, class V>
concept has_copy_n_to_raw_pointer = requires(C copier, H from, std::size_t n, V* to)
{
  copier.copy_n_to_raw_pointer(from, n, to);
};

template<class C, class H>
concept has_copy_n = requires(C copier, H from, std::size_t n, H to)
{
  copier.copy_n(from, n, to);
};

template<class C, class V, class H>
concept has_copy_n_from_raw_pointer = requires(C copier, const V* from, std::size_t n, H to)
{
  copier.copy_n_from_raw_pointer(from, n, to);
};


} // end detail


template<class T>
concept copier = requires
{
  // a copier must declare a handle_type
  typename T::handle_type;

  // the handle type must be regular
  requires std::regular<typename T::handle_type>;

  // the handle type must be totally ordered
  requires std::totally_ordered<typename T::handle_type>;

  // XXX might also require handle_type to be default constructible (and null)
  //     otherwise, need to require copier.null_handle() function
  //

  // if handle_type is not a pointer, a copier must be able to advance a handle_type
  
  // a copier must declare an element_type
  // XXX this could be relaxed if handle_type is a pointer
  typename T::element_type;

  // a copier must be able to copy from a handle to a raw pointer
  requires detail::has_copy_n_to_raw_pointer<T, typename T::handle_type, std::remove_cv_t<typename T::element_type>>;

  // a copier must be able to copy from a raw pointer to a handle if element_type is assignable
  requires (!std::is_assignable_v<typename T::element_type, typename T::element_type> or
    detail::has_copy_n_from_raw_pointer<T, std::remove_cv_t<typename T::element_type>, typename T::handle_type>
  );

  // a copier must be able to copy from a handle to a handle if element_type is assignable
  requires (!std::is_assignable_v<typename T::element_type, typename T::element_type> or
    detail::has_copy_n<T, typename T::handle_type>
  );
};


template<class T, copier C>
class fancy_ptr;


namespace detail
{


template<class Copier>
concept has_null_handle = requires(Copier c)
{
  typename Copier::handle_type;
  {c.null_handle()} -> std::same_as<typename Copier::handle_type>;
};


template<class Copier, class Difference>
concept has_advance = requires(const Copier& c, typename Copier::handle_type& h, Difference n)
{
  c.advance(h, n);
};


template<class T, copier C>
class fancy_ref : private C
{
  private:
    // derive from copier for EBCO
    using super_t = C;
    using handle_type = typename C::handle_type;

    static_assert(std::same_as<T, typename C::element_type>);
    using element_type = T;

    using value_type = std::remove_cv_t<element_type>;

  public:
    fancy_ref() = default;

    fancy_ref(const fancy_ref&) = default;

    constexpr fancy_ref(const handle_type& handle, const C& copier)
      : super_t{copier}, handle_{handle}
    {}

    template<class = T>
      requires std::is_default_constructible_v<value_type>
    operator value_type () const
    {
      value_type result{};
      copier().copy_n_to_raw_pointer(handle_, 1, &result);
      return result;
    }

    // address-of operator returns fancy_ptr
    constexpr fancy_ptr<T,C> operator&() const
    {
      return {handle_, copier()};
    }

    fancy_ref operator=(const fancy_ref& ref) const
    {
      copier().copy_n(ref.handle_, 1, handle_);
      return *this;
    }

    template<class = T>
      requires std::is_assignable_v<element_type&, value_type>
    fancy_ref operator=(const value_type& value) const
    {
      copier().copy_n_from_raw_pointer(&value, 1, handle_);
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

    handle_type handle_;
};


} // end detail


template<class T, copier C>
class fancy_ptr : private C
{
  private:
    // derive from copier for EBCO
    using super_t = C;

  public:
    static_assert(std::same_as<T, typename C::element_type>);
    using element_type = T;

    using handle_type = typename C::handle_type;

    // iterator traits
    using difference_type = std::ptrdiff_t; // XXX or ask the copier
    using value_type = std::remove_cv_t<element_type>;
    using pointer = fancy_ptr;
    using reference = detail::fancy_ref<T,C>;
    using iterator_category = std::random_access_iterator_tag;
    using iterator_concept = std::random_access_iterator_tag;

    fancy_ptr() = default;

    constexpr fancy_ptr(std::nullptr_t) noexcept
      : fancy_ptr{null_handle(C{})}
    {}

    fancy_ptr(const fancy_ptr&) = default;
    fancy_ptr& operator=(const fancy_ptr&) = default;

    constexpr fancy_ptr(const handle_type& h) noexcept
      : fancy_ptr{h, C{}}
    {}

    constexpr fancy_ptr(const handle_type& h, const C& c) noexcept
      : super_t{c}, handle_{h}
    {}

    constexpr fancy_ptr(const handle_type& h, C&& c) noexcept
      : super_t{std::move(c)}, handle_{h}
    {}

    template<class... Args>
      requires std::constructible_from<C,Args&&...>
    constexpr fancy_ptr(const handle_type& h, Args&&... copier_args)
      : fancy_ptr{h, C{std::forward<Args&&>(copier_args)...}}
    {}

    template<class U, copier OtherC>
      requires (std::convertible_to<U*,T*> and
                std::convertible_to<typename fancy_ptr<U,OtherC>::handle_type, handle_type> and
                std::convertible_to<OtherC, C>)
    constexpr fancy_ptr(const fancy_ptr<U,OtherC>& other)
      : fancy_ptr{other.native_handle(), other.copier()}
    {}

    // returns the underlying handle
    const handle_type& native_handle() const noexcept
    {
      return handle_;
    }

    // synonym for native_handle
    const handle_type& get() const noexcept
    {
      return native_handle();
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
      return native_handle() != null_handle(copier());
    }

    // dereference
    reference operator*() const
    {
      return {native_handle(), copier()};
    }

    // subscript
    reference operator[](difference_type i) const
    {
      return *(*this + i);
    }

    // pre-increment
    fancy_ptr& operator++()
    {
      this->advance(copier(), handle_, 1);
      return *this;
    }

    // pre-decrement
    fancy_ptr& operator--()
    {
      this->advance(copier(), handle_, -1);
      return *this;
    }

    // post-increment
    fancy_ptr operator++(int)
    {
      fancy_ptr result = *this;
      operator++();
      return result;
    }

    // post-decrement
    fancy_ptr operator--(int)
    {
      fancy_ptr result = *this;
      operator--();
      return result;
    }

    // plus
    fancy_ptr operator+(difference_type n) const
    {
      fancy_ptr result = *this;
      result += n;
      return result;
    }

    friend fancy_ptr operator+(difference_type n, const fancy_ptr& rhs)
    {
      return rhs + n;
    }

    // minus
    fancy_ptr operator-(difference_type n) const
    {
      fancy_ptr result = *this;
      result -= n;
      return result;
    }

    // plus-equal
    fancy_ptr& operator+=(difference_type n)
    {
      this->advance(copier(), handle_, n);
      return *this;
    }

    // minus-equal
    fancy_ptr& operator-=(difference_type n)
    {
      this->advance(copier(), handle_, -n);
      return *this;
    }

    // difference
    difference_type operator-(const fancy_ptr& other) const noexcept
    {
      return get() - other.get();
    }

    // equality
    bool operator==(const fancy_ptr& other) const noexcept
    {
      return native_handle() == other.native_handle();
    }

    friend bool operator==(const fancy_ptr& self, std::nullptr_t) noexcept
    {
      return self.native_handle() == null_handle(self.copier());
    }

    friend bool operator==(std::nullptr_t, const fancy_ptr& self) noexcept
    {
      return null_handle(self.copier()) == self.native_handle();
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
      return native_handle() < other.native_handle();
    }

    // lequal
    bool operator<=(const fancy_ptr& other) const noexcept
    {
      return native_handle() <= other.native_handle();
    }

    // greater
    bool operator>(const fancy_ptr& other) const noexcept
    {
      return native_handle() > other.native_handle();
    }

    // gequal
    bool operator>=(const fancy_ptr& other) const noexcept
    {
      return native_handle() >= other.native_handle();
    }

    // spaceship
    bool operator<=>(const fancy_ptr& other) const noexcept
    {
      return native_handle() <=> other.native_handle();
    }

  private:
    handle_type handle_;

    static handle_type null_handle(const C& c)
    {
      if constexpr(detail::has_null_handle<C>)
      {
        return c.null_handle();
      }

      return handle_type{};
    }

    static void advance(const C& c, handle_type& h, difference_type n)
    {
      if constexpr(detail::has_advance<C,difference_type>)
      {
        c.advance(h, n);
      }
      else
      {
        h += n;
      }
    }
};


// copy_n overloads

template<class T, copier C>
  requires std::is_assignable_v<T,T>
fancy_ptr<T,C> copy_n(const std::remove_cv_t<T>* first, std::size_t count, fancy_ptr<T,C> result)
{
  return result.copier().copy_n_from_raw_pointer(first, count, result.native_handle());
}

template<class T, copier C>
  requires std::is_assignable_v<T,T>
std::remove_cv_t<T>* copy_n(fancy_ptr<T,C> first, std::size_t count, std::remove_cv_t<T>* result)
{
  return first.copier().copy_n_to_raw_pointer(first.native_handle(), count, result);
}

// XXX U needs to be either T or const T
template<class T, copier C, class U>
  requires std::is_assignable_v<T&,U>
fancy_ptr<T,C> copy_n(fancy_ptr<U,C> first, std::size_t count, fancy_ptr<T,C> result)
{
  return first.copier().copy_n(first.native_handle(), count, result.native_handle());
}


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

