#pragma once

#include "../detail/prologue.hpp"

#include <concepts>
#include <type_traits>

ASPERA_NAMESPACE_OPEN_BRACE


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
class pointer_adaptor;


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
class pointer_adaptor_reference : private C
{
  private:
    // derive from copier for EBCO
    using super_t = C;
    using handle_type = typename C::handle_type;

    static_assert(std::same_as<T, typename C::element_type>);
    using element_type = T;

    using value_type = std::remove_cv_t<element_type>;

  public:
    pointer_adaptor_reference() = default;

    pointer_adaptor_reference(const pointer_adaptor_reference&) = default;

    constexpr pointer_adaptor_reference(const handle_type& handle, const C& copier)
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

    // address-of operator returns pointer_adaptor
    constexpr pointer_adaptor<T,C> operator&() const
    {
      return {handle_, copier()};
    }

    pointer_adaptor_reference operator=(const pointer_adaptor_reference& ref) const
    {
      copier().copy_n(ref.handle_, 1, handle_);
      return *this;
    }

    template<class = T>
      requires std::is_assignable_v<element_type&, value_type>
    pointer_adaptor_reference operator=(const value_type& value) const
    {
      copier().copy_n_from_raw_pointer(&value, 1, handle_);
      return *this;
    }

    // equality
    friend bool operator==(const pointer_adaptor_reference& self, const value_type& value)
    {
      return self.operator value_type () == value;
    }

    friend bool operator==(const value_type& value, const pointer_adaptor_reference& self)
    {
      return self.operator value_type () == value;
    }

    friend bool operator==(const pointer_adaptor_reference& lhs, const pointer_adaptor_reference& rhs)
    {
      return lhs.operator value_type () == rhs.operator value_type ();
    }

    // inequality
    friend bool operator!=(const pointer_adaptor_reference& self, const value_type& value)
    {
      return !(self == value);
    }

    friend bool operator!=(const value_type& value, const pointer_adaptor_reference& self)
    {
      return !(self == value);
    }

    friend bool operator!=(const pointer_adaptor_reference& lhs, const pointer_adaptor_reference& rhs)
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
class pointer_adaptor : private C
{
  private:
    // derive from copier for EBCO
    using super_t = C;

  public:
    static_assert(std::same_as<T, typename C::element_type>);
    using element_type = T;

    using handle_type = typename C::handle_type;
    using difference_type = std::ptrdiff_t; // XXX or ask the Accessor

    // iterator traits
    using value_type = std::remove_cv_t<element_type>;
    using iterator_category = std::random_access_iterator_tag;
    using pointer = pointer_adaptor;
    using reference = detail::pointer_adaptor_reference<T,C>;

    pointer_adaptor() = default;

    constexpr pointer_adaptor(std::nullptr_t) noexcept
      : pointer_adaptor{null_handle(C{})}
    {}

    pointer_adaptor(const pointer_adaptor&) = default;
    pointer_adaptor& operator=(const pointer_adaptor&) = default;

    constexpr pointer_adaptor(const handle_type& h) noexcept
      : pointer_adaptor{h, C{}}
    {}

    constexpr pointer_adaptor(const handle_type& h, const C& c) noexcept
      : super_t{c}, handle_{h}
    {}

    constexpr pointer_adaptor(const handle_type& h, C&& c) noexcept
      : super_t{std::move(c)}, handle_{h}
    {}

    template<class... Args>
      requires std::constructible_from<C,Args&&...>
    constexpr pointer_adaptor(const handle_type& h, Args&&... copier_args)
      : pointer_adaptor{h, C{std::forward<Args&&>(copier_args)...}}
    {}

    template<class U, copier OtherC>
      requires (std::convertible_to<U*,T*> and
                std::convertible_to<typename pointer_adaptor<U,OtherC>::handle_type, handle_type> and
                std::convertible_to<OtherC, C>)
    constexpr pointer_adaptor(const pointer_adaptor<U,OtherC>& other)
      : pointer_adaptor{other.native_handle(), other.copier()}
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

    // conversion to bool
    explicit operator bool() const noexcept
    {
      return get() != null_handle(copier());
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
    pointer_adaptor& operator++()
    {
      this->advance(copier(), handle_, 1);
      return *this;
    }

    // pre-decrement
    pointer_adaptor& operator--()
    {
      this->advance(copier(), handle_, -1);
      return *this;
    }

    // post-increment
    pointer_adaptor operator++(int)
    {
      pointer_adaptor result = *this;
      operator++();
      return result;
    }

    // post-decrement
    pointer_adaptor operator--(int)
    {
      pointer_adaptor result = *this;
      operator--();
      return result;
    }

    // plus
    pointer_adaptor operator+(difference_type n) const
    {
      pointer_adaptor result = *this;
      result += n;
      return result;
    }

    // minus
    pointer_adaptor operator-(difference_type n) const
    {
      pointer_adaptor result = *this;
      result -= n;
      return result;
    }

    // plus-equal
    pointer_adaptor& operator+=(difference_type n)
    {
      this->advance(copier(), handle_, n);
      return *this;
    }

    // minus-equal
    pointer_adaptor& operator-=(difference_type n)
    {
      this->advance(copier(), handle_, -n);
      return *this;
    }

    // difference
    difference_type operator-(const pointer_adaptor& other) const noexcept
    {
      return get() - other.get();
    }

    // equality
    bool operator==(const pointer_adaptor& other) const noexcept
    {
      return native_handle() == other.native_handle();
    }

    friend bool operator==(const pointer_adaptor& self, std::nullptr_t) noexcept
    {
      return self.native_handle() == null_handle(self.copier());
    }

    friend bool operator==(std::nullptr_t, const pointer_adaptor& self) noexcept
    {
      return null_handle(self.copier()) == self.native_handle();
    }

    // inequality
    bool operator!=(const pointer_adaptor& other) const noexcept
    {
      return !operator==(other);
    }

    friend bool operator!=(const pointer_adaptor& self, std::nullptr_t) noexcept
    {
      return !(self == nullptr);
    }

    friend bool operator!=(std::nullptr_t, const pointer_adaptor& self) noexcept
    {
      return !(nullptr == self);
    }

    // less
    bool operator<(const pointer_adaptor& other) const noexcept
    {
      return native_handle() < other.native_handle();
    }

    // lequal
    bool operator<=(const pointer_adaptor& other) const noexcept
    {
      return native_handle() <= other.native_handle();
    }

  private:
    handle_type handle_;

    static handle_type null_handle(const C& c)
    {
      if constexpr(detail::has_null_handle<C>)
      {
        return c.null_handle();
      }

      return nullptr;
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


ASPERA_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

