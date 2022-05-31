#pragma once

#include "../detail/prologue.hpp"

#include <concepts>

UBU_NAMESPACE_OPEN_BRACE


template<class T, class P>
concept has_property =
  // the property must be applicable to objects of type T
  P::template is_applicable_v<T>

  and requires(T t, P prop)
  {
    // the property needs a name
    // XXX P::name must be convertible to a string constant
    P::name;

    // the property needs a member named value that must be equality comparable
    { prop.value } -> std::equality_comparable;

    // the property must be queryable from objects of type T and it should have the expected type
    { prop(t) } -> std::same_as<decltype(P::value)>;
  }

  // T shouldn't have a member function named mix_name because that's nonsense
  // XXX consider eliminating this constraint
#if defined (__circle_lang__)
  and not requires(T t, P prop)
  {
    t.@("mix_" + P::name)(prop.value);
  }
#endif

  // T shouldn't have a member function .mix(prop) because that's nonsense
  // XXX consider eliminating this constraint
  and not requires(T t, P prop)
  {
    t.mix(prop);
  }

  // T shouldn't have a free function mix(t,prop) because that's nonsense
  // XXX consider eliminating this constraint
  and not requires(T t, P prop)
  {
    mix(t,prop);
  }
;


namespace detail
{


#if defined(__circle_lang__)
template<const char name[], class T, class... Args>
concept has_member_function_named = requires(T t, Args... args) { t.@(name)(args...); };
#endif


} // end detail


#if defined(__circle_lang__)
template<const char name_[], std::equality_comparable T, template<class> concept applicable /* = any_type */>
struct property
{
  template<class U>
  static constexpr bool is_applicable_v = applicable<U>;

  T value;
  constexpr static const char name[] = name_;

  constexpr property(T v) : value(v) {}

  constexpr property() : property(T()) {}

  constexpr property operator()(const T& v) const
  {
    return property(v);
  }

  template<class O>
    requires is_applicable_v<O&&> and detail::has_member_function_named<name, O&&>
  constexpr T operator()(O&& object) const
  {
    // return the value of the property
    return std::forward<O&&>(object).@(name)();
  }

  template<class O>
    requires is_applicable_v<O&> and detail::has_member_function_named<name, O&, const T&>
  constexpr void operator()(O& object, const T& value) const
  {
    // set the value of the property
    object.@(name)(value); 
  }
};
#else

#define DEFINE_PROPERTY_TEMPLATE(name_, applicable) \
template<class T> \
struct name_##_property \
{ \
  template<class U> \
  static constexpr bool is_applicable_v = applicable<U>; \
 \
  T value; \
  constexpr static const char name[] = #name_; \
 \
  constexpr name_##_property(T v) : value(v) {} \
 \
  constexpr name_##_property() : name_##_property(T()) {} \
 \
  constexpr name_##_property operator()(const T& v) const \
  { \
    return name_##_property(v); \
  } \
 \
  template<class O> \
    requires (is_applicable_v<O&&> and not std::same_as<std::remove_cvref_t<O>,T>) \
  constexpr T operator()(O&& object) const \
  { \
    return std::forward<O&&>(object).name_(); \
  } \
 \
  template<class O> \
    requires is_applicable_v<O&> \
  constexpr void operator()(O& object, const T& value) const \
  { \
    return object.name_(value); \
  } \
};
#endif


UBU_NAMESPACE_CLOSE_BRACE

#include "../detail/epilogue.hpp"

