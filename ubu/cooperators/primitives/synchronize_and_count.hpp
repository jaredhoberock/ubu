#pragma once

#include "../../detail/prologue.hpp"
#include "../../utilities/integrals/integral_like.hpp"
#include "../traits/cooperator_size.hpp"
#include <cassert>


namespace ubu
{
namespace detail
{

template<class C, class B>
concept has_synchronize_and_count_member_function = requires(C self, B value)
{
  { self.synchronize_and_count(value) } -> integral_like;
};

template<class C, class B>
concept has_synchronize_and_count_free_function = requires(C self, B value)
{
  { synchronize_and_count(self, value) } -> integral_like;
};

// this is the type of synchronize_and_count
struct dispatch_synchronize_and_count
{
  template<class C, class B>
    requires has_synchronize_and_count_member_function<C&&,B&&>
  constexpr integral_like auto operator()(C&& self, B&& value) const
  {
    return std::forward<C>(self).synchronize_and_count(std::forward<B>(value));
  }

  template<class C, class B>
    requires (not has_synchronize_and_count_member_function<C&&,B&&>
              and has_synchronize_and_count_free_function<C&&,B&&>)
  constexpr integral_like auto operator()(C&& self, B&& value) const
  {
    return synchronize_and_count(std::forward<C>(self), std::forward<B>(value));
  }

  template<cooperator C>
    requires (    not has_synchronize_and_count_member_function<C&&,bool>
              and not has_synchronize_and_count_free_function<C&&,bool>)
  constexpr cooperator_size_t<C&&> operator()(C&& self, bool value) const
  {
    // XXX the default implementation would allocate some memory and do a reduction
    printf("dispatch_synchronize_and_count: Default unimplemented.\n");
    assert(false);
    return 0;
  }

};

} // end detail


constexpr detail::dispatch_synchronize_and_count synchronize_and_count;


} // end ubu

#include "../../detail/epilogue.hpp"

