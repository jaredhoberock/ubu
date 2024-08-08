#pragma once

#include "../../../detail/prologue.hpp"

#include "../../causality/happening.hpp"
#include "../../execution/executors/concepts/executor.hpp"
#include "../../execution/executors/bulk_execute_after.hpp"
#include "allocate_after.hpp"
#include "concepts/asynchronous_allocation.hpp"
#include "concepts/asynchronous_allocator.hpp"
#include "detail/custom_allocate_and_zero_after.hpp"
#include <cstring>
#include <utility>

namespace ubu
{
namespace detail
{


// this is the type of allocate_and_zero_after
template<class T>
struct dispatch_allocate_and_zero_after
{
  // this dispatch path calls the customization of allocate_and_zero_after
  template<class A, class E, class B, class S>
    requires has_custom_allocate_and_zero_after<T,A&&,E&&,B&&,S&&>
  constexpr asynchronous_allocation auto operator()(A&& alloc, E&& exec, B&& before, S&& shape) const
  {
    return custom_allocate_and_zero_after<T>(std::forward<A>(alloc), std::forward<E>(exec), std::forward<B>(before), std::forward<S>(shape));
  }

  // this dispatch path calls allocate_after and then bulk_execute_after to fill the tensor
  template<coordinate S, asynchronous_allocator_of<T,S> A, executor E, happening B>
    requires (not has_custom_allocate_and_zero_after<T,A&&,E&&,B&&,S>
              and std::is_scalar_v<T>)
  constexpr asynchronous_allocation auto operator()(A&& alloc, E&& exec, B&& before, S shape) const
  {
    // asynchronously allocate the memory
    auto [allocation_finished, tensor] = allocate_after<T>(std::forward<A>(alloc), std::forward<B>(before), shape);

    // asynchronously zero the bits
    happening auto zero_finished = bulk_execute_after(std::forward<E>(exec), std::move(allocation_finished), shape, [tensor](auto coord)
    {
      // XXX generalize this to work with non-scalar T
      //     we need to use memset, but the problem is that &tensor[coord] may not be a raw pointer
      tensor[coord] = T{};
    });

    // return the pair
    return std::pair(std::move(zero_finished), tensor);
  }
};


} // end detail


namespace
{

template<class T>
constexpr detail::dispatch_allocate_and_zero_after<T> allocate_and_zero_after;

} // end anonymous namespace


} // end ubu


#include "../../../detail/epilogue.hpp"

