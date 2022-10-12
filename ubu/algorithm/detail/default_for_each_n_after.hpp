#pragma once

#include "../../detail/prologue.hpp"

#include "../../execution/executor/bulk_execute_after.hpp"
#include "../../execution/policy.hpp"
#include <iterator>
#include <utility>

namespace ubu::detail
{


// XXX E should be execution_policy_happening_t<P>
template<execution_policy P, happening H, std::random_access_iterator I, std::indirectly_unary_invocable<I> F>
auto default_for_each_n_after(P&& policy, H&& before, I first, std::iter_difference_t<I> n, F f)
{
  // get an executor
  auto ex = associated_executor(std::forward<P>(policy));

  // XXX n could be too large for the executor
  return bulk_execute_after(ex, std::forward<H>(before), n, [=](std::iter_difference_t<I> i)
  {
    f(first[i]);
  });
}


} // end ubu::detail

#include "../../detail/epilogue.hpp"

