#pragma once

#include "../../detail/prologue.hpp"
#include "workspace.hpp"

namespace ubu
{
namespace detail
{

template<class T>
concept has_local_workspace_member_variable = requires(T arg)
{
// XXX WAR circle bug
#if defined(__circle_lang__)
  arg.local_workspace;
  workspace<decltype(arg.local_workspace)>;
#else
  { arg.local_workspace } -> workspace;
#endif
};

template<class T>
concept has_get_local_workspace_member_function = requires(T arg)
{
  { arg.get_local_workspace() } -> workspace;
};

template<class T>
concept has_get_local_workspace_free_function = requires(T arg)
{
  { get_local_workspace(arg) } -> workspace;
};


struct dispatch_get_local_workspace
{
  template<class T>
    requires has_local_workspace_member_variable<T&&>
  constexpr workspace decltype(auto) operator()(T&& arg) const
  {
    return std::forward<T>(arg).local_workspace;
  }

  template<class T>
    requires (not has_local_workspace_member_variable<T&&>
              and has_get_local_workspace_member_function<T&&>)
  constexpr workspace decltype(auto) operator()(T&& arg) const
  {
    return std::forward<T>(arg).get_local_workspace();
  }

  template<class T>
    requires (not has_local_workspace_member_variable<T&&>
              and not has_get_local_workspace_member_function<T&&>
              and has_get_local_workspace_free_function<T&&>)
  constexpr workspace decltype(auto) operator()(T&& arg) const
  {
    return get_local_workspace(std::forward<T>(arg));
  }
};

} // end detail


inline constexpr detail::dispatch_get_local_workspace get_local_workspace;

template<class T>
using local_workspace_t = std::remove_cvref_t<decltype(get_local_workspace(std::declval<T>()))>;

} // end ubu

#include "../../detail/epilogue.hpp"

