// note that this header file is special and does not use #pragma once

#if !defined(REPO_NAMESPACE)

// this branch is taken the first time this header is included

#  if defined(REPO_NAMESPACE_OPEN_BRACE) or defined(REPO_NAMESPACE_CLOSE_BRACE)
#    error "Either all of REPO_NAMESPACE, REPO_NAMESPACE_OPEN_BRACE, and REPO_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#  define REPO_NAMESPACE repo
#  define REPO_NAMESPACE_OPEN_BRACE namespace repo {
#  define REPO_NAMESPACE_CLOSE_BRACE }
#  define REPO_NAMESPACE_NEEDS_UNDEF

#elif defined(REPO_NAMESPACE_NEEDS_UNDEF)

// this branch is taken the second time this header is included

#  undef REPO_NAMESPACE
#  undef REPO_NAMESPACE_OPEN_BRACE
#  undef REPO_NAMESPACE_CLOSE_BRACE
#  undef REPO_NAMESPACE_NEEDS_UNDEF

#elif defined(REPO_NAMESPACE) or defined(REPO_NAMESPACE_OPEN_BRACE) or defined(REPO_CLOSE_BRACE)

// this branch is taken the first time this header is included, and the user has misconfigured these namespace-related symbols

#  if !defined(REPO_NAMESPACE) or !defined(REPO_NAMESPACE_OPEN_BRACE) or !defined(REPO_NAMESPACE_CLOSE_BRACE)
#    error "Either all of REPO_NAMESPACE, REPO_NAMESPACE_OPEN_BRACE, and REPO_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#endif

#if !defined(REPO_DETAIL_NAMESPACE)

// allow the user to define a singly-nested namespace for private implementation details
#  define REPO_DETAIL_NAMESPACE detail
#  define REPO_DETAIL_NAMESPACE_NEEDS_UNDEF

#elif defined(REPO_DETAIL_NAMESPACE_NEEDS_UNDEF)

#  undef REPO_DETAIL_NAMESPACE
#  undef REPO_DETAIL_NAMESPACE_NEEDS_UNDEF

#endif

