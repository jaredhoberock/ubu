// note that this header file is special and does not use #pragma once

#if !defined(ASPERA_NAMESPACE)

// this branch is taken the first time this header is included

#  if defined(ASPERA_NAMESPACE_OPEN_BRACE) or defined(ASPERA_NAMESPACE_CLOSE_BRACE)
#    error "Either all of ASPERA_NAMESPACE, ASPERA_NAMESPACE_OPEN_BRACE, and ASPERA_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#  define ASPERA_NAMESPACE repo
#  define ASPERA_NAMESPACE_OPEN_BRACE namespace repo {
#  define ASPERA_NAMESPACE_CLOSE_BRACE }
#  define ASPERA_NAMESPACE_NEEDS_UNDEF

#elif defined(ASPERA_NAMESPACE_NEEDS_UNDEF)

// this branch is taken the second time this header is included

#  undef ASPERA_NAMESPACE
#  undef ASPERA_NAMESPACE_OPEN_BRACE
#  undef ASPERA_NAMESPACE_CLOSE_BRACE
#  undef ASPERA_NAMESPACE_NEEDS_UNDEF

#elif defined(ASPERA_NAMESPACE) or defined(ASPERA_NAMESPACE_OPEN_BRACE) or defined(ASPERA_CLOSE_BRACE)

// this branch is taken the first time this header is included, and the user has misconfigured these namespace-related symbols

#  if !defined(ASPERA_NAMESPACE) or !defined(ASPERA_NAMESPACE_OPEN_BRACE) or !defined(ASPERA_NAMESPACE_CLOSE_BRACE)
#    error "Either all of ASPERA_NAMESPACE, ASPERA_NAMESPACE_OPEN_BRACE, and ASPERA_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#endif

#if !defined(ASPERA_DETAIL_NAMESPACE)

// allow the user to define a singly-nested namespace for private implementation details
#  define ASPERA_DETAIL_NAMESPACE detail
#  define ASPERA_DETAIL_NAMESPACE_NEEDS_UNDEF

#elif defined(ASPERA_DETAIL_NAMESPACE_NEEDS_UNDEF)

#  undef ASPERA_DETAIL_NAMESPACE
#  undef ASPERA_DETAIL_NAMESPACE_NEEDS_UNDEF

#endif

