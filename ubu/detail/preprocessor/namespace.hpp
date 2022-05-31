// note that this header file is special and does not use #pragma once

#if !defined(UBU_NAMESPACE)

// this branch is taken the first time this header is included

#  if defined(UBU_NAMESPACE_OPEN_BRACE) or defined(UBU_NAMESPACE_CLOSE_BRACE)
#    error "Either all of UBU_NAMESPACE, UBU_NAMESPACE_OPEN_BRACE, and UBU_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#  define UBU_NAMESPACE ubu
#  define UBU_NAMESPACE_OPEN_BRACE namespace ubu {
#  define UBU_NAMESPACE_CLOSE_BRACE }
#  define UBU_NAMESPACE_NEEDS_UNDEF

#elif defined(UBU_NAMESPACE_NEEDS_UNDEF)

// this branch is taken the second time this header is included

#  undef UBU_NAMESPACE
#  undef UBU_NAMESPACE_OPEN_BRACE
#  undef UBU_NAMESPACE_CLOSE_BRACE
#  undef UBU_NAMESPACE_NEEDS_UNDEF

#elif defined(UBU_NAMESPACE) or defined(UBU_NAMESPACE_OPEN_BRACE) or defined(UBU_CLOSE_BRACE)

// this branch is taken the first time this header is included, and the user has misconfigured these namespace-related symbols

#  if !defined(UBU_NAMESPACE) or !defined(UBU_NAMESPACE_OPEN_BRACE) or !defined(UBU_NAMESPACE_CLOSE_BRACE)
#    error "Either all of UBU_NAMESPACE, UBU_NAMESPACE_OPEN_BRACE, and UBU_NAMESPACE_CLOSE_BRACE must be defined, or none of them."
#  endif

#endif

#if !defined(UBU_DETAIL_NAMESPACE)

// allow the user to define a singly-nested namespace for private implementation details
#  define UBU_DETAIL_NAMESPACE detail
#  define UBU_DETAIL_NAMESPACE_NEEDS_UNDEF

#elif defined(UBU_DETAIL_NAMESPACE_NEEDS_UNDEF)

#  undef UBU_DETAIL_NAMESPACE
#  undef UBU_DETAIL_NAMESPACE_NEEDS_UNDEF

#endif

