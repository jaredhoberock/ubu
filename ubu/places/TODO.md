This directory is intended to define a `place` concept.

The idea of a `place` is that it collects an `executor`, `asynchronous_allocator`, and `loader` into a single object.

It's intended to be more ergonomic to use a single `place` object than by providing separate objects to APIs like `bulk_execute_with_workspace`.

A `place` could either satisfy all of those concepts, or it could simply provide getters which return objects satisfying those concepts. Or maybe both.

