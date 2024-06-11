This directory contains tests exercising functionality defined by this repository.

To build and execute all test programs in all subdirectories rooted in this directory:

    $ make all

Protip: to get color-coded build output, pipe to `ccze`:

    $ make all | ccze -A

# Subdirectories

    * `ubu` : Unit tests for functionality defined by the library.
    * `zippyness` : Performance regression tests for parallel algorithms implemented using library components. So-named to force this subdirectory to build last.

