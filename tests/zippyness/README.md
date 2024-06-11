This directory contains performance regression test programs implemented using this library's components.

Each directory `<directory>` contains source to build a separate test program whose name is `<directory>.out`.

To build and execute all test programs from this directory:

    $ make all

To build and execute the test program from within any subdirectory:

    $ make test

Protip: to get color-coded build output, pipe to `ccze`:

    $ make all | ccze -A

# Details

Each peformance regression test program implements a parallel algorithm. The program tests both the correctness of the algorithm and also checks for unexpected performance. By default, the program will test large problem sizes which may take a while to complete.

A test program can be executed "quickly" on a smaller problem size with the `quick` command line argument.

For example, to build and do a quick test named `foo`:

    $ make foo.out
    $ ./foo.out quick

# Style Guide

To the degree possible, these unit test programs should be self-contained. This makes it easy to isolate test failures and produce minimal reproducer programs for bug reports.

The tradeoff is that common constructs these programs use are repeated.

