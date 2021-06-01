# ALTO
ALTO is a template-based implementation of the Adaptive Linearized Tensor Order format for storing and processing sparse tensors. It performs key tensor decomposition operations along every mode (dimension) using a unified approach that works on a single tensor copy. The current implementation supports multi-core processors and it allows effortless specialization of tensor kernels not only for common tensor orders, but also for typical application features (such as the decomposition rank). The detailed algorithms are described in the following paper:
* **ALTO: Adaptive Linearized Storage of Sparse Tensors**. Ahmed E. Helal, Jan Laukemann, Fabio Checconi, Jesmin Jahan Tithi, Teresa M. Ranadive, Fabrizio Petrini, Jeewhan Choi. In Proceedings of the ACM International Conference on Supercomputing (ICS), June 2021. [doi:10.1145/3447818.3461703](https://doi.org/10.1145/3447818.3461703). 

## Getting started
To compile the code, simply run `make` in the root directory. This will create the `cpd64` binary for running CANDECOMP/PARAFAC tensor decomposition (CPD) with a 64-bit ALTO mask. 
By default, the Intel ICPC compiler is used.

The code currently requires either Intel MKL or OpenBlas available on the system. See [Settings](#settings) for more configuration options.

## Usage
You can perform CPD on a given tensor like this:
```bash
./cpd64 --rank 16 -m 100 -i /path/to/tensor.tns
```
This runs CPD with a 64-bit ALTO mask, a rank-16 decomposition, and a maximum number of 100 iterations (or until convergence).

For only running the matricized tensor times Khatri-Rao product (MTTKRP) use:
```bash
./cpd64 --rank 16 -m 100 -i /path/to/tensor.tns -p -t 0 
```
This executes 100 iterations of the MTTKRP operation (`-p`) with a rank-16 decomposition on the target mode 0 (`-t 0`), i.e., the first mode.

Make sure you allocate enough huge pages if you have [activated the usage in config.mk](#transparent-huge-pages). 
Check out the `help` message for all possible runtime configurations.
```bash
./cpd -h
```

## Settings
All compilation settings can be changed in `config.mk`. Make sure to have no whitespace after your parameters.

#### Compiler
Currently, `ICC` and `GCC` are supported.

#### BLAS library
Currently, `MKL` or any library that conforms to the BLAS interface (tested with `OpenBlas`) is supported.

#### Length of ALTO mask
Set `ALTO_MASK_LENGTH` either to `64` or `128` for a 64-bit or 128-bit ALTO-mask, respectively.

#### Mode and Rank Specialization
The build system is setup so that certain mode and rank combinations are optimized at compile time.  To specify which modes and ranks are optimized, set MODES_SPECIALIZED and RANKS_SPECIALIZED to a comma-separated list of the desired values. To disable either mode or rank specialization, use 0 as the value to be specialized.

#### Alternative bit extraction
By default, ALTO uses the *Bit Manipulation Instruction Set 2 (BMI2)*.
If you are running on a machine with no support for BMI2 instructions (e.g., any ARM system), set `ALTERNATIVE_PEXT` to `true`.

#### Transparent Huge Pages
You can activate the usage of pre-allocated THPs by setting the option `THP_PRE_ALLOCATION` to `true`. 
By default, 2M THPs are used.
To use 1G pages instead, set the `USE_1G` definition in `common.h` to `1`:
```cpp
#define USE_1G 1
```

#### MEMTRACE output
For getting specific information about the memory access patterns in particular steps of the MTTKRP benchmark or the ALTO code, add `-Dmemtrace` or `-DALTO_MEM_TRACE` to `CFLAGS` in `Makefile`, respectively.


## Contributors
* Ahmed E. Helal (ahmed.helal@intel.com)
* Jan Laukemann  (jan.laukemann@intel.com)
* Fabio Checconi (fabio.checconi@intel.com)
* Jesmin Jahan Tithi (jesmin.jahan.tithi@intel.com)
* Jeewhan Choi (jeec@uoregon.edu)
* Yongseok Soh (ysoh@uoregon.edu)

## Licensing
ALTO is released under the MIT License. Please see the 'LICENSE' file for details.