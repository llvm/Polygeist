# BLAS Primitives Implementation and Raising Plan

## Overview
This document outlines the action plan for implementing BLAS primitives in C and raising them to MLIR using the Polygeist pipeline. The goal is to create a comprehensive library of BLAS operations that can be automatically optimized and scheduled.

## Target BLAS Primitives

### Level 1 BLAS (Vector-Vector Operations)
- **DOT**: `sum(x[i] * y[i])` - Dot product of two vectors
- **SCAL**: `x *= alpha` - Scale vector by scalar
- **COPY**: `y = x` - Copy vector
- **AXPY**: `y = alpha * x + y` - Scaled vector addition

### Level 2 BLAS (Matrix-Vector Operations)
- **GEMV**: `y = alpha * A * x + beta * y` - Matrix-vector multiply
- **GER**: `A = alpha * x * y^T + A` - Outer product (rank-1 update)

### Level 3 BLAS (Matrix-Matrix Operations)
- **GEMM**: `C = alpha * A * B + beta * C` - Matrix-matrix multiply

### Matrix Utility Operations
- **LACPY**: Copy matrix with optional transposition
- **LASCL**: `A *= alpha` - Scale matrix by scalar

## Naming Convention
Following standard BLAS naming: `[prefix][type][operation]`
- **Prefix**: `""` (none), `"cublas"`, `"cblas_"`
- **Type**: `s` (float), `d` (double), `c` (complex float), `z` (complex double)
- **Operation**: BLAS function name

Examples:
- `dgemv` - Double precision GEMV
- `sgemm` - Single precision GEMM
- `cublasDgemv` - CUDA double precision GEMV

## Action Items

### Phase 1: C Implementation (Foundation)
- [ ] **1.1** Create basic C implementations for all primitives
  - [ ] Implement with proper LDA/stride support
  - [ ] Include alpha/beta scaling parameters
  - [ ] Add comprehensive error checking
  - [ ] Create both simple and optimized versions

- [ ] **1.2** Create test harnesses
  - [ ] Unit tests for each primitive
  - [ ] Performance benchmarks
  - [ ] Verification against reference implementations
  - [ ] Edge case testing (zero dimensions, negative strides)

- [ ] **1.3** File structure organization
  ```
  blas/
  ├── src/
  │   ├── level1/          # DOT, SCAL, COPY, AXPY
  │   ├── level2/          # GEMV, GER
  │   ├── level3/          # GEMM
  │   └── utils/           # LACPY, LASCL
  ├── include/
  │   └── blas.h           # Header with all declarations
  ├── tests/
  │   ├── test_level1.c
  │   ├── test_level2.c
  │   └── test_level3.c
  └── simple/              # Minimal versions for Polygeist
      ├── simple_gemm.c
      ├── simple_gemv.c
      └── ...
  ```

### Phase 2: Polygeist Pipeline Integration
- [ ] **2.1** Test each primitive with cgeist
  - [ ] Convert C to initial MLIR: `cgeist primitive.c --function=* --resource-dir=/usr/lib/clang/14 --raise-scf-to-affine -fPIC -S -g -c -o primitive.mlir`
  - [ ] Verify successful parsing and basic structure
  - [ ] Document any conversion issues

- [ ] **2.2** Apply affine-to-linalg pipeline
  - [ ] Run: `polygeist-opt --affine-parallelize --raise-affine-to-linalg-pipeline primitive.mlir -o primitive_linalg.mlir`
  - [ ] Verify linalg operations are generated correctly
  - [ ] Check for proper loop nest structure

- [ ] **2.3** Apply debufferization
  - [ ] Run: `polygeist-opt --linalg-debufferize primitive_linalg.mlir -o primitive_debufferized.mlir`
  - [ ] Ensure tensor operations are created properly
  - [ ] Verify memory access patterns

- [ ] **2.4** Apply kernel extraction
  - [ ] Run: `polygeist-opt primitive_debufferized.mlir --linalg-to-kernel="kernel-library-path=/home/arjaiswal/Polygeist/generic_solver/kernel_library.mlir"`
  - [ ] Verify kernel definitions are generated
  - [ ] Check integration with existing kernel library

### Phase 3: Kernel Library Integration
- [ ] **3.1** Update kernel library
  - [ ] Add all BLAS primitive kernels to `kernel_library.mlir`
  - [ ] Ensure consistent naming and interfaces
  - [ ] Add metadata for scheduler integration

- [ ] **3.2** Create automation scripts
  - [ ] Script to process all BLAS primitives through pipeline
  - [ ] Batch processing with error handling
  - [ ] Output validation and reporting

- [ ] **3.3** Integration testing
  - [ ] Test composite operations using multiple primitives
  - [ ] Verify scheduler can handle BLAS operations
  - [ ] Performance validation against reference implementations

### Phase 4: Advanced Features
- [ ] **4.1** Multiple precision support
  - [ ] Implement float, double, complex variants
  - [ ] Template-based approach for code reuse
  - [ ] Type-specific optimizations

- [ ] **4.2** GPU backend support
  - [ ] CUDA implementations (`cublas*` variants)
  - [ ] ROCm implementations
  - [ ] Backend selection logic

- [ ] **4.3** Optimization variants
  - [ ] Blocked implementations for cache efficiency
  - [ ] Vectorized versions
  - [ ] Thread-parallel versions

### Phase 5: Scheduler Integration
- [ ] **5.1** Cost model development
  - [ ] Performance characterization of each primitive
  - [ ] Memory access pattern analysis
  - [ ] Scheduling heuristics

- [ ] **5.2** Composition analysis
  - [ ] Identify common BLAS operation patterns
  - [ ] Fusion opportunities (e.g., GEMV + AXPY)
  - [ ] Memory reuse optimization

- [ ] **5.3** End-to-end validation
  - [ ] Complex linear algebra algorithms
  - [ ] Performance comparison with optimized libraries
  - [ ] Correctness verification

## Implementation Priority

### High Priority (Core Operations)
1. **GEMM** - Most computationally intensive, widely used
2. **GEMV** - Foundation for many algorithms
3. **DOT** - Simple but fundamental
4. **AXPY** - Common in iterative methods

### Medium Priority (Supporting Operations)
5. **SCAL** - Simple scaling operation
6. **COPY** - Basic data movement
7. **GER** - Rank-1 updates

### Lower Priority (Utility Operations)
8. **LACPY** - Matrix copying
9. **LASCL** - Matrix scaling

## File Naming Convention
- C implementations: `[type][operation].c` (e.g., `dgemm.c`, `sgemv.c`)
- Simple versions: `simple_[operation].c` (e.g., `simple_gemm.c`)
- MLIR outputs: `[operation]_[stage].mlir` (e.g., `gemm_linalg.mlir`)

## Testing Strategy
- **Unit Tests**: Each primitive tested independently
- **Integration Tests**: Multiple primitives working together
- **Performance Tests**: Comparison with reference implementations
- **Pipeline Tests**: Full Polygeist pipeline for each primitive
- **Regression Tests**: Ensure changes don't break existing functionality

## Success Criteria
- [ ] All primitives successfully convert through Polygeist pipeline
- [ ] Generated kernels integrate with existing kernel library
- [ ] Performance within 80% of reference implementations
- [ ] Scheduler can effectively utilize BLAS operations
- [ ] Documentation and examples complete

## Dependencies
- Polygeist toolchain (cgeist, polygeist-opt)
- Kernel library infrastructure
- Testing framework
- Reference BLAS implementation for validation

## Timeline Estimate
- **Phase 1**: 2-3 weeks (C implementations and tests)
- **Phase 2**: 2-3 weeks (Pipeline integration)
- **Phase 3**: 1-2 weeks (Kernel library integration)
- **Phase 4**: 3-4 weeks (Advanced features)
- **Phase 5**: 2-3 weeks (Scheduler integration)

**Total**: ~10-15 weeks for complete implementation

## Notes
- Start with simple, correct implementations before optimizing
- Maintain compatibility with standard BLAS interfaces
- Document any limitations or assumptions
- Consider memory alignment and padding requirements
- Plan for both CPU and GPU backends from the beginning
