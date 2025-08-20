# Code Coverage Guide for Pup

This document provides comprehensive guidance on code coverage tools, configuration, and interpretation for the pup video processing project.

## Coverage Tools and Setup

### Primary Tool: cargo-llvm-cov

The project uses `cargo-llvm-cov` as the primary code coverage tool, which leverages LLVM's source-based code coverage instrumentation for accurate and detailed coverage reporting.

#### Installation

```bash
# Install cargo-llvm-cov if not already installed
cargo install cargo-llvm-cov
```

#### Why cargo-llvm-cov?

- **Accuracy**: Uses LLVM source-based instrumentation for precise coverage data
- **Performance**: No extra layers between rustc, cargo, and llvm-tools
- **Features**: Supports line, region, and branch coverage (branch on nightly)
- **Integration**: Works seamlessly with cargo test, cargo nextest, and doc tests
- **Output formats**: HTML, LCOV, JSON, and text reports

### Configuration

The project includes a `.llvm-cov.toml` configuration file that:
- Generates HTML, LCOV, and text reports
- Excludes test files and build scripts from coverage analysis
- Focuses coverage on core library code
- Excludes problematic benchmark and integration tests

## Running Coverage Reports

### Quick Coverage Check

```bash
# Generate all coverage formats
cargo llvm-cov

# Library code only (recommended for development)
cargo llvm-cov test --lib

# Generate HTML report for detailed analysis
cargo llvm-cov test --lib --html --output-dir target/coverage/html
```

### Comprehensive Coverage

```bash
# All tests excluding problematic ones
cargo llvm-cov test --lib --test unit_tests --test roadmap_validation_tests

# Include documentation tests (requires nightly)
cargo +nightly llvm-cov test --lib --doctests

# Generate LCOV for CI/CD integration
cargo llvm-cov test --lib --lcov --output-path target/coverage/lcov.info
```

### Coverage Reports by Category

#### Unit Tests Only
```bash
cargo llvm-cov test --lib
```

#### Specific Test Modules
```bash
# Preprocessing module tests
cargo llvm-cov test preprocessing::tests

# Inference module tests
cargo llvm-cov test inference::tests

# Detection utilities tests
cargo llvm-cov test utils::detection::tests
```

## Understanding Coverage Reports

### Current Coverage Statistics

Based on the latest coverage analysis:

| Module | Line Coverage | Function Coverage | Region Coverage |
|--------|---------------|-------------------|-----------------|
| **preprocessing/mod.rs** | 91.24% | 100.00% | 94.21% |
| **utils/detection.rs** | 89.06% | 90.48% | 88.04% |
| **inference/mod.rs** | 89.80% | 88.89% | 92.00% |
| **metrics.rs** | 74.70% | 70.15% | 75.47% |
| **config/mod.rs** | 67.63% | 70.00% | 70.40% |
| **error.rs** | 72.63% | 66.67% | 74.62% |
| **inference/ort_backend.rs** | 49.32% | 52.38% | 45.29% |
| **pipeline/mod.rs** | 21.83% | 18.52% | 21.67% |
| **gst_plugins/** | 0.00% | 0.00% | 0.00% |
| **common.rs** | 0.00% | 0.00% | 0.00% |

**Overall Coverage**: 42.41% lines, 53.61% functions, 41.13% regions

### Coverage Quality Assessment

#### Well-Tested Modules ✅
- **preprocessing/mod.rs**: Excellent coverage (>90%) with comprehensive unit tests
- **utils/detection.rs**: Strong coverage (>88%) with good edge case handling
- **inference/mod.rs**: Good coverage (>88%) for core inference logic

#### Moderately Tested Modules ⚠️
- **metrics.rs**: Decent coverage (~75%) but could benefit from more edge case tests
- **config/mod.rs**: Adequate coverage (~70%) with room for validation improvements
- **error.rs**: Fair coverage (~73%) with opportunity for more error path testing

#### Needs Attention ❌
- **inference/ort_backend.rs**: Low coverage (~49%) - critical inference backend needs more tests
- **pipeline/mod.rs**: Very low coverage (~22%) - core pipeline functionality undertested
- **gst_plugins/**: No coverage (0%) - GStreamer plugins not tested in unit tests
- **common.rs**: No coverage (0%) - platform-specific code not tested

## Coverage Targets and Goals

### Immediate Targets (Short-term)
- **Overall line coverage**: Increase from 42% to 65%
- **Core modules**: Achieve >80% coverage for preprocessing, inference, and utils
- **Error handling**: Improve error path coverage in inference backend
- **Configuration**: Add validation tests for edge cases

### Medium-term Goals
- **Overall line coverage**: Reach 75%
- **Pipeline coverage**: Increase pipeline module coverage to >50%
- **Integration**: Add GStreamer plugin integration tests
- **Documentation**: Achieve >90% doc test coverage

### Long-term Vision
- **Overall line coverage**: Target 85%+ for production-ready code
- **Critical paths**: 95%+ coverage for inference and preprocessing
- **Platform-specific**: Coverage for macOS-specific functionality
- **Performance**: Coverage-aware performance regression testing

## Integration with CI/CD

### GitHub Actions Integration

```yaml
# Example CI workflow step
- name: Generate Coverage Report
  run: |
    cargo llvm-cov test --lib --lcov --output-path coverage.info

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.info
    flags: rust
    name: pup-coverage
```

### Coverage Badges

After integrating with a coverage service, add badges to README.md:

```markdown
[![Coverage](https://codecov.io/gh/tallamjr/pup/branch/main/graph/badge.svg)](https://codecov.io/gh/tallamjr/pup)
```

## Development Workflow

### Pre-commit Coverage Check

```bash
# Quick coverage check before committing
cargo llvm-cov test --lib --summary-only

# Fail if coverage drops below threshold
cargo llvm-cov test --lib --fail-under-lines 40
```

### Coverage-driven Development

1. **Write failing test** - Ensure new test fails
2. **Implement feature** - Write minimal code to pass test
3. **Check coverage** - `cargo llvm-cov test --lib`
4. **Add edge cases** - Improve coverage with additional tests
5. **Refactor** - Clean up while maintaining coverage

### Coverage Analysis Workflow

```bash
# 1. Generate comprehensive report
cargo llvm-cov test --lib --html --output-dir target/coverage/html

# 2. Open HTML report
open target/coverage/html/index.html

# 3. Identify uncovered lines (marked in red)

# 4. Write tests for uncovered code paths

# 5. Verify improvement
cargo llvm-cov test --lib --summary-only
```

## Advanced Coverage Features

### Branch Coverage (Nightly only)

```bash
# Requires nightly Rust
cargo +nightly llvm-cov test --lib --branch
```

### Coverage with Different Features

```bash
# Test with Python feature enabled
cargo llvm-cov test --lib --features python

# Test without default features
cargo llvm-cov test --lib --no-default-features
```

### Doctests Coverage

```bash
# Include documentation tests (nightly only)
cargo +nightly llvm-cov test --lib --doctests
```

## Coverage Exclusions

### Code Excluded from Coverage

The following code is intentionally excluded from coverage analysis:

1. **Platform-specific code**: macOS-only functions (common.rs)
2. **GStreamer plugins**: Tested separately in integration tests
3. **Demo code**: Example implementations not part of core library
4. **Build scripts**: build.rs and procedural macros
5. **Test utilities**: Helper functions only used in tests

### Marking Code for Exclusion

```rust
// Exclude function from coverage
#[cfg(not(coverage))]
fn debug_only_function() {
    // Debug implementation
}

// Exclude specific lines
let result = dangerous_operation(); // coverage:ignore-line

// Exclude entire module
#[cfg(not(tarpaulin_include))]
mod platform_specific {
    // Platform-specific code
}
```

## Troubleshooting Coverage Issues

### Common Issues and Solutions

#### Coverage Tool Not Found
```bash
# Reinstall cargo-llvm-cov
cargo install --force cargo-llvm-cov
```

#### Low Coverage Due to Unused Code
- Remove dead code or mark as `#[cfg(not(coverage))]`
- Add tests for genuinely needed but untested code

#### Flaky Integration Tests Affecting Coverage
- Exclude problematic tests with `--exclude-tests`
- Run coverage on unit tests only: `--lib`

#### Missing Coverage for Conditional Compilation
```bash
# Test different feature combinations
cargo llvm-cov test --lib --features "feature1,feature2"
cargo llvm-cov test --lib --no-default-features
```

### Coverage Report Interpretation

- **Red lines**: Uncovered code that should have tests
- **Yellow lines**: Partially covered (some branches not tested)
- **Green lines**: Fully covered code
- **Gray lines**: Excluded from coverage analysis

## Best Practices

### Writing Coverage-friendly Code
1. **Small functions**: Easier to achieve 100% coverage
2. **Early returns**: Use guard clauses for error conditions
3. **Explicit error handling**: Don't hide error paths in generic handlers
4. **Testable design**: Separate I/O from logic

### Test Organization
1. **One test per code path**: Each branch/condition should have a test
2. **Edge case focus**: Test boundary conditions and error states
3. **Integration balance**: Unit tests for coverage, integration for workflows
4. **Mock external dependencies**: Focus coverage on your code, not dependencies

### Coverage Maintenance
1. **Regular monitoring**: Track coverage trends over time
2. **Coverage gates**: Prevent coverage regression in CI
3. **Meaningful tests**: Prefer tests that catch real bugs over coverage games
4. **Documentation**: Keep coverage documentation updated

## Coverage Goals by Module

### Target Coverage Levels

| Module | Current | Target | Priority |
|--------|---------|--------|----------|
| preprocessing | 91% | 95% | Low |
| utils/detection | 89% | 93% | Low |
| inference/mod | 90% | 93% | Low |
| metrics | 75% | 85% | Medium |
| config | 68% | 80% | Medium |
| error | 73% | 85% | Medium |
| inference/ort_backend | 49% | 75% | **HIGH** |
| pipeline | 22% | 60% | **HIGH** |
| gst_plugins | 0% | 40% | High |
| common | 0% | 50% | Medium |

### Implementation Strategy

1. **Phase 1** (Immediate): Focus on critical paths in ort_backend and pipeline
2. **Phase 2** (Short-term): Improve test coverage for error handling and configuration
3. **Phase 3** (Medium-term): Add integration tests for GStreamer plugins
4. **Phase 4** (Long-term): Platform-specific and edge case coverage

This comprehensive coverage strategy ensures the pup project maintains high code quality while providing clear guidance for developers to write effective tests and maintain coverage standards.
