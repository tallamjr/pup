#!/bin/bash
# Code Coverage Script for Pup Video Processing Project
#
# This script provides convenient commands for generating and viewing
# code coverage reports using cargo-llvm-cov.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Coverage directories
COVERAGE_DIR="target/coverage"
HTML_DIR="${COVERAGE_DIR}/html"
LCOV_FILE="${COVERAGE_DIR}/lcov.info"

# Functions
print_usage() {
    echo -e "${BLUE}Pup Coverage Tool${NC}"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  summary      Generate and display coverage summary"
    echo "  html         Generate HTML coverage report"
    echo "  lcov         Generate LCOV report for CI/CD"
    echo "  all          Generate all coverage reports"
    echo "  open         Open HTML coverage report in browser"
    echo "  clean        Clean coverage artifacts"
    echo "  check        Check if coverage meets minimum threshold"
    echo ""
    echo "Examples:"
    echo "  $0 summary   # Quick coverage check"
    echo "  $0 html      # Generate detailed HTML report"
    echo "  $0 all       # Generate all reports and open HTML"
}

check_prerequisites() {
    if ! command -v cargo-llvm-cov &> /dev/null; then
        echo -e "${RED}Error: cargo-llvm-cov not found${NC}"
        echo "Install with: cargo install cargo-llvm-cov"
        exit 1
    fi
}

generate_summary() {
    echo -e "${BLUE}Generating coverage summary...${NC}"
    cargo llvm-cov test --lib --summary-only
}

generate_html() {
    echo -e "${BLUE}Generating HTML coverage report...${NC}"
    mkdir -p "${COVERAGE_DIR}"
    cargo llvm-cov test --lib --html --output-dir "${HTML_DIR}"
    echo -e "${GREEN}HTML report generated: ${HTML_DIR}/index.html${NC}"
}

generate_lcov() {
    echo -e "${BLUE}Generating LCOV coverage report...${NC}"
    mkdir -p "${COVERAGE_DIR}"
    cargo llvm-cov test --lib --lcov --output-path "${LCOV_FILE}"
    echo -e "${GREEN}LCOV report generated: ${LCOV_FILE}${NC}"
}

open_html_report() {
    if [[ -f "${HTML_DIR}/index.html" ]]; then
        echo -e "${BLUE}Opening coverage report in browser...${NC}"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open "${HTML_DIR}/index.html"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            xdg-open "${HTML_DIR}/index.html"
        else
            echo -e "${YELLOW}Please open manually: ${HTML_DIR}/index.html${NC}"
        fi
    else
        echo -e "${RED}HTML report not found. Run '$0 html' first.${NC}"
        exit 1
    fi
}

clean_coverage() {
    echo -e "${BLUE}Cleaning coverage artifacts...${NC}"
    rm -rf "${COVERAGE_DIR}"
    rm -rf target/llvm-cov*
    echo -e "${GREEN}Coverage artifacts cleaned${NC}"
}

check_coverage_threshold() {
    local threshold=${1:-40}  # Default threshold of 40%

    echo -e "${BLUE}Checking coverage threshold (${threshold}%)...${NC}"

    # Run coverage and capture output
    local coverage_output
    coverage_output=$(cargo llvm-cov test --lib --summary-only 2>&1)

    # Extract overall line coverage percentage
    local line_coverage
    line_coverage=$(echo "$coverage_output" | grep "TOTAL" | awk '{print $NF}' | sed 's/%//')

    if (( $(echo "$line_coverage >= $threshold" | bc -l) )); then
        echo -e "${GREEN}✓ Coverage ${line_coverage}% meets threshold ${threshold}%${NC}"
        return 0
    else
        echo -e "${RED}✗ Coverage ${line_coverage}% below threshold ${threshold}%${NC}"
        return 1
    fi
}

# Main script logic
case "${1:-summary}" in
    "summary")
        check_prerequisites
        generate_summary
        ;;
    "html")
        check_prerequisites
        generate_html
        ;;
    "lcov")
        check_prerequisites
        generate_lcov
        ;;
    "all")
        check_prerequisites
        echo -e "${BLUE}Generating all coverage reports...${NC}"
        generate_html
        generate_lcov
        echo ""
        generate_summary
        echo ""
        echo -e "${GREEN}All reports generated successfully!${NC}"
        read -p "Open HTML report in browser? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            open_html_report
        fi
        ;;
    "open")
        open_html_report
        ;;
    "clean")
        clean_coverage
        ;;
    "check")
        check_prerequisites
        threshold=${2:-40}
        check_coverage_threshold "$threshold"
        ;;
    "help"|"-h"|"--help")
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
