#!/bin/bash
# Build script for Jupyter Book documentation
# Usage: Run from project root: ./docs/build.sh
#        Or from docs directory: cd docs && ./build.sh

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine project root (parent of docs/)
if [ -f "$SCRIPT_DIR/_config.yml" ]; then
    # We're in docs/ directory
    DOCS_DIR="$SCRIPT_DIR"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    # We're in project root
    PROJECT_ROOT="$SCRIPT_DIR"
    DOCS_DIR="$PROJECT_ROOT/docs"
fi

echo "📦 Installing atom package in editable mode..."
cd "$PROJECT_ROOT"
pip install -e . > /dev/null 2>&1 || pip install -e .

echo "📚 Building Jupyter Book..."
echo "   Project root: $PROJECT_ROOT"
echo "   Docs directory: $DOCS_DIR"

# Clean any existing build in project root (shouldn't exist, but clean it)
if [ -d "$PROJECT_ROOT/_build" ]; then
    echo "⚠️  Removing _build from project root (shouldn't be here)"
    rm -rf "$PROJECT_ROOT/_build"
fi

# Build from project root, pointing to docs directory
cd "$PROJECT_ROOT"

# Check Jupyter Book version
JB_VERSION=$(jupyter-book --version 2>/dev/null || echo "unknown")
echo "   Jupyter Book version: $JB_VERSION"

# Try different methods for different Jupyter Book versions
BUILD_SUCCESS=false

echo "   Attempting build with jupyter-book..."
if jupyter-book build docs/ --all > /tmp/jb_build.log 2>&1; then
    # Check if HTML was actually generated
    if [ -f "$DOCS_DIR/_build/html/index.html" ] || [ -f "$PROJECT_ROOT/_build/html/index.html" ]; then
        BUILD_SUCCESS=true
        echo "✅ Build successful with 'jupyter-book build docs/ --all'"
    else
        echo "⚠️  Command succeeded but no HTML generated. Checking logs..."
        cat /tmp/jb_build.log | tail -20
    fi
fi

if [ "$BUILD_SUCCESS" = false ] && command -v jb &> /dev/null; then
    echo "   Trying with 'jb' command..."
    if jb build docs/ > /tmp/jb_build.log 2>&1; then
        if [ -f "$DOCS_DIR/_build/html/index.html" ] || [ -f "$PROJECT_ROOT/_build/html/index.html" ]; then
            BUILD_SUCCESS=true
            echo "✅ Build successful with 'jb build docs/'"
        fi
    fi
fi

if [ "$BUILD_SUCCESS" = false ]; then
    echo "   Trying from docs directory..."
    cd "$DOCS_DIR"
    if jupyter-book build . --all > /tmp/jb_build.log 2>&1; then
        if [ -f "_build/html/index.html" ]; then
            BUILD_SUCCESS=true
            echo "✅ Build successful from docs directory"
        fi
    fi
    cd "$PROJECT_ROOT"
fi

if [ "$BUILD_SUCCESS" = false ]; then
    echo ""
    echo "❌ All build methods failed."
    echo ""
    echo "Error details:"
    cat /tmp/jb_build.log | tail -30
    echo ""
    echo "Your Jupyter Book version: $JB_VERSION"
    echo ""
    echo "🔧 RECOMMENDED FIX:"
    echo "   Jupyter Book v2.x has compatibility issues. Downgrade to v1.x:"
    echo "   pip install 'jupyter-book<2.0'"
    echo ""
    echo "   Then run this script again."
    exit 1
fi

# Clean up any _build in project root (shouldn't be there)
if [ -d "$PROJECT_ROOT/_build" ] && [ ! -f "$PROJECT_ROOT/_build/html/index.html" ]; then
    echo "⚠️  Removing empty _build from project root..."
    rm -rf "$PROJECT_ROOT/_build"
fi

echo ""
echo "✅ Build complete!"
if [ -f "$DOCS_DIR/_build/html/index.html" ]; then
    echo "📖 Documentation is available at: docs/_build/html/index.html"
    echo ""
    echo "To view in browser, run:"
    echo "  cd docs/_build/html && python -m http.server 8000"
    echo "  Then open http://localhost:8000 in your browser"
elif [ -f "$PROJECT_ROOT/_build/html/index.html" ]; then
    echo "📖 Documentation is available at: _build/html/index.html"
    echo "⚠️  Note: Build output is in project root, not docs/"
    echo ""
    echo "To view in browser, run:"
    echo "  cd _build/html && python -m http.server 8000"
else
    echo "⚠️  Warning: Could not find index.html. Build may have failed."
fi
