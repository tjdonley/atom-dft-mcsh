# Documentation Build Guide

This directory contains the Jupyter Book documentation for Atom.

**All documentation files and configuration are contained within this `docs/` directory.**

## Quick Start

1. **Install dependencies (from project root):**
   ```bash
   cd /path/to/delta/atom
   pip install -e ".[docs]"
   pip install -e .
   ```

2. **Build the documentation (from project root):**
   ```bash
   # Option 1: Use the build script
   ./docs/build.sh
   
   # Option 2: Build manually
   jupyter-book build docs/ --all
   ```

3. **View the documentation:**
   ```bash
   # Start a local server
   cd docs/_build/html
   python -m http.server 8000
   
   # Then open http://localhost:8000 in your browser
   ```

## Important Notes

- **All files in `docs/`**: Configuration files (`_config.yml`, `_toc.yml`) and all documentation are in this directory
- **Build from project root**: Run `jupyter-book build docs/` from the project root (`delta/atom/`), not from inside `docs/`
- **Output in `docs/_build/`**: All build outputs are contained within `docs/_build/`
- **Use `--all` flag**: This ensures all formats are built correctly
- **Code execution**: Code cells will be executed during build (configured in `_config.yml`)

## Development Workflow

1. **Edit documentation files** in `docs/`
2. **Modify atom code** in `atom/` (outside docs/)
3. **Rebuild documentation (from project root):**
   ```bash
   jupyter-book build docs/ --all
   ```
4. **View results** in `docs/_build/html/index.html`

## Troubleshooting

### Error: "EISDIR: illegal operation on a directory" or "No site configuration found"
- Make sure you're running the command from the **project root** (`delta/atom/`), not from inside `docs/`
- Use: `jupyter-book build docs/ --all` (from project root)
- Verify `_config.yml` and `_toc.yml` exist in `docs/` directory

### Build output is empty
- Check that all dependencies are installed: `pip install -e ".[docs]"`
- Verify the `_config.yml` and `_toc.yml` files are in `docs/` and are correct
- Try cleaning and rebuilding: `rm -rf docs/_build && jupyter-book build docs/ --all`

### Code cells not executing
- Check `_config.yml` has `execute_notebooks: "force"`
- Ensure atom package is installed: `pip install -e .`
- Check for import errors in the code cells

## File Structure

All files are contained within `docs/`:

```
docs/
├── _config.yml          # Jupyter Book configuration
├── _toc.yml            # Table of contents
├── build.sh            # Build script
├── intro.md            # Homepage
├── installation.md     # Installation guide
├── tutorials/          # Tutorial files
│   ├── 01_basic_solver.md
│   └── 02_data_loading.md
├── cookbook.md         # Cookbook examples
├── api/                # API documentation
│   └── reference.md
└── _build/             # Build output (generated, not in git)
    └── html/           # HTML output
        └── index.html
```
