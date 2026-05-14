# Environment Information

This file documents the virtual environments used in this project. These environments are excluded from version control to keep the repository clean.

## Environments

### venv
- **Type:** Standard Python Virtual Environment
- **Python Version:** 3.11.9
- **Location:** `venv/`
- **Configuration:** `pyvenv.cfg` present.
- **Notes:** Main environment used for development and running the backend/scripts.

### venv_new
- **Type:** Custom or Incomplete Environment
- **Location:** `venv_new/`
- **Notable Packages:** `torch` (found in `Lib/site-packages/torch`)
- **Notes:** This directory appears to contain some packages but lacks the standard `Scripts` or `bin` directories and `pyvenv.cfg`. It might be a partial installation or a specific package cache.

## Git Exclusion
Both `venv/` and `venv_new/` are listed in `.gitignore` to prevent them from being pushed to GitHub.
