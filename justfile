# Alias for running all checks
alias a := all


# Run tests using Pytest
test:
    pytest tests/ -v 

# Type check code using Ty
types:
    ty check src/pocketeer

# Lint code using Ruff
lint:
    ruff check src/pocketeer

# Format code using Ruff
format:
    ruff format src/pocketeer

# Fix ruff issues
fix:
    ruff check . --fix
    
# Check docstring coverage
cov:
    interrogate src/ -v

# Run all checks as in CI
all:
    ty check src/pocketeer
    ruff check src/ tests/
    ruff format src/ tests/
    interrogate src/ -v
    pytest tests/ -n=auto -v  


# Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf site/
	rm -rf dist/
	rm -rf pockets/