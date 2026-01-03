# Alias for running all checks
alias a := all


# Run tests using Pytest
test:
    uv run pytest tests/ -v 

# Type check code using Ty
types:
    uv run ty check src/pocketeer

# Lint code using Ruff
lint:
    uv run ruff check src/pocketeer

# Format code using Ruff
format:
    uv run ruff format src/pocketeer

# Fix ruff issues
fix:
    uv run ruff check . --fix
    
# Check docstring coverage
cov:
    uv run interrogate src/ -v

# Run all checks as in CI
all:
    uv run ty check src/pocketeer
    uv run ruff check src/ tests/
    uv run ruff format src/ tests/
    uv run interrogate src/ -v
    uv run pytest tests/ -n=auto -v  

# Sync uv 
sync:
    uv sync --all-extras

benchmark:
    uv run experiments/benchmark.py

analyze_pocket_properties:
    uv run --with pandas,matplotlib,seaborn,rich experiments/analyze_pocket_properties.py

# Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf site/
	rm -rf dist/
	rm -rf pockets/