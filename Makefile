# 1. VARIABLE DEFINITIONS
# Use := for immediate assignment and ?= for default values that can be overridden
PROJECT_NAME := end-to-end-semiconductor-yield-and-analytics
COMPOSE_FILE := docker-compose.yml
ENV_FILE := .env

export DOCKER_BUILDKIT = 1
export COMPOSE_DOCKER_CLI_BUILD = 1

# 2. ENVIRONMENT SETUP
# This loads your secrets/configs from .env into the Makefile environment
ifneq ("$(wildcard $(ENV_FILE))","")
    include $(ENV_FILE)
    export
endif

# 3. HELP SYSTEM (The most important part)
# This snippet automatically generates documentation from your comments
.PHONY: help
help: ## Display this help screen
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# 4. DOCKER COMMANDS
.PHONY: build up down logs
build: ## Rebuild all custom application images (ingestion, processing)
	docker compose --progress=plain -f $(COMPOSE_FILE) -p $(PROJECT_NAME) build 

up: ## Start all services in the background
	docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) up -d

down: ## Stop and remove all services and networks
	docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) down

logs: ## Follow logs for all services
	docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) logs -f

# 5. UTILITIES & CLEANUP
.PHONY: clean prune
clean: ## Remove temporary files, pycache, etc.
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache

prune: ## Deep clean Docker (use with caution)
	docker system prune -af --volumes

# 6. DEVELOPMENT/TESTING
.PHONY: test
test: ## Run test suite
	docker compose run --rm app pytest .