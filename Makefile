.PHONY: help up down build logs index migrate

help:
	@echo "Makefile for nexlab-brain"
	@echo ""
	@echo "Usage:"
	@echo "  make up              - Start all services in detached mode"
	@echo "  make down            - Stop and remove all services"
	@echo "  make build           - Build or rebuild services"
	@echo "  make logs            - View output from containers"
	@echo "  make index           - Run the document indexing script"
	@echo "  make migrate         - Apply database migrations"

up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build

logs:
	docker-compose logs -f

index:
	docker-compose exec app python scripts/run_indexing.py

migrate:
	docker-compose exec app alembic upgrade head
