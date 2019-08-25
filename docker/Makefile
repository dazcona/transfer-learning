run:
	docker-compose up -d --build
dev:
	docker exec -it tf_container bash
build:
	docker-compose build
status:
	docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
logs:
	docker logs tf_container -f
down:
	docker-compose -f docker-compose.yml down -v
reset:
	docker stop $(docker ps -q); docker rm $(docker ps -a -q)
prune:
	docker network prune -f; docker volume prune -f; docker image prune -a
kill:
	make down; make reset; make prune; make run