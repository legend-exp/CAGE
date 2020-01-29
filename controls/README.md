# Containerized controls system for CAGE

This repository houses a minimal controls infrastructure for a lab teststand, customized for CAGE at UW.
It is envisioned to work on a local network involving other machines, with the core functionality provided here.
The principal ingredients conceived are:
* rabbitMQ broker : This container, `rabbit_mq`, uses the standard rabbit container.  This is the communications backbone for all dripline communication.  The standard rabbit port 15672 is exposed. 
* postgres database : This container, `psql_db`, is built from the postgres directory.  It contains a minimal set of tables for tracking endpoints and logging string, numeric, or json data types.  The standard postgres port 5432 is exposed and the database is persistent as a volume.
* metabase : This container uses the standard metabase container.  It provides a graphical interface to the database and some customizability for users, which requires a persistent volume.  The exposed port 3000 allows users to navigate to the metabase with their machine's browser.
* dragonfly : This container is a custom build of dragonfly on dripline-python.  It provides the core (non-sensor) functionality - database interaction, slack relay, and a pinger.  All these services are managed together by supervisor, necessitating the last container...
* cesi : This container builds CeSI for observing the status of processes across the broader controls ecosystem.  It can be deprecated in favor or a centralized container deployment like kubernetes.


# FAQ

### How do I make it work?
This framework is built on docker-compose, which will spin up an ecosystem of containers, volumes, and networks.
On a machine with docker-compose installed, you can `docker-compose up -d` to bring up the entire system in the background.
Consult docker-compose documentation for more details on interacting, in short:
* `docker-compose ps` will list the containers, their statuses, and the port forward structure
* `docker-compose exec dragonfly <cmd>` will execute a command in the specific container
  * <cmd> can open a persistent terminal in that container if it has no definite completion (e.g., `bash`)
  * <cmd> can promptly return a value if it has defined completion (e.g., `date`)
* `docker-compose down` will bring down the entire container set

### How do the persistent volumes work?
Both psql_db and metabase use docker volumes so that their data is persistent across multiple cycles of the container.
The build directive of these containers tests if the volume has been configured properly for the application, and will only rebuild if this is not true.
To force the volumes to rebuild (please be careful), use `docker-compose down -v` to remove the declared named volumes, or use `docker volume` to interact specifically with the volumes.

### How do the other volumes work?
Both dragonfly and cesi have shared files with your machine's directory structure to allow for local modifications and to preserve these in a git repo.
Anything that feels like a config file should have this property - cesi's server config, dragonfly's service configs, dragonfly's supervisor process configs.
