A migration of the Project 8 postgres database for the CAGE teststand, based primarily on work by Ben LaRoque.
In these instructions, a higher-valued port is used rather than the standard postgres port (5432) to avoid any conflict with other host database instance.
Any utilities attempting to access the database would need to be modified for this testing setup, which is not the case in the docker-compose environment.

# Usage
The minimal instructions to get started are

```
docker build -t <image_name> .
docker run -p 127.0.0.1:<PORT>:5432 <image_name>
```

my preferred steps usually look more like
```
docker build -t cage_pg .
docker run --rm -it --name=cage_postgres -p 127.0.0.1:65432:5432 cage_pg
```
but obviously others may prefer other options.

Remember that you can use `docker exec -it <container_name> bash` to start a shell in a new pty on a running container, from which  you can use the psql command to interact with the database in the usual way.

Explicitly you probably want to do this:
`docker ps`
figure out what the container is.  Then
`docker exec -it <container name> bash`
followed by
`psql -d cage_sc_db -U cage_db_user`.
Alternatively you can run database queries from your own command line by
`docker exec <container_name> psql -d cage_sc_db -U cage_db_user -c 'SELECT * FROM endpoint_id_map;'`

# Updating the database
The database is constructed every time an image is started (_see below for volume mounting_) using the SQL scripts located in the sql_init.d directory.
These scripts are executed in sort order so please continue to prefix the names with numbers and use sensible values (ie leaving large gaps to insert new items etc.).
It is possible to add new scripts or add commands to existing scripts, however it is also possible to dump the configuration of an existing database using:
```pg_dumpall -U cage_db_user --password --schema-only > cage_schema.sql```
Where you'll need to move and rename the output file, the standard cage user is listed here, and you will be prompted for the password.
Note that the above command will include commands to create a user named postgres which already exists in the container image, causing an error at runtime, you should remove two lines: `CREATE ROLE postgres;` and `ALTER ROLE postgres WITH ...;` before trying to launch the new container.

## Volume-mounting the database
The database is stored in the $PGDATA directory, declared in the postgres image to be /var/lib/postgresql/data.
It can therefore be recreated by mounting a previously-generated version to this volume so that the lengthy generation process need only run once.
This is performed automatically in the insectarium docker-compose, and that same volume can be accessed `docker run --rm -it --volume=cage_db:/var/lib/postgresql/data cage_sc_db`.
If you want to wipe the database to start clean, you will need to `docker-compose down -v` (remove volumes option) or use `docker volume` to remove the volume manually.

