# Data Run Virtualization
* Check out the Dragonfly/Dripline branch of interest in their respective submodules within this repo.
  - One way to do this is to change the default line
    ```
    FROM project8/dragonfly:latest
    ```
    in the Dockerfile for the following lines: 
    ```
    FROM project8/dripline-python
    RUN pip install <any_extra_dependencies>
    RUN git clone https://github.com/project8/dragonfly.git && cd dragonfly && git checkout <your_dragonfly_branch> && pip install .
    ```
* All Dragonfly services are controlled and monitored by Supervisor which will be installed and ran in the insectarium_dragonfly_1 container upon initialization of the container or container network.
* Service config files reside in insectarium/dragonfly/config/ and are started by supervisor via their specification in insectarium/dragonfly/etc.supervisord/conf.d/test_run.conf

# What services are running
1. not-glenlivet is a "kv_store" where each endpoint just stores a value and emits it on some schedule. Config: kv_store_tutorial.yaml
2. not-sc_db connects to the postgres database and logs the values emitted by not-glenlivet. **Note** that only sensors configured in the postgres service will be able to be stored in the database (i.e. named endpoints must be in the endpoint_id_map); this is done via addition (SQL) of new endpoints in the insectarium/postgresql/sql_init.d/50_cage_endpoints.sql file. Config: slow_control_table.yaml
3. not-snapshot_service serves as the interface between us and the postgres database. This services is used to collect "snapshots" of values of sensors in the db both at the beginning and at the end of a run, constituting metadata. Config: slow_control_snapshot.yaml
4. not-run_table serves as the interface between the DAQProvider and the postgres database. It is utilized to collect useful information about a run such as the run name and start time. Config: run_table.yaml
5. not-status_multido serves as a replacement for our insert status and/or run metadata MultiGet endpoints' host; these are the endpoints of interest for the pre-run metadata collection. Config: status_multigets.yaml
6. not-generic_daq is the generic DAQProvider service and serves as our interface to simulate a run. Config: generic_daq.yaml

# Adding new services
A dragonfly "service" (in the dripline sense of the word) can be added to this "service" (docker-compose sense) without needing to run in its own container. In fact, that is desirable since then it can be controlled using supervisord/cesi. The steps for adding a new service are:
1. Write a dragonfly config file and place it in the config subdirectory (note that while most config parameters from a production service are still valid, the containers do not not map one-to-one with lab server names, and so you will need to change these - see the config files already provided for examples.)
2. Write a supervisord config file and place it in etc.supervisord/conf.d. You can just copy either of the example files and change the config file to be the one you added in the prior step. At the moment, all services are specified within the test_run.conf supervisor file.

# FAQ

### I've made changes in dripline-python and/or dragonfly without effect; what gives?
The container uses the project8/dragonfly base image which ships with the latest version of dragonfly (and a proper version of dripline-python). Even if you volume mount a version of the code with changes, those do not show up in the installed code. You have two options:

1. Use `docker exec` to connect to the the container (with source volume mounted) and then `python setup.py develop` the package you want to work on, this should create the proper symbolic links. You will need to restart the python processes for the changes to take effect. Fortunately supervisor controls them so that's a single click of the restart button
2. If you are trying to develop code, I suggest starting your own dragonfly container, using the `--net=insectarium_default` flag. This will place your container on the same network as everything in the compose project. All the other services will be running and you can interact with them, but your particular project is in a container which can be stopped/started etc. as needed.

### How do I interact with a container and its applications?
For containers built from an image with bash, you can run <code>docker exec -it <container_name> bash</code>. This will place you in a bash shell within the container, giving you access to its applications.

### How do I use supervisor to monitor services?
You will need to access the container in which supervisor is running, that is "insectarium_dragonfly_1", to use the supervisor command line interface. Full documentaion of Supervisor can be found [here](http://supervisord.org/). However, here are a few useful commands. In the root directory of the container, you can:
  - check service status by <code>supervisorctl status</code>
  - start a service <code>supervisorctl start service_name</code>
  - start all services <code>supervisorctl start all</code>
  - stop a service <code>supervisorctl stop service_name</code>
  - stop all services <code>supervisorctl stop all</code>.

### Some services aren't running and I can't start them with CESI (they show FAILED or EXITED)
This is a known issue on Mac and a limitation of CESI. Furthermore, some services are known to fatally exit as the container network is being set up at execution. To fix in one line do `docker exec -it insectarium_dragonfly_1 bash -c "supervisorctl start all"`. You can also use `docker exec` to connect to the container and use `supervisorctl` to work with it interactively.

### Where can I access a service's logs? (Useful for debugging)
As specified in etc.supervisord/conf.d/test_run.conf for each running service, the log and err file of a service will be found at the addresses stdout_logfile and stderr_logfile respectively within the insectarium_dragonfly_1 container. Use <code>cat</code> to view the file.

### How can I start a virtual run?
As an example, go into the bash shell of the insectarium_dragonfly_1 container and run <code>dragonfly cmd -vv -b rabbit_broker daq_interface.start_timed_run virtual_run 100</code> to start a timed run with name "virtual run" for 100 seconds.
