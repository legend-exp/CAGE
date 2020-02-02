# The CAGE slow controls system

## Contents

(Click the links for quick navigation)

* [What is it?](#what-is-it)
    * [How does the broker work?](#how-does-the-broker-work)
    * [Where does the database save the data?](#where-does-the-database-save-the-data)
* [User Guide](#user-guide)
  * [Operations on mjcenpa](#operations-on-mjcenpa)
    * [Installing broker software](#installing-broker-software)
    * [Querying the database](#querying-the-database)
    * [Operating the container system](#operating-the-container-system)
    * [Advanced usage](#advanced-usage)
  * [Operations on a Raspberry Pi](#operations-on-a-raspberry-pi)
    * [Installation](#installation)
    * [Automatic background processes with supervisorctl](#automatic-background-processes-with-supervisorctl)
    * [Using dragonfly](#using-dragonfly)
    * [Manual background processes with tmux](#manual-background-processes-with-tmux)


## What is it?

  * **A "main server" or "broker"** run on a computer in the lab (`mjcenpa`), hosting a database storing "slow controls" values of interest (pressure sensors, temperature, etc.)  It runs a set of Docker containers, explained below.  
    * We use the Metabase utility to conveniently display information that has been saved to the database, via any web browser connected to the UW network.  The IP address should be obtained from a member of the CAGE group.
    * The broker allows commands to be sent to certain hardware devices (such as high voltage modules), from any "node" on the system, to any other node.  


  * **An "auxiliary system" (usually a Raspberry Pi)** connected to slow controls equipment, that periodically posts messages to the database. (`cagepi, mj60pi`)  Typically these processes are started with the RPi using the `supervisorctl` utility.  Other processes can be managed interactively with `tmux`.


#### How does the broker work?

  The principal ingredients are four Docker containers, run on the "main" or "broker" computer.  In the CAGE lab, this computer is `mjcenpa`.  
  They are controlled by the file `cage/controls/docker-compose.yaml`. 

  * **rabbitMQ broker** : This container, `rabbit_mq`, uses the standard rabbit container.  This is the communications backbone for all dripline communication.  The port 5672 is exposed. 

  * **postgres database** : This container, `psql_db`, is built from the postgres directory.  It contains a minimal set of tables for tracking endpoints and logging string, numeric, or json data types.  The standard postgres port 5432 is exposed and the database is persistent as a volume.

  * **metabase** : This container uses the standard metabase container.  It provides a graphical interface to the database and some customizability for users, which requires a persistent volume.  The exposed port 3000 allows users to navigate to the metabase with their machine's web browser.

  * **dragonfly** : This container is a custom build of dragonfly on dripline-python.  It provides the core (non-sensor) functionality - database interaction, slack relay, and a pinger.  All these services are managed together by `supervisor` (often `supervisorctl`), necessitating the last container...

  * **cesi** : This container builds CeSI for observing the status of processes across the broader controls ecosystem.  *This can eventually be deprecated in favor or a centralized container deployment like Kubernetes.*


#### Where does the database save the data?

  Both `psql_db` and `metabase` use "docker volumes" so that their data is persistent across multiple cycles of the container.  

  This means that the containers can be restarted without losing any database information.
  The build directive of these containers tests if the volume has been configured properly for the application, and will only rebuild if this is not true.

  Both dragonfly and cesi have shared files with your machine's directory structure to allow for local modifications and to preserve these in a git repo.
  Anything that feels like a config file should have this property - cesi's server config, dragonfly's service configs, dragonfly's supervisor process configs.


# User Guide

  **Note:** two credentials files are required for operation, `config.json` and `project8_authentications.json`.  
  If you want to use e.g. Metabase on your own laptop (instead of mjcenpa), you need to obtain both files from a member of the CAGE group.


## Operations on mjcenpa

#### Installing broker software
  ```
  cd ~/software
  git clone [https://github.com/legend-exp/CAGE.git] (or a personal fork)
  mv CAGE cage; cd cage
  git submodule update --init --recursive
  ```

  **NOTE:** the git submodules in the `dragonfly` folder will be downloaded in "detached HEAD" state, which means they will be "frozen" at that particular commit, no matter what other changes are made to the repository.

  Also, if you are developing code on your laptop, you likely will not need to run the submodule command, since only the RPis and broker computer need an actively working copy of `dragonfly` and its dependencies.

  **NOTE:** Docker must be running (check the icon in the system tray).


#### Querying the database

  If the database is active, on `mjcenpa` you can use a web browser to navigate to **[localhost:3000](localhost:3000)**, or if you are on your laptop on the UW network, **[mjcenpa_ip_address]:3000** will work.

  From there, you can click "Ask a question" and "Native Query" to bring up a SQL interface that can display values from the database.

  A few useful commands are below.
  Note that asterisks are "wildcards" here, and will return all available columns (timestamp, value_raw, value_cal, memo, etc.)

  * `select * from endpoint_id_map;` this will display a list of all "endpoints" in the database, i.e. "columns" in the table that are written to if their subsystem (typically the CAGE RPi or MJ60 RPi) is active.

  * `select value_cal, timestamp from numeric_data where endpoint_name = 'mj60_baseline' and timestamp > '2020-01-30T00:00';`  This can be modified to includ a date range (two `and` statments).  To display a plot with Metabase, change "Visualization" from Table (the default) to Line and set the appropriate X and Y axes.

  * `SELECT value_cal, timestamp, timestamp at time zone 'gmt' at time zone 'pst' as time FROM numeric_data WHERE endpoint_name='cage_topHat_temp' AND timestamp>'2020-01-30T00:00';` By default, `timestamp` is in GMT time.  This statement converts the timestamp to Pacific Standard Time.

  * `SELECT * FROM numeric_data WHERE endpoint_name='mj60_baseline' ORDER BY timestamp DESC;`  This shows recent data.

  * `SELECT to_timestamp(AVG(extract(epoch from timestamp))),AVG(value_cal) FROM numeric_data
  WHERE endpoint_name='mj60_temp' AND value_cal<100 AND timestamp > '2019-06-01' AND timestamp < '2019-08-10'
  GROUP BY FLOOR(extract(epoch from timestamp)/3600) ORDER BY MIN(timestamp) ASC;` : This is a really neat command from Walter, that downsamples the data and only returns one point per hour (averaged from all the points in the range).  This way you can get more than 3 days of data to display on Metabase, with a selectable downsample range.

  **NOTE:** The `CAGE/examples` folder contains a routine **[sql_to_pandas.py](https://github.com/legend-exp/CAGE/blob/master/examples/sql_to_pandas.py)** which provides a few simple examples of reading the database in Python, and converting to Pandas/DataFrame formats, which is an excellent starting point for making custom plots and performing more complicated analyses.

  The database can also be accessed via terminal (and SSH) alone:
  ```
  [ssh into mjcenpa]
  cd ~/software/cage/controls
  docker-compose exec psql_db psql -U cage_db_user cage_sc_db
  ```

  From here, we can run standard SQL queries: 
  ```
  cage_sc_db=> \d  [shows a list of relations]
  cage_sc_db=> SELECT * FROM endpoint_id_map;
  ```

  Endpoint names can be corrected:
  ```
  UPDATE endpoint_id_map SET endpoint_name='cage_magic' WHERE endpoint_name='cage_sparkles'
  ```


#### Operating the container system`

  The framework is built on docker-compose, which manages an ecosystem of Docker containers, storage volumes, and networks.
  Several relevant commands are given below, and must be run in the directory containing the file `docker-compose.yaml`.

  * `docker-compose down`: bring down an active or misbehaving system

  * `docker-compose ps`, or `docker ps`: List containers, status, and the port forward structure

  * `docker-compose up -d`: restart the container system in the background.

  * `docker-compose exec [container_name] <cmd>` will execute a bash command in a specific container.

    * `<cmd>` can open a persistent terminal in that container if it has no definite completion (e.g., `bash`)
    
    * `<cmd>` can promptly return a value if it has defined completion (e.g., `date`)


#### Advanced usage

  Under the hood, dragonfly uses a service called RabbitMQ.  (Clint loves this.)
  It has a web interface that can be run from `mjcenpa` at the web address **[cageIP]:5672**.  

  Some quick notes:
  * This shows the various running queues of the database system
  * All the data goes into the `alerts` exchange
  * A message to the requests exchange expects a response

  We can also get a live display via terminal of the data going into the `alerts` exchange on mjcenpa, by running a command directly inside the dragonfly container:
  ```
  cd ~/software/cage/controls
  docker-compose exec dragonfly dragonfly monitor -e alerts -k sensor_value.# -b rabbit_broker
  ```
  This under the hood behavior is described on github at `project8/dragonfly/subcommands/message_monitor.py`.


## Operations on a Raspberry Pi

  Here we assume you have begun an SSH session with the CAGE or MJ60 RPi's:
  ```
  ssh pi@[IP ADDRESS]
  ```

#### Installation

  First, you will need to obtain the `config.json` and `project8_authentications.json` files from a member of the CAGE group.

  We first set up a Python virtual environment (used for all dragonfly commands):
  * `python3 -m venv cage_venv` : creates a folder automatically
  * `source ~/cage_venv/bin/activate` : activates the venv.
  * `deactivate` : removes a user from the venv.

  To install the CAGE repo and its dependencies:
  ```
  git clone [primary cage url, or a fork]
  mv CAGE cage; cd cage
  git submodule update --init --recursive
  python3 -m pip install -e ~/cage/controls/dragonfly/dripline-python
  python3 -m pip install -e ~/cage/controls/dragonfly/dragonfly[colorlog,gpio,ads1x15]
  python3 -m pip install adafruit-circuitpython-max31865
  python3 -m pip install pyserial
  ```

  To activate this behavior on a "fresh" RPi, one must make two symlinks.  
  Here we use the `cagepi` as an example.
  ```
  cd /etc/supervisor
  sudo ln -sf /home/pi/cage/controls/cagepi/conf.d conf.d
  sudo ln -sf /home/pi/cage/controls/cagepi/supervisord.conf supervisord.conf
  ```
  This tells `supervisor` (which is run at startup) to run the tasks in the `cagepi` folder, described by `conf.d` and `supervisord.conf` at startup of the RPi.  
  This is typically done for passive sensors such as temperature readouts, LN weights, and pressure gauges, which have no active controls -- they either work or they don't.


#### Automatic background processes with supervisorctl

  The `.yaml` files in the folders `CAGE/controls/[cagepi,mj60pi]` manage which sensors post messages to the database, and how often they do so.  Both RPis are configured to post messages automatically on startup using the `supervisorctl` utility.


  **Common `supervisorctl` commands**:

  * `supervisorctl help` : tells you the list of commands
  * `supervisorctl status` : list running (or failed!) processes
  * `supervisorctl` : enter an interactive control so you can run any command without prefixing it
  * `supervisor [start/stop/reload]` : useful to restart processes and look for bugs without rebooting the RPi

  * **Accessing log files:** These are stored in `/var/log/supervisor`.  Note, each process has a `processname-stdout-uniqueid.log` (and stderr) file which you can `vi` or `tail`.  


#### Using dragonfly

  All commands involving `dragonfly` must be run from the Python virtual environment, `cage_venv`.

  **Remote biasing of detectors:**
  * `[ssh into cagepi or mj60pi]`
  * `source ~/cage_venv/bin/activate`
  * `dragonfly get [mj60,cage]_hv_vmon -b mjcenpa`: displays current bias
  * `dragonfly set [mj60,cage]_hv_vset [integer] -b mjcenpa`: changes V_bias

  **Changing database logging interval (in seconds)**
  * `dragonfly get mj60_baseline.schedule_interval -b mjcenpa`: This can be done with any endpoint.  Typically slow controls values report once every 30 seconds, and should generally not report faster than once every 5 seconds for extended periods of time.
  * `dragonfly set mj60_baseline.schedule_interval 5 -b mjcenpa`: Sets the report rate for this endpoint in seconds.


  **Working with the CAENHV service:** This is the USB communication between the CAGE RPi and the CAEN HV card which controls the detector bias.  It is typically run by `supervisorctl` on the CAGE RPi.

  * `supervisorctl status` :check if "caenhv" is already running. If not, ask Walter or someone what to do / what's going on

  * dragonfly get cage_hv_status -b mjcenpa` : see if HV controls are "Killed", "Disabled", "Off", or "On".  If "Killed" or "Disabled": flip the switch up on the front panel of the CAEN HV (have someone show you if you haven't done it before). Re-check status and verify that you get "Off".

  * `dragonfly set cage_hv_status 1 -b mjcenpa` : if status was "Off"
  * `dragonfly set cage_hv_vset [value] -b mjcenpa` : set a new HV set point
  * `dragonfly get cage_hv_rdown -b mjcenpa`: show the rampdown speed (in V/sec)
  * `dragonfly set cage_hv_rdown [value] -b mjcenpa` : set the rampdown speed
  * `dragonfly get cage_hv_rup -b mjcenpa`: show the ramp up speed (in V/sec)
  * `dragonfly set cage_hv_rup [value] -b mjcenpa` : set the ramp up speed


#### Manual background processes with tmux

  `tmux` is a simple program on linux machines which allows a user to create an interactive terminal session which can be "sent to the background", or operated without interfering with your main termainal session.  Some useful links:
  * **[tmux Cheat Sheet](https://tmuxcheatsheet.com/)**
  * **[A Gentle Introduction to tmux](https://hackernoon.com/a-gentle-introduction-to-tmux-8d784c404340)**
  * **[StackOverflow: tmux vs. screen](https://superuser.com/questions/236158/tmux-vs-screen)**

  Useful tmux commands:
  * `tmux ls` : show running sessions 
  * `tmux new -s [name]` : start a new tmux session with [name]
  * `tmux a -t [name]` : attach to a running session
  * `tmux kill-session -t [name]` : kill a running tmux session

  From within a tmux session:
  * `Ctrl-b d` : detach from session
  * `Ctrl-b [arrow keys]` : scroll backwards/forwards in the tmux history

  **OUR RULE:** The HV interlock for each detector (CAGE or MJ60) must be run on its respective RPi.  **DO NOT** run the MJ60 interlock from the CAGE RPi, it will only create confusion.

  The HV interlock system `[cagepi,mj60pi]/interlock.yaml`, is an example of a process which should be manually activated and deactivated by users.  
  It should only be engaged during stable periods of operation, not when the bias voltage of a detector is being actively changed by an operator.  
  This is why `tmux` is used for this process.

  **To start/stop the interlock** (using the MJ60 RPi as an example):
  ```
  source ~/cage_venv/bin/activate
  cd cage/controls/mj60pi
  dragonfly serve -vv -c interlock.yaml  [starts the interlock]
  tmux ls  [make sure 'interlock' is visible, may also want to check its output]
  tmux kill-session -t interlock [stops the interlock]
  ```

