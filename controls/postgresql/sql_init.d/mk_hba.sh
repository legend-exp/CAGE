cat > $PGDATA/pg_hba.conf <<- _EOF1_
# TYPE  DATABASE        USER            ADDRESS                 METHOD
# "local" is for Unix domain socket connections only
local   all             postgres                                peer
local   cage_sc_db      all                                     md5
# IPv4 local connections:
host    cage_sc_db      cage_db_user    samenet                 md5
host    cage_sc_db      cage_db_read    samenet                 md5
_EOF1_
