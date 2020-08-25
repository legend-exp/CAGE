--
-- PostgreSQL database cluster dump
--

SET default_transaction_read_only = off;

SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;

--
-- Roles
--

CREATE ROLE cage_db_admin;
ALTER ROLE cage_db_admin WITH SUPERUSER INHERIT CREATEROLE CREATEDB LOGIN REPLICATION PASSWORD 'md5aca48ab210561816e9c0f9f5a4689b9e';
CREATE ROLE cage_db_user;
ALTER ROLE cage_db_user WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION PASSWORD 'md533e214f6f50bdbd7e8fe18de05765c45';
CREATE ROLE cage_db_read;
ALTER ROLE cage_db_read WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION PASSWORD 'md52c2be64aff811b072940d9410393a649';






--
-- Database creation
--

CREATE DATABASE cage_sc_db WITH TEMPLATE = template0 OWNER = cage_db_admin;
REVOKE ALL ON DATABASE cage_sc_db FROM PUBLIC;
REVOKE ALL ON DATABASE cage_sc_db FROM cage_db_admin;
GRANT ALL ON DATABASE cage_sc_db TO cage_db_admin;
GRANT CONNECT,TEMPORARY ON DATABASE cage_sc_db TO PUBLIC;
GRANT CONNECT ON DATABASE cage_sc_db TO cage_db_user;
GRANT CONNECT ON DATABASE cage_sc_db TO cage_db_read;
REVOKE ALL ON DATABASE template1 FROM PUBLIC;
REVOKE ALL ON DATABASE template1 FROM postgres;
GRANT ALL ON DATABASE template1 TO postgres;
GRANT CONNECT ON DATABASE template1 TO PUBLIC;


\connect cage_sc_db

SET default_transaction_read_only = off;

--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;


--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET search_path = public, pg_catalog;

--
-- Name: endpoint_data_type; Type: TYPE; Schema: public; Owner: cage_db_admin
--

CREATE TYPE endpoint_data_type AS ENUM (
    'numeric',
    'string',
    'json'
);


ALTER TYPE endpoint_data_type OWNER TO cage_db_admin;

SET search_path = public, pg_catalog;

--
-- Name: insert_json_data_on_view(); Type: FUNCTION; Schema: public; Owner: cage_db_admin
--

CREATE FUNCTION insert_json_data_on_view() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
  declare
    ep_id endpoint_id_map.endpoint_id%TYPE;
    tab_type endpoint_data_type;
    new_meas_id endpoint_json_data.meas_id%TYPE;
  begin

  select
    type, endpoint_id
  from
    endpoint_id_map
  into
    tab_type, ep_id
  where
    endpoint_name = NEW.endpoint_name;

  if found is TRUE then
    case tab_type
      when 'json' then
        if (NEW.value_raw is NULL and NEW.memo is NULL) then
            raise exception 'One of value_raw or memo must be non-null!';
        else
            new_meas_id := nextval('measurement_ids');

      if NEW.memo is not NULL then
        insert into
          measurement_metadata(meas_id, endpoint_id, timestamp, memo)
        values
          (new_meas_id, ep_id, NEW.timestamp, NEW.memo);
      end if;

      if NEW.value_raw is not NULL then
          if NEW.value_cal is not NULL then
            insert into
              endpoint_json_data(endpoint_id, value_raw, value_cal, timestamp, meas_id)
            values
              (ep_id, NEW.value_raw, NEW.value_cal, NEW.timestamp, new_meas_id);
          else
            insert into
              endpoint_json_data(endpoint_id, value_raw, timestamp, meas_id)
            values
              (ep_id, NEW.value_raw, NEW.timestamp, new_meas_id);
          end if;
      elsif NEW.value_cal is not NULL then
        insert into
          endpoint_json_data(endpoint_id, value_cal, timestamp, meas_id)
        values
          (ep_id, NEW.value_cal, NEW.timestamp, new_meas_id);
        end if;
      end if;

    ELSE
      raise exception 'ERROR: endpoint % is of non-json type but insert is on json table.', NEW.endpoint_name;
    end case;

  else
    raise exception 'ERROR: no known endpoint with name %', NEW.endpoint_name;
  end if;

  return NULL;
end; $$;


ALTER FUNCTION public.insert_json_data_on_view() OWNER TO cage_db_admin;

--
-- Name: insert_numeric_data_on_view(); Type: FUNCTION; Schema: public; Owner: cage_db_admin
--

CREATE FUNCTION insert_numeric_data_on_view() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
  declare
    ep_id endpoint_id_map.endpoint_id%TYPE;
    tab_type endpoint_data_type;
    new_meas_id endpoint_numeric_data.meas_id%TYPE;
  begin

  select
    type, endpoint_id
  from
    endpoint_id_map
  into
    tab_type, ep_id
  where
    endpoint_name = NEW.endpoint_name;

  if found is TRUE then
    case tab_type
      when 'numeric' then
        if (NEW.value_raw is NULL and NEW.memo is NULL) then
 	  raise exception 'One of value_raw or memo must be non-null!';
	else
	  new_meas_id := nextval('measurement_ids');	  

	  if NEW.memo is not NULL then
  	    insert into
	      measurement_metadata(meas_id, endpoint_id, timestamp, memo)
	    values
	      (new_meas_id, ep_id, NEW.timestamp, NEW.memo);
	  end if;

          if NEW.value_raw is not NULL then
	    if NEW.value_cal is not NULL then
              insert into
                endpoint_numeric_data(endpoint_id, value_raw, value_cal, timestamp, meas_id)
              values
	        (ep_id, NEW.value_raw, NEW.value_cal, NEW.timestamp, new_meas_id);
            else
              insert into
                endpoint_numeric_data(endpoint_id, value_raw, timestamp, meas_id)
 	      values
	        (ep_id, NEW.value_raw, NEW.timestamp, new_meas_id);
	    end if;
	  elsif NEW.value_cal is not NULL then
	    insert into 
	      endpoint_numeric_data(endpoint_id, value_cal, timestamp, meas_id)
    	    values
 	      (ep_id, NEW.value_cal, NEW.timestamp, new_meas_id);
	  end if;
	end if;

      ELSE
        raise exception 'ERROR: endpoint % is of non-numeric type but insert is on numeric table.', NEW.endpoint_name;
      end case;

    else
      raise exception 'ERROR: no known endpoint with name %', NEW.endpoint_name;
   end if;

   return NULL;
end; $$;


ALTER FUNCTION public.insert_numeric_data_on_view() OWNER TO cage_db_admin;

--
-- Name: insert_string_data_on_view(); Type: FUNCTION; Schema: public; Owner: cage_db_admin
--

CREATE FUNCTION insert_string_data_on_view() RETURNS trigger
    LANGUAGE plpgsql
    AS $$  declare
        ep_id endpoint_id_map.endpoint_id%TYPE;
    tab_type endpoint_data_type;
    new_meas_id endpoint_string_data.meas_id%TYPE;
    backend_row endpoint_string_data%ROWTYPE;
  begin

  select
    type, endpoint_id
  from
    endpoint_id_map
  into
    tab_type, ep_id
  where
    endpoint_name = NEW.endpoint_name;

  if found is TRUE then
    case tab_type
      when 'string' then
        if (NEW.value_raw is NULL and NEW.memo is NULL) then
 	  raise exception 'One of value_raw or memo must be non-null!';
	else
	  new_meas_id := nextval('measurement_ids');
	  backend_row.meas_id := new_meas_id;
	  backend_row.endpoint_id := ep_id;
	  backend_row.timestamp = NEW.timestamp;
	  backend_row.value_raw = NEW.value_raw;
	  backend_row.value_cal = NEW.value_cal;

	  if NEW.memo is not NULL then
  	    insert into
	      measurement_metadata(meas_id, endpoint_id, timestamp, memo)
	    values
	      (new_meas_id, ep_id, NEW.timestamp, NEW.memo);
	  end if;

          insert into
            endpoint_string_data
          values
	    (backend_row.*);

	end if;

      ELSE
        raise exception 'endpoint % is of non-string type but insert is on string table.', NEW.endpoint_name;
      end case;

    else
      raise exception 'no known endpoint with name %', NEW.endpoint_name;
   end if;

   return NEW;
end; $$;


ALTER FUNCTION public.insert_string_data_on_view() OWNER TO cage_db_admin;

SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: endpoint_id_map; Type: TABLE; Schema: public; Owner: cage_db_admin; Tablespace: 
--

CREATE TABLE endpoint_id_map (
    endpoint_id integer NOT NULL,
    endpoint_name character varying(256),
    type endpoint_data_type NOT NULL
);


ALTER TABLE endpoint_id_map OWNER TO cage_db_admin;

--
-- Name: measurement_ids; Type: SEQUENCE; Schema: public; Owner: cage_db_admin
--

CREATE SEQUENCE measurement_ids
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE measurement_ids OWNER TO cage_db_admin;

--
-- Name: endpoint_json_data; Type: TABLE; Schema: public; Owner: cage_db_admin; Tablespace: 
--

CREATE TABLE endpoint_json_data (
    endpoint_id integer NOT NULL,
    "timestamp" timestamp without time zone NOT NULL,
    value_raw jsonb,
    value_cal jsonb,
    meas_id bigint DEFAULT nextval('measurement_ids'::regclass) NOT NULL
);


ALTER TABLE endpoint_json_data OWNER TO cage_db_admin;

--
-- Name: endpoint_numeric_data; Type: TABLE; Schema: public; Owner: cage_db_admin; Tablespace: 
--

CREATE TABLE endpoint_numeric_data (
    endpoint_id integer NOT NULL,
    "timestamp" timestamp without time zone NOT NULL,
    value_raw double precision,
    value_cal double precision,
    meas_id bigint DEFAULT nextval('measurement_ids'::regclass) NOT NULL
);


ALTER TABLE endpoint_numeric_data OWNER TO cage_db_admin;

--
-- Name: endpoint_string_data; Type: TABLE; Schema: public; Owner: cage_db_admin; Tablespace: 
--

CREATE TABLE endpoint_string_data (
    endpoint_id integer NOT NULL,
    "timestamp" timestamp without time zone NOT NULL,
    value_raw text,
    value_cal text,
    meas_id bigint DEFAULT nextval('measurement_ids'::regclass) NOT NULL
);


ALTER TABLE endpoint_string_data OWNER TO cage_db_admin;

--
-- Name: endpoint_id_map_endpoint_id_seq; Type: SEQUENCE; Schema: public; Owner: cage_db_admin
--

CREATE SEQUENCE endpoint_id_map_endpoint_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE endpoint_id_map_endpoint_id_seq OWNER TO cage_db_admin;

--
-- Name: endpoint_id_map_endpoint_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: cage_db_admin
--

ALTER SEQUENCE endpoint_id_map_endpoint_id_seq OWNED BY endpoint_id_map.endpoint_id;

--
-- Name: endpoint_json_data_endpoint_id_seq; Type: SEQUENCE; Schema: public; Owner: cage_db_admin
--

CREATE SEQUENCE endpoint_json_data_endpoint_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE endpoint_json_data_endpoint_id_seq OWNER TO cage_db_admin;

--
-- Name: endpoint_json_data_endpoint_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: cage_db_admin
--

ALTER SEQUENCE endpoint_json_data_endpoint_id_seq OWNED BY endpoint_json_data.endpoint_id;

--
-- Name: endpoint_numeric_data_endpoint_id_seq; Type: SEQUENCE; Schema: public; Owner: cage_db_admin
--

CREATE SEQUENCE endpoint_numeric_data_endpoint_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE endpoint_numeric_data_endpoint_id_seq OWNER TO cage_db_admin;

--
-- Name: endpoint_numeric_data_endpoint_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: cage_db_admin
--

ALTER SEQUENCE endpoint_numeric_data_endpoint_id_seq OWNED BY endpoint_numeric_data.endpoint_id;


--
-- Name: endpoint_string_data_endpoint_id_seq; Type: SEQUENCE; Schema: public; Owner: cage_db_admin
--

CREATE SEQUENCE endpoint_string_data_endpoint_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE endpoint_string_data_endpoint_id_seq OWNER TO cage_db_admin;

--
-- Name: endpoint_string_data_endpoint_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: cage_db_admin
--

ALTER SEQUENCE endpoint_string_data_endpoint_id_seq OWNED BY endpoint_string_data.endpoint_id;

--
-- Name: measurement_metadata; Type: TABLE; Schema: public; Owner: cage_db_admin; Tablespace: 
--

CREATE TABLE measurement_metadata (
    meas_id integer NOT NULL,
    endpoint_id integer NOT NULL,
    "timestamp" timestamp without time zone NOT NULL,
    memo text
);


ALTER TABLE measurement_metadata OWNER TO cage_db_admin;

--
-- Name: json_data; Type: VIEW; Schema: public; Owner: cage_db_admin
--

CREATE VIEW json_data AS
 SELECT endpoint_id_map.endpoint_name,
    n."timestamp",
    n.value_raw,
    n.value_cal,
    measurement_metadata.memo
   FROM ((endpoint_json_data n
     JOIN endpoint_id_map USING (endpoint_id))
     FULL JOIN measurement_metadata measurement_metadata(meas_id, endpoint_id_1, "timestamp", memo) USING (meas_id))
  WHERE ((n.value_raw IS NOT NULL) OR (n.value_cal IS NOT NULL));


ALTER TABLE json_data OWNER TO cage_db_admin;

--
-- Name: measurement_metadata_endpoint_id_seq; Type: SEQUENCE; Schema: public; Owner: cage_db_admin
--

CREATE SEQUENCE measurement_metadata_endpoint_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE measurement_metadata_endpoint_id_seq OWNER TO cage_db_admin;

--
-- Name: measurement_metadata_endpoint_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: cage_db_admin
--

ALTER SEQUENCE measurement_metadata_endpoint_id_seq OWNED BY measurement_metadata.endpoint_id;


--
-- Name: measurement_metadata_meas_id_seq; Type: SEQUENCE; Schema: public; Owner: cage_db_admin
--

CREATE SEQUENCE measurement_metadata_meas_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE measurement_metadata_meas_id_seq OWNER TO cage_db_admin;

--
-- Name: measurement_metadata_meas_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: cage_db_admin
--

ALTER SEQUENCE measurement_metadata_meas_id_seq OWNED BY measurement_metadata.meas_id;


--
-- Name: numeric_data; Type: VIEW; Schema: public; Owner: cage_db_admin
--

CREATE VIEW numeric_data AS
 SELECT endpoint_id_map.endpoint_name,
    n."timestamp",
    n.value_raw,
    n.value_cal,
    measurement_metadata.memo
   FROM ((endpoint_numeric_data n
     JOIN endpoint_id_map USING (endpoint_id))
     FULL JOIN measurement_metadata measurement_metadata(meas_id, endpoint_id_1, "timestamp", memo) USING (meas_id))
  WHERE ((n.value_raw IS NOT NULL) OR (n.value_cal IS NOT NULL));


ALTER TABLE numeric_data OWNER TO cage_db_admin;


--
-- Name: string_data; Type: VIEW; Schema: public; Owner: cage_db_admin
--

CREATE VIEW string_data AS
 SELECT endpoint_id_map.endpoint_name,
    s."timestamp",
    s.value_raw,
    s.value_cal,
    measurement_metadata.memo
   FROM ((endpoint_string_data s
     JOIN endpoint_id_map USING (endpoint_id))
     FULL JOIN measurement_metadata measurement_metadata(meas_id, endpoint_id_1, "timestamp", memo) USING (meas_id))
  WHERE ((s.value_raw IS NOT NULL) OR (s.value_cal IS NOT NULL));


--
-- Name: endpoint_id; Type: DEFAULT; Schema: public; Owner: cage_db_admin
--

ALTER TABLE ONLY endpoint_id_map ALTER COLUMN endpoint_id SET DEFAULT nextval('endpoint_id_map_endpoint_id_seq'::regclass);


--
-- Name: endpoint_id; Type: DEFAULT; Schema: public; Owner: cage_db_admin
--

ALTER TABLE ONLY endpoint_json_data ALTER COLUMN endpoint_id SET DEFAULT nextval('endpoint_json_data_endpoint_id_seq'::regclass);


--
-- Name: endpoint_id; Type: DEFAULT; Schema: public; Owner: cage_db_admin
--

ALTER TABLE ONLY endpoint_numeric_data ALTER COLUMN endpoint_id SET DEFAULT nextval('endpoint_numeric_data_endpoint_id_seq'::regclass);


--
-- Name: endpoint_id; Type: DEFAULT; Schema: public; Owner: cage_db_admin
--

ALTER TABLE ONLY endpoint_string_data ALTER COLUMN endpoint_id SET DEFAULT nextval('endpoint_string_data_endpoint_id_seq'::regclass);


--
-- Name: meas_id; Type: DEFAULT; Schema: public; Owner: cage_db_admin
--

ALTER TABLE ONLY measurement_metadata ALTER COLUMN meas_id SET DEFAULT nextval('measurement_metadata_meas_id_seq'::regclass);


--
-- Name: endpoint_id; Type: DEFAULT; Schema: public; Owner: cage_db_admin
--

ALTER TABLE ONLY measurement_metadata ALTER COLUMN endpoint_id SET DEFAULT nextval('measurement_metadata_endpoint_id_seq'::regclass);


--
-- Name: endpoint_id_map_endpoint_name_key; Type: CONSTRAINT; Schema: public; Owner: cage_db_admin; Tablespace: 
--

ALTER TABLE ONLY endpoint_id_map
    ADD CONSTRAINT endpoint_id_map_endpoint_name_key UNIQUE (endpoint_name);


--
-- Name: endpoint_id_map_pkey; Type: CONSTRAINT; Schema: public; Owner: cage_db_admin; Tablespace: 
--

ALTER TABLE ONLY endpoint_id_map
    ADD CONSTRAINT endpoint_id_map_pkey PRIMARY KEY (endpoint_id);


--
-- Name: endpoint_json_data_pkey; Type: CONSTRAINT; Schema: public; Owner: cage_db_admin; Tablespace: 
--

ALTER TABLE ONLY endpoint_json_data
    ADD CONSTRAINT endpoint_json_data_pkey PRIMARY KEY (endpoint_id, "timestamp");


--
-- Name: endpoint_numeric_data_pkey; Type: CONSTRAINT; Schema: public; Owner: cage_db_admin; Tablespace: 
--

ALTER TABLE ONLY endpoint_numeric_data
    ADD CONSTRAINT endpoint_numeric_data_pkey PRIMARY KEY (endpoint_id, "timestamp");


--
-- Name: endpoint_string_data_pkey; Type: CONSTRAINT; Schema: public; Owner: cage_db_admin; Tablespace: 
--

ALTER TABLE ONLY endpoint_string_data
    ADD CONSTRAINT endpoint_string_data_pkey PRIMARY KEY (endpoint_id, "timestamp");


--
-- Name: measurement_metadata_pkey; Type: CONSTRAINT; Schema: public; Owner: cage_db_admin; Tablespace: 
--

ALTER TABLE ONLY measurement_metadata
    ADD CONSTRAINT measurement_metadata_pkey PRIMARY KEY (meas_id);


--
-- Name: json_data_view_insert; Type: TRIGGER; Schema: public; Owner: cage_db_admin
--

CREATE TRIGGER json_data_view_insert INSTEAD OF INSERT ON json_data FOR EACH ROW EXECUTE PROCEDURE insert_json_data_on_view();


--
-- Name: numeric_data_view_insert; Type: TRIGGER; Schema: public; Owner: cage_db_admin
--

CREATE TRIGGER numeric_data_view_insert INSTEAD OF INSERT ON numeric_data FOR EACH ROW EXECUTE PROCEDURE insert_numeric_data_on_view();


--
-- Name: string_data_view_insert; Type: TRIGGER; Schema: public; Owner: cage_db_admin
--

CREATE TRIGGER string_data_view_insert INSTEAD OF INSERT ON string_data FOR EACH ROW EXECUTE PROCEDURE insert_string_data_on_view();


--
-- Name: endpoint_json_data_endpoint_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: cage_db_admin
--

ALTER TABLE ONLY endpoint_json_data
    ADD CONSTRAINT endpoint_json_data_endpoint_id_fkey FOREIGN KEY (endpoint_id) REFERENCES endpoint_id_map(endpoint_id);


--
-- Name: endpoint_numeric_data_endpoint_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: cage_db_admin
--

ALTER TABLE ONLY endpoint_numeric_data
    ADD CONSTRAINT endpoint_numeric_data_endpoint_id_fkey FOREIGN KEY (endpoint_id) REFERENCES endpoint_id_map(endpoint_id);


--
-- Name: endpoint_string_data_endpoint_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: cage_db_admin
--

ALTER TABLE ONLY endpoint_string_data
    ADD CONSTRAINT endpoint_string_data_endpoint_id_fkey FOREIGN KEY (endpoint_id) REFERENCES endpoint_id_map(endpoint_id);


--
-- Name: measurement_metadata_endpoint_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: cage_db_admin
--

ALTER TABLE ONLY measurement_metadata
    ADD CONSTRAINT measurement_metadata_endpoint_id_fkey FOREIGN KEY (endpoint_id) REFERENCES endpoint_id_map(endpoint_id);


--
-- Name: endpoint_id_map; Type: ACL; Schema: public; Owner: cage_db_admin
--

REVOKE ALL ON TABLE endpoint_id_map FROM PUBLIC;
REVOKE ALL ON TABLE endpoint_id_map FROM cage_db_admin;
GRANT ALL ON TABLE endpoint_id_map TO cage_db_admin;
GRANT SELECT,INSERT,DELETE,UPDATE ON TABLE endpoint_id_map TO cage_db_user;
GRANT SELECT ON TABLE endpoint_id_map TO cage_db_read;


--
-- Name: measurement_ids; Type: ACL; Schema: public; Owner: cage_db_admin
--

REVOKE ALL ON SEQUENCE measurement_ids FROM PUBLIC;
REVOKE ALL ON SEQUENCE measurement_ids FROM cage_db_admin;
GRANT ALL ON SEQUENCE measurement_ids TO cage_db_admin;
GRANT SELECT,USAGE ON SEQUENCE measurement_ids TO cage_db_user;


--
-- Name: endpoint_json_data; Type: ACL; Schema: public; Owner: cage_db_admin
--

REVOKE ALL ON TABLE endpoint_json_data FROM PUBLIC;
REVOKE ALL ON TABLE endpoint_json_data FROM cage_db_admin;
GRANT ALL ON TABLE endpoint_json_data TO cage_db_admin;
GRANT SELECT,INSERT,DELETE ON TABLE endpoint_json_data TO cage_db_user;
GRANT SELECT ON TABLE endpoint_json_data TO cage_db_read;


--
-- Name: endpoint_numeric_data; Type: ACL; Schema: public; Owner: cage_db_admin
--

REVOKE ALL ON TABLE endpoint_numeric_data FROM PUBLIC;
REVOKE ALL ON TABLE endpoint_numeric_data FROM cage_db_admin;
GRANT ALL ON TABLE endpoint_numeric_data TO cage_db_admin;
GRANT SELECT,INSERT,DELETE ON TABLE endpoint_numeric_data TO cage_db_user;
GRANT SELECT ON TABLE endpoint_numeric_data TO cage_db_read;


--
-- Name: endpoint_string_data; Type: ACL; Schema: public; Owner: cage_db_admin
--

REVOKE ALL ON TABLE endpoint_string_data FROM PUBLIC;
REVOKE ALL ON TABLE endpoint_string_data FROM cage_db_admin;
GRANT ALL ON TABLE endpoint_string_data TO cage_db_admin;
GRANT SELECT,INSERT,DELETE ON TABLE endpoint_string_data TO cage_db_user;
GRANT SELECT ON TABLE endpoint_string_data TO cage_db_read;


--
-- Name: endpoint_id_map_endpoint_id_seq; Type: ACL; Schema: public; Owner: cage_db_admin
--

REVOKE ALL ON SEQUENCE endpoint_id_map_endpoint_id_seq FROM PUBLIC;
REVOKE ALL ON SEQUENCE endpoint_id_map_endpoint_id_seq FROM cage_db_admin;
GRANT ALL ON SEQUENCE endpoint_id_map_endpoint_id_seq TO cage_db_admin;
GRANT SELECT,USAGE ON SEQUENCE endpoint_id_map_endpoint_id_seq TO cage_db_user;


--
-- Name: measurement_metadata; Type: ACL; Schema: public; Owner: cage_db_admin
--

REVOKE ALL ON TABLE measurement_metadata FROM PUBLIC;
REVOKE ALL ON TABLE measurement_metadata FROM cage_db_admin;
GRANT ALL ON TABLE measurement_metadata TO cage_db_admin;
GRANT SELECT,INSERT ON TABLE measurement_metadata TO cage_db_user;


--
-- Name: json_data; Type: ACL; Schema: public; Owner: cage_db_admin
--

REVOKE ALL ON TABLE json_data FROM PUBLIC;
REVOKE ALL ON TABLE json_data FROM cage_db_admin;
GRANT ALL ON TABLE json_data TO cage_db_admin;
GRANT SELECT,INSERT ON TABLE json_data TO cage_db_user;
GRANT SELECT ON TABLE json_data TO cage_db_read;


--
-- Name: numeric_data; Type: ACL; Schema: public; Owner: cage_db_admin
--

REVOKE ALL ON TABLE numeric_data FROM PUBLIC;
REVOKE ALL ON TABLE numeric_data FROM cage_db_admin;
GRANT ALL ON TABLE numeric_data TO cage_db_admin;
GRANT SELECT,INSERT ON TABLE numeric_data TO cage_db_user;
GRANT SELECT ON TABLE numeric_data TO cage_db_read;


--
-- Name: string_data; Type: ACL; Schema: public; Owner: cage_db_admin
--

REVOKE ALL ON TABLE string_data FROM PUBLIC;
REVOKE ALL ON TABLE string_data FROM cage_db_admin;
GRANT ALL ON TABLE string_data TO cage_db_admin;
GRANT SELECT,INSERT ON TABLE string_data TO cage_db_user;
GRANT SELECT ON TABLE string_data TO cage_db_read;


--
-- PostgreSQL database dump complete
--

\connect postgres

SET default_transaction_read_only = off;

--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- Name: postgres; Type: COMMENT; Schema: -; Owner: postgres
--

COMMENT ON DATABASE postgres IS 'default administrative connection database';


--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


--
-- Name: adminpack; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS adminpack WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION adminpack; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION adminpack IS 'administrative functions for PostgreSQL';


--
-- Name: public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

\connect template1

SET default_transaction_read_only = off;

--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- Name: template1; Type: COMMENT; Schema: -; Owner: postgres
--

COMMENT ON DATABASE template1 IS 'default template for new databases';


--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


--
-- Name: public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

--
-- PostgreSQL database cluster dump complete
--

