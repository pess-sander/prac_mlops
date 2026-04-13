--
-- PostgreSQL database dump
--

\restrict mqHwqxdNxzScqPT7cJFshE6TUxeg5K3sijA4yQJRWKufJfoMppGDa326KxqKaK8

-- Dumped from database version 16.13 (Ubuntu 16.13-0ubuntu0.24.04.1)
-- Dumped by pg_dump version 16.13 (Ubuntu 16.13-0ubuntu0.24.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: batches; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.batches (
    batch_id integer NOT NULL,
    ingestion_time timestamp without time zone,
    rows_count integer,
    missing_ratio double precision,
    status text,
    message text,
    source_path text
);


--
-- Name: data_quality; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.data_quality (
    batch_id integer,
    column_name text,
    metric_name text,
    metric_value double precision,
    created_at timestamp without time zone,
    metric_text text
);


--
-- Name: raw_data; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.raw_data (
    row_id integer NOT NULL,
    id text,
    name text,
    artists text,
    release_date text,
    year integer,
    popularity integer,
    duration_ms double precision,
    explicit boolean,
    key integer,
    mode integer,
    acousticness double precision,
    danceability double precision,
    energy double precision,
    instrumentalness double precision,
    liveness double precision,
    loudness double precision,
    speechiness double precision,
    tempo double precision,
    valence double precision,
    batch_id integer NOT NULL,
    ingestion_time timestamp without time zone NOT NULL
);


--
-- Name: raw_data_row_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.raw_data_row_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: raw_data_row_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.raw_data_row_id_seq OWNED BY public.raw_data.row_id;


--
-- Name: raw_data row_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.raw_data ALTER COLUMN row_id SET DEFAULT nextval('public.raw_data_row_id_seq'::regclass);


--
-- Name: batches batches_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.batches
    ADD CONSTRAINT batches_pkey PRIMARY KEY (batch_id);


--
-- Name: raw_data raw_data_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.raw_data
    ADD CONSTRAINT raw_data_pkey PRIMARY KEY (row_id);


--
-- PostgreSQL database dump complete
--

\unrestrict mqHwqxdNxzScqPT7cJFshE6TUxeg5K3sijA4yQJRWKufJfoMppGDa326KxqKaK8

