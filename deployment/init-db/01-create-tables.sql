--
-- Create database init tables
--

--
-- Name: chats; Type: TABLE; Schema: public; Owner: gptadmin
--

CREATE TABLE public.chats (
    id integer NOT NULL,
    user_id integer,
    chat_name character varying,
    created_at timestamp without time zone,
    keep_original_files boolean DEFAULT false
);


ALTER TABLE public.chats OWNER TO gptadmin;

--
-- Name: chats_id_seq; Type: SEQUENCE; Schema: public; Owner: gptadmin
--

CREATE SEQUENCE public.chats_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.chats_id_seq OWNER TO gptadmin;

--
-- Name: chats_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: gptadmin
--

ALTER SEQUENCE public.chats_id_seq OWNED BY public.chats.id;


--
-- Name: files; Type: TABLE; Schema: public; Owner: gptadmin
--

CREATE TABLE public.files (
    id integer NOT NULL,
    chat_id integer,
    file_type character varying(10) DEFAULT 'doc'::character varying,
    file_name character varying(255) NOT NULL
);


ALTER TABLE public.files OWNER TO gptadmin;

--
-- Name: file_id_seq; Type: SEQUENCE; Schema: public; Owner: gptadmin
--

CREATE SEQUENCE public.file_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.file_id_seq OWNER TO gptadmin;

--
-- Name: file_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: gptadmin
--

ALTER SEQUENCE public.file_id_seq OWNED BY public.files.id;


--
-- Name: messages; Type: TABLE; Schema: public; Owner: gptadmin
--

CREATE TABLE public.messages (
    id integer NOT NULL,
    chat_id integer,
    sender character varying,
    content text,
    "timestamp" timestamp without time zone
);


ALTER TABLE public.messages OWNER TO gptadmin;

--
-- Name: messages_id_seq; Type: SEQUENCE; Schema: public; Owner: gptadmin
--

CREATE SEQUENCE public.messages_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.messages_id_seq OWNER TO gptadmin;

--
-- Name: messages_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: gptadmin
--

ALTER SEQUENCE public.messages_id_seq OWNED BY public.messages.id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: gptadmin
--

CREATE TABLE public.users (
    id integer NOT NULL,
    username character varying,
    api_key character varying
);


ALTER TABLE public.users OWNER TO gptadmin;

--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: gptadmin
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.users_id_seq OWNER TO gptadmin;

--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: gptadmin
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: chats id; Type: DEFAULT; Schema: public; Owner: gptadmin
--

ALTER TABLE ONLY public.chats ALTER COLUMN id SET DEFAULT nextval('public.chats_id_seq'::regclass);


--
-- Name: files id; Type: DEFAULT; Schema: public; Owner: gptadmin
--

ALTER TABLE ONLY public.files ALTER COLUMN id SET DEFAULT nextval('public.file_id_seq'::regclass);


--
-- Name: messages id; Type: DEFAULT; Schema: public; Owner: gptadmin
--

ALTER TABLE ONLY public.messages ALTER COLUMN id SET DEFAULT nextval('public.messages_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: gptadmin
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Name: chats chats_pkey; Type: CONSTRAINT; Schema: public; Owner: gptadmin
--

ALTER TABLE ONLY public.chats
    ADD CONSTRAINT chats_pkey PRIMARY KEY (id);


--
-- Name: files files_pkey; Type: CONSTRAINT; Schema: public; Owner: gptadmin
--

ALTER TABLE ONLY public.files
    ADD CONSTRAINT files_pkey PRIMARY KEY (id);


--
-- Name: messages messages_pkey; Type: CONSTRAINT; Schema: public; Owner: gptadmin
--

ALTER TABLE ONLY public.messages
    ADD CONSTRAINT messages_pkey PRIMARY KEY (id);


--
-- Name: users users_api_key_key; Type: CONSTRAINT; Schema: public; Owner: gptadmin
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_api_key_key UNIQUE (api_key);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: gptadmin
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: ix_chats_id; Type: INDEX; Schema: public; Owner: gptadmin
--

CREATE INDEX ix_chats_id ON public.chats USING btree (id);


--
-- Name: ix_users_id; Type: INDEX; Schema: public; Owner: gptadmin
--

CREATE INDEX ix_users_id ON public.users USING btree (id);


--
-- Name: ix_users_username; Type: INDEX; Schema: public; Owner: gptadmin
--

CREATE UNIQUE INDEX ix_users_username ON public.users USING btree (username);


--
-- Name: files files_chat_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gptadmin
--

ALTER TABLE ONLY public.files
    ADD CONSTRAINT files_chat_id_fkey FOREIGN KEY (chat_id) REFERENCES public.chats(id);


--
-- Name: messages messages_chat_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gptadmin
--

ALTER TABLE ONLY public.messages
    ADD CONSTRAINT messages_chat_id_fkey FOREIGN KEY (chat_id) REFERENCES public.chats(id);
