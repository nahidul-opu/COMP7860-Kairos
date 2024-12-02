 create table event_table(
    src_id     varchar,
    src_type   varchar,
    edge_type  varchar,
    dst_id     varchar,
    dst_type   varchar,
    hostname   varchar,
    timestamp  bigint,
    data_label varchar
);
 alter table event_table owner to postgres;

 create table nodeid2msg
(
    node_id varchar,
    msg     varchar
);
 alter table nodeid2msg owner to postgres;