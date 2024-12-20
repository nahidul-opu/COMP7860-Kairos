 create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial
);
 alter table event_table owner to postgres;
 create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

 create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
 alter table file_node_table owner to postgres;

 create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
 alter table netflow_node_table owner to postgres;

 create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
 alter table subject_node_table owner to postgres;

 create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
 alter table node2id owner to postgres;
 create unique index node2id_hash_id_uindex on node2id (hash_id);