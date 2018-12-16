CREATE TABLE transactions (
 currency varchar(3) not null,
 amount bigint not null,
 state varchar(25) not null,
 created_date timestamp without time zone not null,
 merchant_category varchar(100),
 merchant_country varchar(3),
 entry_method varchar(4) not null,
 user_id uuid not null,
 type varchar(20) not null,
 source varchar(20) not null,
 id uuid primary key 
); 

CREATE TABLE users (
 id uuid primary key,
 has_email boolean not null,
 phone_country varchar(300),
 is_fraudster boolean not null,
 terms_version date,
 created_date timestamp without time zone not null,
 state varchar(25) not null,
 country varchar(2),
 birth_year integer,
 kyc varchar(20),
 failed_sign_in_attempts integer
);

CREATE TABLE fx_rates (
 ts timestamp without time zone,
 base_ccy varchar(3),
 ccy varchar(10),
 rate double precision,
 PRIMARY KEY(ts, base_ccy, ccy)
);

CREATE TABLE currency_details (
 ccy varchar(10) primary key,
 iso_code integer,
 exponent integer,
 is_crypto boolean not null
);

