# Plugin: plugin_database_mysql

MySQL database tool. Gives the LLM read/write access to a local MySQL database.

## Tool provided

`db_query(sql: str) → str` — executes any SQL statement and returns formatted results.

## Tool access

Access to `db_query` is controlled per-model via the `llm_tools` field in `llm-models.json`. Add `"db_query"` to a model's tool list to grant database access.

```
!llm_tools read <model>           show which tools a model can use
!llm_tools write <model> db_query grant database access to a model
```

## Dependencies

```bash
pip install mysql-connector-python>=8.0
```

## Environment variables

```
MYSQL_USER=<username>
MYSQL_PASS=<password>
```

Database name is `mymcp` on `localhost` (hardcoded default; change in `config.py` if needed).

## Enable

```bash
python llmemctl.py enable plugin_database_mysql
```
