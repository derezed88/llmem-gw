# Plugin: plugin_storage_googledrive

Google Drive CRUD tool. Gives the LLM access to files in an authorized Drive folder.

## Tool provided

`google_drive(operation, file_id, file_name, content, folder_id) → str`

Operations: `list`, `create`, `read`, `append`, `delete`

## Tool access

Access to `google_drive` is controlled per-model via the `llm_tools` field in `llm-models.json`. Add `"google_drive"` to a model's tool list to grant Drive access.

```
!llm_tools read <model>               show which tools a model can use
!llm_tools write <model> google_drive  grant Drive access to a model
```

## Dependencies

```bash
pip install google-auth google-auth-oauthlib google-api-python-client
```

## Environment variables

```
FOLDER_ID=<Google Drive folder ID>
```

## Configuration files

- `credentials.json` — OAuth2 credentials (download from Google Cloud Console)
- `token.json` — auto-generated after first auth (do not commit)

## First-time setup

1. Create a project in Google Cloud Console
2. Enable the Google Drive API
3. Create OAuth2 credentials (Desktop app)
4. Download as `credentials.json` into the llmem-gw directory
5. Run the server — it will open a browser for authorization on first use
6. `token.json` is created automatically

## Enable

```bash
python llmemctl.py enable plugin_storage_googledrive
```
