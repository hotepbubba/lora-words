# LoRA Trigger Word Sync

This repository contains a lightweight Python utility for collecting trigger
words associated with local LoRA checkpoints. The script can read metadata from
`.safetensors` files and/or query the public [CivitAI API](https://civitai.com/api-docs)
using each file's SHA256 hash. Results are stored in a JSON database so the
lookups only have to happen once per model.

## Getting started

1. Install the required dependencies (Python 3.9+ is recommended):

   ```bash
   pip install -r requirements.txt
   ```


   On Windows you can use the provided helper script, which creates a virtual
   environment (defaults to `.venv`) and installs the requirements into it:

   ```bat
   install.bat
   ```

   Provide a custom path (e.g. on a different drive) via `install.bat path\to\venv`,
   and use the same location with `run.bat --venv path\to\venv` when launching the
   sync script.

=======

2. Run the synchronisation script, pointing it at the directory that contains
your LoRA models:

   ```bash
   python lora_trigger_sync.py /path/to/lora/directory --database trigger_words.json
   ```


   On Windows, the companion `run.bat` activates the virtual environment created
   by `install.bat` and forwards any additional arguments to the Python script:

   ```bat
   run.bat "D:\\Models" --database trigger_words.json --force
   ```

=======

   Provide a `--api-token` if you have a CivitAI personal access token, and set
   `--force` to refresh entries that already exist in the JSON database.

The resulting `trigger_words.json` file will map each LoRA's SHA256 hash to the
list of trigger words discovered for that file, the files where the hash was
seen, and any metadata that the script could recover.

## Requirements

- [requests](https://pypi.org/project/requests/) – used to communicate with the
  CivitAI API.
- [safetensors](https://pypi.org/project/safetensors/) – optional but recommended;
  enables extracting trigger words directly from LoRA metadata without needing a
  network request.

If `safetensors` is not installed the script will still run, but it will only be
able to query the CivitAI API for trigger words.
