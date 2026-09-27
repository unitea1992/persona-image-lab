# Generation History

Completed local generations are shown newest first in the recent-generation
gallery. A hidden dataset keeps prompt/reference/size/step/seed/runtime values
for restore logic without taking space in the everyday interface.

Every successful generation refreshes both views in the submitting browser.
Reloading the page reads history from disk, including after a server restart.
Other browser sessions refresh when reloaded or after their next generation;
this release does not push live history updates between sessions.

Select a gallery image to restore its original user prompt, references,
dimensions, steps, seed, and result preview. Selection does not generate an
image. "Use result as extra reference" makes the selected result the manual
reference input for a subsequent edit.

To remove a generation, select it, choose Delete, then confirm Delete in the
second-step confirmation UI. The app removes the generation PNG and JSON record.
Saved reference copies are removed only when no remaining generation uses them.
Deletion cannot be undone.

## Storage

Each generation writes `<id>.png` and `<id>.json` in `outputs/`. JSON is written
atomically after the PNG. Only a complete record with a present output image is
listed. Missing or malformed records are skipped with a server log warning.
Reference originals are stored under `outputs/references/`, addressed by SHA-256
to deduplicate identical uploads. Treat them as private data, including any
embedded camera metadata. Generated PNGs do not copy that reference metadata.

Version 1 records contain:

- `schema_version`, app version, `id`, and UTC `created_at`.
- Model identifier, pinned revision, and relevant library versions.
- Effective model prompt, width, height, steps, seed, guidance, and KV-cache settings.
- The original `user_prompt` when generated through the UI, so restoring a
  generation puts only the user's text back into the prompt field instead of
  exposing internal persona instructions.
- Prompt-enhancer metadata when used, including PE-T2I/PE-I2I model identity,
  pinned revision, rewritten prompt, enhancer seed, and enhancer runtime.
- Optional local persona identifier/name context when a persona preset was used.
- `elapsed_seconds`, `peak_allocated_gib`, image mode, and alpha extrema.
- Reference filename, content hash, and relative persistent path.

Timing excludes model loading and disk persistence; peak allocated GPU memory
is not total system memory usage. Same seeds/settings help reproduce results
but do not promise bitwise identity across different libraries or hardware.

Existing prototype PNG/JSON pairs are shown without rewriting them. Their
reference records may have only names and hashes, not reusable copies. Selecting
such an edit warns about missing references; re-upload the originals before
regenerating it. Standalone PNG files without generation records are not listed.

## Retention and Backup

No automatic deletion is enabled. Back up the entire `outputs/` directory,
including `references/`. Backups and `.gitignore` protect against accidental
commits, not against deletion in the app or local users with filesystem access.
