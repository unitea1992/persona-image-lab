# Upstream Policy

Persona Image Lab remains a GitHub fork of
[`joeynyc/spark-image-lab`](https://github.com/joeynyc/spark-image-lab) for
attribution and change discovery. The repository name and product identity are
intentionally different; renaming a fork does not break its GitHub fork
relationship or the local `upstream` remote.

Upstream changes are **not merged automatically**. Persona Image Lab now has a
different UI, command name, data boundary, and private-persona workflow, so a
blind merge or GitHub "Sync fork" can reintroduce upstream branding or undo
local security assumptions.

## Recommended update flow

1. Ensure the upstream remote exists and cannot be pushed to accidentally:

   ```bash
   git remote add upstream https://github.com/joeynyc/spark-image-lab.git
   git remote set-url --push upstream DISABLED
   ```

   If `upstream` already exists, only the second command is needed.

2. Fetch the upstream remote without modifying `main`:

   ```bash
   git fetch upstream --prune
   git log --oneline main..upstream/main
   ```

3. Review each new upstream change for relevance. Prioritize fixes to the
   Qwen-Image-2.1 inference path, DGX Spark / GB10 compatibility, pinned
   dependencies, security, and generation-history correctness.
4. Port relevant changes on a Persona Image Lab branch. Prefer a selective
   cherry-pick when the commit applies cleanly; otherwise reimplement the same
   fix against the current Persona code rather than resolving a broad merge.
5. Run the normal tests and, for inference/runtime changes, validate on a GB10
   system before updating `main`.

Do not use automatic fork synchronization as the normal update mechanism.

## Naming boundary

The upstream product name is allowed only where it is needed for attribution or
to identify the upstream repository. Runtime names, commands, environment
variables, Docker resources, UI text, and user-facing setup instructions use
Persona Image Lab naming.

`scripts/check_branding.py` enforces this boundary in CI. When a new intentional
upstream reference is required, update the allowlist narrowly instead of
disabling the check.
