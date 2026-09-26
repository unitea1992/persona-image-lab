# Releases and Tags

Persona Image Lab is a public fork. Repository visibility is not part of the
release workflow, and no automation publishes containers or model weights.

## Versioning

Use semantic versions in `VERSION` and matching annotated Git tags prefixed
with `v`. Tag only a reviewed commit with passing checks, and do not move or
overwrite an existing tag.

Before tagging, update `CHANGELOG.md`, run CPU/container tests, perform a GB10
smoke test when inference behavior changed, and record the tested hardware and
software revisions.

Suggested GitHub topics: `dgx-spark`, `nvidia`, `gb10`, `qwen-image`,
`gradio`, `local-ai`, `image-generation`, `image-editing`,
`character-consistency`, `python`.

## Public release checklist

- [ ] Fresh-clone Docker setup works on a DGX Spark.
- [ ] CPU/container tests and relevant GB10 generation checks pass.
- [ ] Repository history and current tree contain no secrets, private persona
  files, personal images, machine-specific paths, or generated outputs.
- [ ] `PERSONA_DATA_DIR` remains a runtime-only setting and persona data is
  mounted read-only.
- [ ] README, security notes, issue templates, topics, description, and license
  remain accurate.
- [ ] Dependency/security advisories have been reviewed.
- [ ] Screenshots, if added, use synthetic or explicitly public-safe content.
- [ ] Release notes distinguish measured DGX Spark results from general claims.

Do not publish private persona packs, prompt canon, local paths, credentials,
model weights, or generation history as release assets.
