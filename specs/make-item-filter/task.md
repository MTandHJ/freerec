# Tasks

## TODO

- [x] 1. Implement CLI parameter wiring.
  - Depends on: `spec.md`, `plan.md`
  - Validate: `freerec make --help` exposes `-mif, --match-item-file`, and parsed args can carry `match_item_file=True`.

- [x] 2. Implement interaction filtering in `AtomicConverter`.
  - Depends on: Task 1
  - Validate: interactions whose raw `ITEM` is absent from `.item` are removed when `match_item_file=True`, and missing `.item` skips with a clear log.

- [x] 3. Add automated tests for the new behavior.
  - Depends on: Tasks 1, 2
  - Validate: tests cover default compatibility, enabled filtering, missing `.item`, custom item column names, and k-core ordering.

- [x] 4. Update user-facing dataset processing documentation.
  - Depends on: Tasks 1, 2
  - Validate: `docs/tutorials/dataset_processing.rst` documents `-mif, --match-item-file`, its timing before k-core, and an example.

- [x] 5. Update the package skill help for `freerec skill --make`.
  - Depends on: Tasks 1, 2
  - Validate: `freerec skill --make` output includes the new option and optional item-file match pipeline step.

- [x] 6. Run final verification.
  - Depends on: Tasks 1-5
  - Validate: targeted tests, full `pytest`, `freerec make --help`, and `freerec skill --make` pass the checks from `validation.md`.

## Blockers

None.

## Notes

- Keep `-mif, --match-item-file` opt-in.
- Do not change output directory naming.
- Do not add special recovery for malformed `.item` files or empty post-filter interactions beyond existing behavior.
- Extra Sphinx HTML build was attempted, but the local environment is missing `sphinxcontrib.mermaid`.
