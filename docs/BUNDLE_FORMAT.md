# nd-world Training Bundle, format version 1 (`.ndtrain`)

The contract between nd-world (producer) and ND Lore Forge (consumer). Both repositories keep a copy of this file and the
sample bundle `tests/fixtures/sample.ndtrain`; a change to the format bumps `version`.

A bundle holds the text of **one world** for **GM-only** training. It is a **zip** file. Every member is UTF-8. `*.json` is one
JSON document; `*.jsonl` is one JSON object per line (no blank lines). Text only: no images, audio, video, accounts or tokens.

## Zip rules (so two implementations produce identical bytes)

- Members are written in the order of the table below (alphabetical within a group), with fixed timestamp `1980-01-01 00:00:00`,
  Unix mode `0o644`, and `ZIP_DEFLATED` level 6.
- JSON is written with `ensure_ascii=False`, keys in the order shown below, `separators=(",", ":")` for JSONL lines and
  `indent=2` for `.json` documents, `\n` line endings, and a final newline.
- Records inside a `.jsonl` file are ordered by `id` ascending unless stated otherwise.

## Members

| Member | Required | Contents |
|---|---|---|
| `manifest.json` | yes | See below. Written last in content terms (it hashes the others) but stored first in the zip. |
| `world.json` | yes | `{"id", "name", "slug", "description", "kinds": [...], "templates": [...], "calendar": {...}, "rulesets": [...]}` |
| `entities.jsonl` | yes | One entity per line (below). |
| `entity_notes.jsonl` | no | `{"id", "entity_id", "content", "visible_to_players", "updated_at"}` |
| `relations.jsonl` | no | `{"id", "source_id", "source_name", "target_id", "target_name", "relation"}`; untyped links use `relation: "related"`. |
| `rules.md` | no | The world's effective rules text (Markdown); the core-rules fallback when the world has none. |
| `rules_sheets.jsonl` | no | `{"id", "template", "rules_md"}` for sheet templates that carry rules text. |
| `sessions.jsonl` | no | `{"id", "session_num", "session_date", "title", "summary", "npcs": [{"entity_id","name"}], "loot": [...]}` |
| `facts.jsonl` | no | `{"id", "game_session_id", "content", "visible_to_players", "tags"}` |
| `quests.jsonl` | no | `{"id", "title", "status", "category", "summary", "body", "linked_entities": [...], "parent_id"}` |
| `tables.jsonl` | no | `{"id", "name", "category", "description", "entries": [{"label", "weight"}]}` |
| `ai_instructions.jsonl` | no | Enabled standing AI instructions: `{"id", "title", "content", "applies_to_players"}` |
| `characters.jsonl` | no | Player characters: `{"id", "name", "race", "char_class", "level", "backstory", "notes", "equipment", "feats"}`; no owner or account fields. |

### Entity record

```json
{"id": 12, "kind": "character", "subtype": "NPC", "name": "Anders de Vallière", "aliases": ["Anders"],
 "folder": "NPCs", "tags": ["hunter"], "summary": "...", "body": "markdown, access tags kept raw",
 "custom_fields": {"rank": "Captain"}, "roleplay_personality": "gruff, ...", "visible_to_players": true,
 "group_ids": [], "rag_priority": false, "updated_at": "2026-11-02T10:00:00"}
```

`body`, `summary`, `roleplay_personality`, notes and facts keep nd-world's access tags **raw** (`[gmonly]…[/gmonly]`,
`[for=IDs]…[/for]`, `[reveal=DATE]…[/reveal]`). The consumer decides what to do with them (ND Lore Forge trains a GM model and
turns them into labelled text). `custom_fields` uses readable labels, resolved through `world.json` templates.

### manifest.json

```json
{"format": "nd-training-bundle", "version": 1, "created": "2026-11-07T12:00:00Z", "nd_build": "…",
 "world": {"id": 1, "name": "…", "slug": "…", "description": "…"},
 "files": {"entities.jsonl": {"sha256": "…", "bytes": 12345, "records": 312}, "world.json": {"sha256": "…", "bytes": 800}}}
```

`records` is present for `.jsonl` members. `files` lists every other member (not the manifest itself).

## Never included

Users, e-mail addresses, passwords, tokens, 2FA data, images/audio/video, map and schematic geometry, dice history, live and
combat state, **character journals and goals**, **GM-to-player private notes**.

## Consumer rules

1. Reject anything whose `format` is not `nd-training-bundle` or whose `version` is higher than supported.
2. Verify every listed member's SHA-256 and size; reject on mismatch or on an unlisted member.
3. Ignore unknown members and unknown keys (forward compatibility); treat missing optional members as empty.
