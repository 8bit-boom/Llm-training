"""Builds the sample training bundle used by tests (and committed as tests/fixtures/sample.ndtrain for nd-world's tests)."""
from pathlib import Path

from ndforge.ingest.bundle import write_bundle

WORLD = {"id": 1, "name": "Ashfall Reach", "slug": "ashfall-reach", "description": "A sample world.",
         "templates": [{"id": 1, "slug": "npc", "name": "NPC", "kind": "character", "fields": [{"id": "rank", "label": "Rank"}]}]}


def entities(variant: int = 1) -> list[dict]:
    anders = {"id": 1, "kind": "character", "subtype": "player character", "name": "Anders de Vallière",
              "aliases": ["Anders"], "folder": "Heroes", "tags": ["hunter", "fire"],
              "summary": "A Widow-Bound Hunter.", "body": "# Anders\nHunts colossal beasts. [gmonly]He fears fire.[/gmonly]",
              "template": "npc", "custom_fields": {"rank": "Captain"}, "roleplay_personality": "gruff, terse", "visible_to_players": True,
              "group_ids": [], "rag_priority": False, "updated_at": "2026-11-01T10:00:00"}
    dock = {"id": 2, "kind": "location", "subtype": "city", "name": "Dockside", "aliases": [], "folder": "", "tags": [],
            "summary": "A harbour district.", "body": "Smugglers meet here.", "template": None, "custom_fields": {}, "roleplay_personality": "",
            "visible_to_players": True, "group_ids": [], "rag_priority": False, "updated_at": "2026-11-01T10:00:00"}
    out = [anders, dock]
    if variant == 2:  # a later export: one changed, one new, one removed (Dockside)
        anders = dict(anders, body=anders["body"] + "\nHe carries the Grandcouteau.", updated_at="2026-11-05T10:00:00")
        out = [anders, {"id": 3, "kind": "organization", "subtype": "guild", "name": "The Greyfather Pack", "aliases": [],
                        "folder": "", "tags": [], "summary": "Beast hunters.", "body": "", "template": None, "custom_fields": {},
                        "roleplay_personality": "", "visible_to_players": False, "group_ids": [], "rag_priority": False,
                        "updated_at": "2026-11-05T10:00:00"}]
    return out


def build(path: Path, variant: int = 1, world: dict | None = None) -> Path:
    members = {
        "entities.jsonl": entities(variant),
        "relations.jsonl": [{"id": 1, "source_id": 1, "source_name": "Anders de Vallière", "target_id": 2,
                             "target_name": "Dockside", "relation": "located_in"}],
        "rules.md": "# Rules\nRoll a dice pool of d6.",
        "sessions.jsonl": [{"id": 1, "session_num": 1, "session_date": "2026-10-01", "title": "Arrival",
                            "summary": "The party reached Dockside.", "npcs": [], "loot": []}],
        "facts.jsonl": [{"id": 1, "game_session_id": 1, "content": "The duke is the cult leader.",
                         "visible_to_players": False, "tags": ""}],
        "tables.jsonl": [{"id": 1, "name": "Dock rumours", "category": "general", "description": "",
                          "entries": [{"label": "A ship vanished", "weight": 1}]}],
    }
    return write_bundle(path, world=world or WORLD, members=members, nd_build="sample",
                        created="2026-11-01T12:00:00Z" if variant == 1 else "2026-11-05T12:00:00Z")


if __name__ == "__main__":  # python tests/sample_bundle.py -> regenerate the committed fixture
    print(build(Path(__file__).parent / "fixtures" / "sample.ndtrain"))
