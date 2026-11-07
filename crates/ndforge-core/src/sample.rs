//! A small made-up world as a training bundle: used by the tests and to regenerate `tests/fixtures/sample.ndtrain`
//! (`cargo run -p ndforge-core --example make_sample`).
use std::path::Path;

use serde_json::{json, Value};

use crate::bundle::{write_bundle, Member};
use crate::error::Result;

pub fn world() -> Value {
    json!({"id": 1, "name": "Ashfall Reach", "slug": "ashfall-reach", "description": "A sample world.",
           "templates": [{"id": 1, "slug": "npc", "name": "NPC", "kind": "character", "fields": [{"id": "rank", "label": "Rank"}]}]})
}

fn entity(id: i64, kind: &str, name: &str, extra: Value) -> Value {
    let mut e = json!({"id": id, "kind": kind, "subtype": "", "name": name, "aliases": [], "folder": "", "tags": [], "summary": "", "body": "",
        "template": null, "custom_fields": {}, "roleplay_personality": "", "visible_to_players": true, "group_ids": [], "rag_priority": false,
        "updated_at": "2026-11-01T10:00:00"});
    if let (Some(m), Some(x)) = (e.as_object_mut(), extra.as_object()) {
        for (k, v) in x {
            m.insert(k.clone(), v.clone());
        }
    }
    e
}

/// `variant` 1 = the first export; 2 = a later one (one entity changed, one new, one removed).
pub fn entities(variant: u8) -> Vec<Value> {
    let mut anders = entity(1, "character", "Anders de Vallière", json!({"subtype": "player character", "aliases": ["Anders"], "folder": "Heroes",
        "tags": ["hunter", "fire"], "summary": "A Widow-Bound Hunter.", "body": "# Anders\nHunts colossal beasts. [gmonly]He fears fire.[/gmonly]",
        "template": "npc", "custom_fields": {"rank": "Captain"}, "roleplay_personality": "gruff, terse"}));
    if variant == 2 {
        anders["body"] = json!("# Anders\nHunts colossal beasts. [gmonly]He fears fire.[/gmonly]\nHe carries the Grandcouteau.");
        anders["updated_at"] = json!("2026-11-05T10:00:00");
        return vec![anders, entity(3, "organization", "The Greyfather Pack", json!({"subtype": "guild", "summary": "Beast hunters.", "visible_to_players": false,
            "updated_at": "2026-11-05T10:00:00"}))];
    }
    vec![anders, entity(2, "location", "Dockside", json!({"subtype": "city", "summary": "A harbour district.", "body": "Smugglers meet here."}))]
}

pub fn build(path: &Path, variant: u8, world: Option<Value>) -> Result<()> {
    let members = vec![
        ("entities.jsonl", Member::Records(entities(variant))),
        ("relations.jsonl", Member::Records(vec![json!({"id": 1, "source_id": 1, "source_name": "Anders de Vallière", "target_id": 2,
            "target_name": "Dockside", "relation": "located_in"})])),
        ("rules.md", Member::Text("# Rules\nRoll a dice pool of d6.".into())),
        ("sessions.jsonl", Member::Records(vec![json!({"id": 1, "session_num": 1, "session_date": "2026-10-01", "title": "Arrival",
            "summary": "The party reached Dockside.", "npcs": [], "loot": []})])),
        ("facts.jsonl", Member::Records(vec![json!({"id": 1, "game_session_id": 1, "content": "The duke is the cult leader.",
            "visible_to_players": false, "tags": ""})])),
        ("tables.jsonl", Member::Records(vec![json!({"id": 1, "name": "Dock rumours", "category": "general", "description": "",
            "entries": [{"label": "A ship vanished", "weight": 1}]})])),
    ];
    let created = if variant == 1 { "2026-11-01T12:00:00Z" } else { "2026-11-05T12:00:00Z" };
    write_bundle(path, &world.unwrap_or_else(self::world), members, "sample", created)
}
