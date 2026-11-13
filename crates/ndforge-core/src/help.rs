//! The instruction texts shown in the help panel of every page (Markdown files in `help/`, compiled in).
use pulldown_cmark::{html, Options, Parser};

const PAGES: [(&str, &str); 10] = [
    ("welcome", include_str!("../help/welcome.md")),
    ("data", include_str!("../help/data.md")),
    ("dataset", include_str!("../help/dataset.md")),
    ("train", include_str!("../help/train.md")),
    ("test", include_str!("../help/test.md")),
    ("use", include_str!("../help/use.md")),
    ("reshape", include_str!("../help/reshape.md")),
    ("setup", include_str!("../help/setup.md")),
    ("guide", include_str!("../help/guide.md")),
    ("troubleshooting", include_str!("../help/troubleshooting.md")),
];

pub fn help_markdown(key: &str) -> String {
    PAGES.iter().find(|(k, _)| *k == key).map(|(_, t)| t.to_string())
        .unwrap_or_else(|| format!("# Help\n\nNo help text found for '{key}'."))
}

pub fn help_html(key: &str) -> String {
    let md = help_markdown(key);
    let mut out = String::new();
    html::push_html(&mut out, Parser::new_ext(&md, Options::ENABLE_STRIKETHROUGH));
    out
}
