//! Semantic versions, only what the updater needs: `1.2.3`, an optional leading `v`, an optional `-prerelease` and `+build` (build is ignored).
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub pre: Option<String>,
}

impl Version {
    pub fn parse(text: &str) -> Option<Version> {
        let t = text.trim();
        let t = t.strip_prefix('v').or_else(|| t.strip_prefix('V')).unwrap_or(t);
        let t = t.split('+').next()?;
        let (core, pre) = match t.split_once('-') {
            Some((c, p)) if !p.is_empty() => (c, Some(p.to_string())),
            Some(_) => return None,
            None => (t, None),
        };
        let mut it = core.split('.');
        let num = |s: Option<&str>| -> Option<u64> {
            let s = s?;
            if s.is_empty() || !s.chars().all(|c| c.is_ascii_digit()) || (s.len() > 1 && s.starts_with('0')) { return None; }
            s.parse().ok()
        };
        let v = Version { major: num(it.next())?, minor: num(it.next())?, patch: num(it.next())?, pre };
        if it.next().is_some() { return None; }
        Some(v)
    }
    pub fn is_prerelease(&self) -> bool {
        self.pre.is_some()
    }
    pub fn text(&self) -> String {
        format!("{}.{}.{}{}", self.major, self.minor, self.patch, self.pre.as_ref().map(|p| format!("-{p}")).unwrap_or_default())
    }
}

fn pre_cmp(a: &str, b: &str) -> Ordering {
    let (mut x, mut y) = (a.split('.'), b.split('.'));
    loop {
        match (x.next(), y.next()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(p), Some(q)) => {
                let (pn, qn) = (p.parse::<u64>().ok(), q.parse::<u64>().ok());
                let o = match (pn, qn) { (Some(a), Some(b)) => a.cmp(&b), (Some(_), None) => Ordering::Less, (None, Some(_)) => Ordering::Greater, (None, None) => p.cmp(q) };
                if o != Ordering::Equal { return o; }
            }
        }
    }
}

impl Ord for Version {
    fn cmp(&self, o: &Self) -> Ordering {
        (self.major, self.minor, self.patch).cmp(&(o.major, o.minor, o.patch)).then_with(|| match (&self.pre, &o.pre) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(a), Some(b)) => pre_cmp(a, b),
        })
    }
}
impl PartialOrd for Version {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
