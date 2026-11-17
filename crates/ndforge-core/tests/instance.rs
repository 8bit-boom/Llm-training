//! One program per settings folder (instance.rs).
use ndforge_core::instance::{acquire, held_by, other_running, Refused};

#[test]
fn a_second_instance_is_refused_and_the_first_one_letting_go_frees_the_folder() {
    let d = tempfile::tempdir().unwrap();
    let dir = d.path().join("settings");
    assert!(!other_running(&dir), "nothing runs yet (the folder is made on the way)");
    let first = acquire(&dir).expect("the first one gets the lock");
    assert_eq!(held_by(&dir), Some(std::process::id()));
    assert_eq!(acquire(&dir).unwrap_err(), Refused::Held(Some(std::process::id())), "the second one is told who holds it");
    assert!(!other_running(&dir), "this program's own hold is not 'another instance'");
    drop(first);
    let again = acquire(&dir).expect("free again once the first has ended");
    assert_eq!(again.dir(), dir.as_path());
}

#[test]
fn another_settings_folder_is_another_instance() {
    let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let _x = acquire(a.path()).unwrap();
    let _y = acquire(b.path()).expect("a portable copy with its own settings folder runs next to it");
}

#[test]
fn a_folder_that_cannot_be_used_is_not_an_instance_that_runs() {
    let d = tempfile::tempdir().unwrap();
    let file = d.path().join("not-a-folder");
    std::fs::write(&file, "x").unwrap();
    assert!(matches!(acquire(&file.join("inside")), Err(Refused::Unusable(_))));
    assert!(!other_running(&file.join("inside")));
}

#[test]
fn a_stale_pid_file_does_not_count_when_nobody_holds_the_lock() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("instance.pid"), "99999999").unwrap();
    assert!(!other_running(d.path()));
    let _i = acquire(d.path()).unwrap();
    assert_eq!(held_by(d.path()), Some(std::process::id()), "the new holder writes its own number");
}
