use larql_vindex_spec::{SlicePreset, UNSLICED_PRESET};

#[cfg(feature = "alloc")]
#[test]
fn every_spelling_parses_to_its_preset() {
    for &preset in SlicePreset::ALL {
        for spelling in preset.spellings() {
            assert_eq!(spelling.parse::<SlicePreset>().unwrap(), preset);
            assert_eq!(
                spelling.to_uppercase().parse::<SlicePreset>().unwrap(),
                preset
            );
        }
    }
}

#[test]
fn no_spelling_is_claimed_twice() {
    let mut all: Vec<&str> = SlicePreset::ALL
        .iter()
        .flat_map(|p| p.spellings().iter().copied())
        .collect();
    let n = all.len();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), n);
}

#[cfg(feature = "alloc")]
#[test]
fn unknown_names_are_refused_with_the_known_list() {
    let err = "xyz".parse::<SlicePreset>().unwrap_err().to_string();
    assert!(
        err.contains("xyz") && err.contains("expert-server"),
        "{err}"
    );
}

#[test]
fn the_unsliced_name_is_not_a_slice_preset() {
    assert_eq!(SlicePreset::from_name(UNSLICED_PRESET), None);
}

#[cfg(feature = "alloc")]
#[test]
fn the_unsliced_name_is_refused_by_from_str() {
    assert!(UNSLICED_PRESET.parse::<SlicePreset>().is_err());
}

// The `core`-layer parse: allocation-free, no `FromStr`, no error type.
#[test]
fn from_name_accepts_every_spelling_of_every_preset() {
    for &preset in SlicePreset::ALL {
        for spelling in preset.spellings() {
            assert_eq!(SlicePreset::from_name(spelling), Some(preset));
            assert_eq!(
                SlicePreset::from_name(&spelling.to_uppercase()),
                Some(preset)
            );
        }
    }
}

#[test]
fn from_name_refuses_unknown_and_empty_names() {
    assert_eq!(SlicePreset::from_name("xyz"), None);
    assert_eq!(SlicePreset::from_name(""), None);
    assert_eq!(SlicePreset::from_name("clien"), None);
    assert_eq!(SlicePreset::from_name("client "), None);
}

#[cfg(feature = "alloc")]
#[test]
fn from_name_agrees_with_from_str() {
    let probes = [
        "Client",
        "ATTN",
        "xyz",
        "",
        "expert_server",
        "full",
        "Moe-Server",
    ];
    for &preset in SlicePreset::ALL {
        for spelling in preset.spellings() {
            assert_eq!(
                SlicePreset::from_name(spelling),
                spelling.parse::<SlicePreset>().ok()
            );
        }
    }
    for probe in probes {
        assert_eq!(
            SlicePreset::from_name(probe),
            probe.parse::<SlicePreset>().ok(),
            "{probe}"
        );
    }
}

#[test]
fn display_is_the_canonical_name() {
    assert_eq!(SlicePreset::Attention.to_string(), "attn");
}
