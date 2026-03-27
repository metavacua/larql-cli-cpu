use std::path::Path;

/// Serialization format for graph files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Pretty-printed JSON (.larql.json)
    Json,
    /// MessagePack binary (.larql.bin)
    #[cfg(feature = "msgpack")]
    MessagePack,
}

impl Format {
    /// Detect format from file extension.
    /// Returns None if the extension is unrecognised.
    pub fn from_path(path: impl AsRef<Path>) -> Option<Self> {
        let path = path.as_ref();
        let name = path.file_name()?.to_str()?;

        if name.ends_with(".larql.json") || name.ends_with(".json") {
            return Some(Self::Json);
        }

        #[cfg(feature = "msgpack")]
        if name.ends_with(".larql.bin") || name.ends_with(".bin") || name.ends_with(".msgpack") {
            return Some(Self::MessagePack);
        }

        None
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Json => ".larql.json",
            #[cfg(feature = "msgpack")]
            Self::MessagePack => ".larql.bin",
        }
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json => write!(f, "json"),
            #[cfg(feature = "msgpack")]
            Self::MessagePack => write!(f, "msgpack"),
        }
    }
}
