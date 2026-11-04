use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub root_path: Utf8PathBuf,
    pub root_hash: u64,
    pub deps_hashes: FxHashMap<Utf8PathBuf, u64>,
    pub reflection: ShaderReflection,
}

static MEMORY_CACHE: LazyLock<RwLock<FxHashMap<String, CacheEntry>>> =
    LazyLock::new(|| RwLock::new(FxHashMap::default()));

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedMember {
    pub name: String,
    pub offset: u32,
    pub size: u32,
    pub kind: raw::MemberKind,
    pub scalar_type: raw::ScalarKind,
    pub row_count: u8,
    pub col_count: u8,
    pub array_count: u32,
    pub parent_idx: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResource {
    pub name: String,
    pub kind: raw::ResourceKind,
    pub set: u32,
    pub binding: u32,
    pub total_size: u32,
    pub members: Vec<CachedMember>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderReflection {
    pub spirv_bytes: Vec<u32>,
    pub resources: Vec<CachedResource>,
    pub dependencies: Vec<String>,
}

impl ShaderReflection {
    pub fn from_compile_result(compile_result: &SafeCompileResult) -> Self {
        let resources = compile_result
            .reflections()
            .map(|res| CachedResource {
                name: res.name().to_string(),
                kind: res.kind(),
                set: res.set(),
                binding: res.binding(),
                total_size: res.total_size(),
                members: res
                    .members()
                    .map(|mem| CachedMember {
                        name: mem.name().to_string(),
                        offset: mem.offset(),
                        size: mem.size(),
                        kind: mem.kind(),
                        scalar_type: mem.scalar_type(),
                        row_count: mem.row_count(),
                        col_count: mem.col_count(),
                        array_count: mem.array_count(),
                        parent_idx: mem.parent_index(),
                    })
                    .collect(),
            })
            .collect();

        let dependencies = compile_result
            .dependencies()
            .map(|dep| dep.to_string())
            .collect();

        Self {
            spirv_bytes: compile_result.spirv_u32().to_vec(),
            resources,
            dependencies,
        }
    }

    pub fn push_constant(&self) -> Option<&CachedResource> {
        self.resources
            .iter()
            .find(|r| r.kind == raw::ResourceKind::PushConstant)
    }

    pub fn find_member(&self, name: &str) -> Option<(usize, &CachedMember)> {
        let pc = self.push_constant()?;
        pc.members.iter().enumerate().find(|(_, m)| m.name == name)
    }

    pub fn child_members(&self, parent_idx: i32) -> impl Iterator<Item = &CachedMember> {
        let pc = self.push_constant();
        pc.into_iter()
            .flat_map(|r| r.members.iter())
            .filter(move |m| m.parent_idx == parent_idx)
    }
}

pub struct ShaderCache;

impl ShaderCache {
    pub fn hash_file(path: &Utf8Path) -> Option<u64> {
        let bytes = std::fs::read(path).ok()?;
        let mut hasher = FxHasher::default();
        hasher.write(&bytes);
        Some(hasher.finish())
    }

    fn is_cache_valid(entry: &CacheEntry) -> bool {
        // 1. Root shader file
        match Self::hash_file(&entry.root_path) {
            Some(h) if h == entry.root_hash => {}
            _ => return false,
        }

        // 2. Slang dependencies(#include)
        for (dep_path, &expected_hash) in &entry.deps_hashes {
            match Self::hash_file(dep_path) {
                Some(h) if h == expected_hash => continue,
                _ => return false,
            }
        }

        true
    }

    pub fn get_cache_dir() -> Utf8PathBuf {
        let manifest_dir_str = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
        let manifest_dir = Utf8PathBuf::from(manifest_dir_str);

        let workspace_root = Self::find_workspace_root(&manifest_dir);
        workspace_root.join("target").join("gpgpru_cache")
    }

    fn find_workspace_root(start_dir: &Utf8Path) -> Utf8PathBuf {
        let mut current = start_dir.to_path_buf();
        let mut highest_root = start_dir.to_path_buf();

        let start_cargo_toml = start_dir.join("Cargo.toml");
        if start_cargo_toml.exists() {
            if let Ok(content) = std::fs::read_to_string(&start_cargo_toml) {
                if content.contains("[workspace]") {
                    return start_dir.to_path_buf();
                }
            }
        }

        while let Some(parent) = current.parent().map(|p| p.to_path_buf()) {
            let cargo_toml = parent.join("Cargo.toml");
            if cargo_toml.exists() {
                highest_root = parent.clone();

                if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                    if content.contains("[workspace]") {
                        return parent;
                    }
                }
            }
            current = parent;
        }

        highest_root
    }

    pub fn flatten_workspace_path(slang_path: &Utf8Path) -> String {
        let manifest_dir_str = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
        let manifest_dir = Utf8PathBuf::from(manifest_dir_str);
        let workspace_root = Self::find_workspace_root(&manifest_dir);

        let rel_path = slang_path
            .strip_prefix(&workspace_root)
            .unwrap_or(slang_path);

        rel_path.as_str().replace('/', "_").replace('\\', "_")
    }

    fn get_cache_file_path(hashed_key: &str) -> Utf8PathBuf {
        Self::get_cache_dir().join(format!("{hashed_key}.json"))
    }

    fn try_load_disk_cache(cache_file_path: &Utf8Path) -> Option<CacheEntry> {
        let content = std::fs::read_to_string(cache_file_path).ok()?;
        let entry: CacheEntry = serde_json::from_str(&content).ok()?;

        if Self::is_cache_valid(&entry) {
            Some(entry)
        } else {
            None
        }
    }

    fn save_disk_cache(
        cache_file_path: &Utf8Path,
        slang_path: &Utf8Path,
        live_result: &SafeCompileResult,
        reflection: &ShaderReflection,
    ) -> CacheEntry {
        let root_hash = Self::hash_file(slang_path).unwrap_or(0);

        let mut deps_hashes = FxHashMap::default();
        for dep_str in live_result.dependencies() {
            let dep_path = Utf8PathBuf::from(dep_str);
            if let Some(h) = Self::hash_file(&dep_path) {
                deps_hashes.insert(dep_path, h);
            }
        }

        let entry = CacheEntry {
            root_path: slang_path.to_path_buf(),
            root_hash,
            deps_hashes,
            reflection: reflection.clone(),
        };

        if let Some(parent) = cache_file_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string(&entry) {
            let _ = std::fs::write(cache_file_path, json);
        }

        entry
    }

    pub fn get_or_compile(
        kernel_name: &str,
        slang_path: &Utf8Path,
        entry_point: &str,
    ) -> Result<ShaderReflection, SlangError> {
        let flattened_path = Self::flatten_workspace_path(slang_path);
        if !slang_path.exists() {
            return Err(SlangError::FileNotFound {
                file_path: flattened_path,
            });
        }
        let cache_key = format!("{}___{}___{}", kernel_name, flattened_path, entry_point);

        // 1. L1 Memory Cache (Read Lock)
        {
            let cache = MEMORY_CACHE.read().unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = cache.get(&cache_key) {
                if Self::is_cache_valid(entry) {
                    return Ok(entry.reflection.clone());
                }
            }
        }

        // Write Lock, when cache miss
        let mut cache = MEMORY_CACHE.write().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = cache.get(&cache_key) {
            if Self::is_cache_valid(entry) {
                return Ok(entry.reflection.clone());
            }
        }

        // L2 cache, disk validation
        let cache_file_path = Self::get_cache_file_path(&cache_key);
        if let Some(disk_entry) = Self::try_load_disk_cache(&cache_file_path) {
            cache.insert(cache_key.clone(), disk_entry.clone());
            return Ok(disk_entry.reflection);
        }

        // Actual compiling - cache miss
        let live_result = GLOBAL_CONTEXT.compile(kernel_name, slang_path.as_str(), entry_point)?;
        let reflection = ShaderReflection::from_compile_result(&live_result);

        // Save disk + memory cache update
        let new_entry =
            Self::save_disk_cache(&cache_file_path, slang_path, &live_result, &reflection);
        cache.insert(cache_key, new_entry);

        Ok(reflection)
    }
}
