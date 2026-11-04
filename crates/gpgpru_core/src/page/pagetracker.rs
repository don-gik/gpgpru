use super::*;

impl PageTracker {
    pub fn unregister_range(&self, bda: u64, size: u64) {
        if size == 0 {
            return;
        }

        let start_page = bda >> PAGE_SHIFT;
        let end_page = (bda + size - 1) >> PAGE_SHIFT;
        let mut current_page = start_page;

        while current_page <= end_page {
            let l1_index = (current_page >> L2_BITS) as usize;
            if l1_index >= self.directory.len() {
                break;
            }

            let l1_end_page = ((l1_index as u64 + 1) << L2_BITS) - 1;
            let chunk_end_page = end_page.min(l1_end_page);

            let slot = &self.directory[l1_index];
            let read_guard = slot.read();

            if let Some(l2_table) = read_guard.as_ref() {
                for page in current_page..=chunk_end_page {
                    let l2_index = (page & L2_MASK) as usize;
                    l2_table.pages[l2_index].reset();
                }
            }

            current_page = chunk_end_page + 1;
        }
    }

    #[inline]
    fn with_l2<R>(&self, l1_index: usize, f: impl FnOnce(&L2Table) -> R) -> R {
        let slot = &self.directory[l1_index];

        // 1. Fast Path (Read Guard)
        {
            let read_guard = slot.read();
            if let Some(l2) = read_guard.as_ref() {
                return f(l2);
            }
        }

        // 2. Slow Path (Write Guard & Alloc)
        let mut write_guard = slot.write();
        let l2 = write_guard.get_or_insert_with(|| Box::new(L2Table::new()));
        f(l2)
    }

    pub fn check_and_update(
        &self,
        bda: u64,
        size: u64,
        access: AccessType,
        current_epoch: u64,
        current_index: u16,
    ) -> Option<HazardResult> {
        if size == 0 {
            return None;
        }

        let encoded_epoch = (current_epoch << TASK_BITS) | (current_index as u64);

        let start_page = bda >> PAGE_SHIFT;
        let end_page = (bda + size - 1) >> PAGE_SHIFT;

        let mut raw_hazard = false;
        let mut war_hazard = false;
        let mut waw_hazard = false;
        let mut accumulated_src_access = AccessFlags2::empty();
        let mut accumulated_deps = SmallVec::<[u16; 8]>::new();

        let mut current_page = start_page;

        while current_page <= end_page {
            let l1_index = (current_page >> L2_BITS) as usize;
            let l1_end_page = ((l1_index as u64 + 1) << L2_BITS) - 1;
            let chunk_end_page = end_page.min(l1_end_page);

            self.with_l2(l1_index, |l2_table| {
                for page in current_page..=chunk_end_page {
                    let l2_index = (page & L2_MASK) as usize;
                    let page_meta = &l2_table.pages[l2_index];

                    let (raw, war, waw, page_src_access, page_deps) =
                        page_meta.update(access, encoded_epoch);

                    raw_hazard |= raw;
                    war_hazard |= war;
                    waw_hazard |= waw;
                    accumulated_src_access |= page_src_access;

                    for dep_idx in page_deps {
                        if !accumulated_deps.contains(&dep_idx) {
                            accumulated_deps.push(dep_idx);
                        }
                    }
                }
            });

            current_page = chunk_end_page + 1;
        }

        if raw_hazard || war_hazard || waw_hazard || !accumulated_deps.is_empty() {
            // dst_access = current access
            let dst_access = access.as_access_flags();

            Some(HazardResult {
                need_barrier: true,
                src_access: accumulated_src_access,
                dst_access,
                deps: accumulated_deps,
            })
        } else {
            None
        }
    }

    pub fn get_last_write_epoch(&self, bda: u64, size: u64) -> u64 {
        if size == 0 {
            return 0;
        }

        let start_page = bda >> PAGE_SHIFT;
        let end_page = (bda + size - 1) >> PAGE_SHIFT;

        let mut max_epoch = 0;
        let mut current_page = start_page;

        while current_page <= end_page {
            let l1_index = (current_page >> L2_BITS) as usize;
            if l1_index >= self.directory.len() {
                break;
            }

            let l1_end_page = ((l1_index as u64 + 1) << L2_BITS) - 1;
            let chunk_end_page = end_page.min(l1_end_page);

            let slot = &self.directory[l1_index];
            let read_guard = slot.read();

            if let Some(l2_table) = read_guard.as_ref() {
                for page in current_page..=chunk_end_page {
                    let l2_index = (page & L2_MASK) as usize;
                    let epoch = l2_table.pages[l2_index].get_last_write();
                    max_epoch = max_epoch.max(epoch);
                }
            }

            current_page = chunk_end_page + 1;
        }

        max_epoch
    }

    pub fn get_last_access_epoch(&self, bda: u64, size: u64) -> u64 {
        if size == 0 {
            return 0;
        }

        let start_page = bda >> PAGE_SHIFT;
        let end_page = (bda + size - 1) >> PAGE_SHIFT;

        let mut max_epoch = 0;
        let mut current_page = start_page;

        while current_page <= end_page {
            let l1_index = (current_page >> L2_BITS) as usize;
            if l1_index >= self.directory.len() {
                break;
            }

            let l1_end_page = ((l1_index as u64 + 1) << L2_BITS) - 1;
            let chunk_end_page = end_page.min(l1_end_page);

            let slot = &self.directory[l1_index];
            let read_guard = slot.read();

            if let Some(l2_table) = read_guard.as_ref() {
                for page in current_page..=chunk_end_page {
                    let l2_index = (page & L2_MASK) as usize;
                    let meta = &l2_table.pages[l2_index];
                    let page_access = std::cmp::max(meta.get_last_read(), meta.get_last_write());
                    max_epoch = max_epoch.max(page_access);
                }
            }

            current_page = chunk_end_page + 1;
        }

        max_epoch
    }
}
