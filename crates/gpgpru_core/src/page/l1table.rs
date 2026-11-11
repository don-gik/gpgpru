use super::*;

impl L1Table {
    #[inline(always)]
    pub fn push_down_if_needed(&mut self, l1_idx: usize) -> &mut L2Table {
        let l1_w = self.l1_write_epochs[l1_idx];
        let l1_r = self.l1_read_epochs[l1_idx];

        let l2_table = self.directory[l1_idx].get_or_insert_with(L2Table::new);

        if l1_w != EPOCH_SPLIT {
            let packed_state = PageState {
                last_write: l1_w,
                last_read: l1_r,
            }
            .pack();

            for page in l2_table.pages.iter_mut() {
                page.state = packed_state;
            }

            self.l1_write_epochs[l1_idx] = EPOCH_SPLIT;
            self.l1_read_epochs[l1_idx] = EPOCH_SPLIT;
        }

        l2_table
    }

    pub fn unregister_range(&mut self, bda: u64, size: u64) {
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

            let l1_start_page = (l1_index as u64) << L2_BITS;
            let l1_end_page = l1_start_page + (L2_SIZE as u64) - 1;
            let chunk_end_page = end_page.min(l1_end_page);

            let is_full_l1_cover = (current_page <= l1_start_page) && (end_page >= l1_end_page);

            if is_full_l1_cover {
                self.l1_write_epochs[l1_index] = 0;
                self.l1_read_epochs[l1_index] = 0;
                if let Some(l2_table) = &mut self.directory[l1_index] {
                    for page in l2_table.pages.iter_mut() {
                        page.reset();
                    }
                }
            } else {
                let l2_table = self.push_down_if_needed(l1_index);
                for page in current_page..=chunk_end_page {
                    let l2_index = (page & L2_MASK) as usize;
                    l2_table.pages[l2_index].reset();
                }
            }

            current_page = chunk_end_page + 1;
        }
    }

    pub fn check_and_update(
        &mut self,
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
        let mut accumulated_src_access = vk::AccessFlags2::empty();
        let mut accumulated_src_stage = vk::PipelineStageFlags2::empty();
        let mut accumulated_deps = SmallVec::<[u16; 8]>::new();

        let mut current_page = start_page;

        while current_page <= end_page {
            let l1_index = (current_page >> L2_BITS) as usize;
            if l1_index >= self.directory.len() {
                break;
            }

            let l1_start_page = (l1_index as u64) << L2_BITS;
            let l1_end_page = l1_start_page + (L2_SIZE as u64) - 1;
            let chunk_end_page = end_page.min(l1_end_page);

            let is_full_l1_cover = (current_page <= l1_start_page) && (end_page >= l1_end_page);

            // Fully covered by L1, fast path
            if is_full_l1_cover {
                let l1_w = self.l1_write_epochs[l1_index];
                let l1_r = self.l1_read_epochs[l1_index];

                if l1_w != EPOCH_SPLIT {
                    // UNIFORM state
                    let mut meta = PageMetadata {
                        state: PageState {
                            last_write: l1_w,
                            last_read: l1_r,
                        }
                        .pack(),
                    };

                    let (raw, war, waw, page_src_access, page_src_stage, page_deps) =
                        meta.update(access, encoded_epoch);

                    raw_hazard |= raw;
                    war_hazard |= war;
                    waw_hazard |= waw;
                    accumulated_src_stage |= page_src_stage;
                    accumulated_src_access |= page_src_access;
                    for dep_idx in page_deps {
                        if !accumulated_deps.contains(&dep_idx) {
                            accumulated_deps.push(dep_idx);
                        }
                    }

                    let new_state = PageState::unpack(meta.state);
                    self.l1_write_epochs[l1_index] = new_state.last_write;
                    self.l1_read_epochs[l1_index] = new_state.last_read;
                } else {
                    // SPLIT -> Full Cover, thus iterate over all to update. (IDK this is optimal)
                    if let Some(ref mut l2_table) = self.directory[l1_index] {
                        for page_meta in l2_table.pages.iter_mut() {
                            let (raw, war, waw, page_src_access, page_src_stage, page_deps) =
                                page_meta.update(access, encoded_epoch);

                            raw_hazard |= raw;
                            war_hazard |= war;
                            waw_hazard |= waw;
                            accumulated_src_stage |= page_src_stage;
                            accumulated_src_access |= page_src_access;
                            for dep_idx in page_deps {
                                if !accumulated_deps.contains(&dep_idx) {
                                    accumulated_deps.push(dep_idx);
                                }
                            }
                        }
                    }

                    let mut meta = PageMetadata::default();
                    meta.update(access, encoded_epoch);
                    let new_state = PageState::unpack(meta.state);
                    self.l1_write_epochs[l1_index] = new_state.last_write;
                    self.l1_read_epochs[l1_index] = new_state.last_read;
                }

                current_page = l1_end_page + 1;
                continue;
            }

            // Partially L1 cover -> L2 process.
            let l2_table = self.push_down_if_needed(l1_index);
            for page in current_page..=chunk_end_page {
                let l2_index = (page & L2_MASK) as usize;
                let page_meta = &mut l2_table.pages[l2_index];

                let (raw, war, waw, page_src_access, page_src_stage, page_deps) =
                    page_meta.update(access, encoded_epoch);

                raw_hazard |= raw;
                war_hazard |= war;
                waw_hazard |= waw;
                accumulated_src_stage |= page_src_stage;
                accumulated_src_access |= page_src_access;

                for dep_idx in page_deps {
                    if !accumulated_deps.contains(&dep_idx) {
                        accumulated_deps.push(dep_idx);
                    }
                }
            }

            current_page = chunk_end_page + 1;
        }

        if raw_hazard || war_hazard || waw_hazard || !accumulated_deps.is_empty() {
            let dst_access = access.as_access_flags();

            Some(HazardResult {
                need_barrier: true,
                src_stage: accumulated_src_stage,
                dst_stage: access.as_pipeline_stage(),
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

            let l1_start_page = (l1_index as u64) << L2_BITS;
            let l1_end_page = l1_start_page + (L2_SIZE as u64) - 1;
            let chunk_end_page = end_page.min(l1_end_page);

            let l1_w = self.l1_write_epochs[l1_index];

            if l1_w != EPOCH_SPLIT {
                let epoch = (l1_w & EPOCH_MASK) >> TASK_BITS;
                max_epoch = max_epoch.max(epoch);
            } else if let Some(l2_table) = &self.directory[l1_index] {
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

            let l1_start_page = (l1_index as u64) << L2_BITS;
            let l1_end_page = l1_start_page + (L2_SIZE as u64) - 1;
            let chunk_end_page = end_page.min(l1_end_page);

            let l1_w = self.l1_write_epochs[l1_index];
            let l1_r = self.l1_read_epochs[l1_index];

            if l1_w != EPOCH_SPLIT {
                let w_epoch = (l1_w & EPOCH_MASK) >> TASK_BITS;
                let r_epoch = (l1_r & EPOCH_MASK) >> TASK_BITS;
                max_epoch = max_epoch.max(w_epoch.max(r_epoch));
            } else if let Some(l2_table) = &self.directory[l1_index] {
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
