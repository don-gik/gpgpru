// src/page/alloc_page.rs

use super::*;

use ash::vk::AccessFlags2;

impl PageTracker {
    // pub fn register_range(&self, bda: u64, size: u64) {
    //     if size == 0 {
    //         return;
    //     }

    //     let start_page = bda >> PAGE_SHIFT;
    //     let end_page = (bda + size - 1) >> PAGE_SHIFT;

    //     for page in start_page..=end_page {
    //         let l1_index = (page >> 16) as usize;
    //         let l2_index = (page & L2_MASK) as usize;

    //         let slot = &self.directory[l1_index];
    //         let mut write_guard = slot.write();

    //         let l2_table = write_guard.get_or_insert_with(|| Box::new(L2Table::new()));
    //         let page_meta = &l2_table.pages[l2_index];

    //         page_meta.last_read_epoch.store(0, Ordering::Relaxed);
    //         page_meta.last_write_epoch.store(0, Ordering::Relaxed);
    //     }
    // }

    pub fn unregister_range(&self, bda: u64, size: u64) {
        if size == 0 {
            return;
        }

        let start_page = bda >> PAGE_SHIFT;
        let end_page = (bda + size - 1) >> PAGE_SHIFT;

        for page in start_page..=end_page {
            let l1_index = (page >> 16) as usize;
            let l2_index = (page & L2_MASK) as usize;

            let slot = &self.directory[l1_index];
            let read_guard = slot.read();

            if let Some(l2_table) = read_guard.as_ref() {
                let page_meta = &l2_table.pages[l2_index];
                page_meta.last_read_epoch.store(0, Ordering::Relaxed);
                page_meta.last_write_epoch.store(0, Ordering::Relaxed);
            }
        }
    }

    pub fn check_and_update(
        &self,
        bda: u64,
        size: u64,
        access: AccessType,
        current_epoch: u64,
        gpu_completed_epoch: u64,
    ) -> Option<HazardResult> {
        if size == 0 {
            return None;
        }

        let start_page = bda >> PAGE_SHIFT;
        let end_page = (bda + size - 1) >> PAGE_SHIFT;

        let mut raw_hazard = false;
        let mut war_hazard = false;
        let mut waw_hazard = false;

        for page in start_page..=end_page {
            let l1_index = (page >> 16) as usize;
            let l2_index = (page & L2_MASK) as usize;

            let read_guard = &self.directory[l1_index].read();
            if let Some(l2_table) = read_guard.as_ref() {
                let page_meta = &l2_table.pages[l2_index];

                let last_write = page_meta.last_write_epoch.load(Ordering::Relaxed);
                let last_read = page_meta.last_read_epoch.load(Ordering::Relaxed);

                match access {
                    AccessType::Read => {
                        // RAW
                        if last_write > gpu_completed_epoch {
                            raw_hazard = true;
                        }

                        page_meta
                            .last_read_epoch
                            .fetch_max(current_epoch, Ordering::Relaxed);
                    }

                    AccessType::Write => {
                        // WAR
                        if last_read > gpu_completed_epoch {
                            war_hazard = true;
                        }

                        // WAW
                        if last_write > gpu_completed_epoch {
                            waw_hazard = true;
                        }

                        page_meta
                            .last_write_epoch
                            .store(current_epoch, Ordering::Relaxed);
                    }
                }
            }
        }

        if raw_hazard || war_hazard || waw_hazard {
            Some(HazardResult {
                need_barrier: true,
                src_access: if raw_hazard || waw_hazard {
                    AccessFlags2::SHADER_WRITE
                } else {
                    AccessFlags2::SHADER_READ
                },
                dst_access: if access == AccessType::Write {
                    AccessFlags2::SHADER_WRITE
                } else {
                    AccessFlags2::SHADER_READ
                },
            })
        } else {
            None
        }
    }
}
