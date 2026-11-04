use super::*;

pub fn copy<'a, T: Copy>(
    src: &'a GpuSlice<T>,
    dst: &'a mut GpuSlice<T>,
) -> DmaTask<impl Fn(&Device) -> (bool, HazardResult) + 'a> {
    assert!(
        src.size() == dst.size(),
        "Source and destination slice sizes must match"
    );

    DmaTask {
        dma_op: DmaOp::Copy {
            src: src.vk_buffer(),
            dst: dst.vk_buffer(),
            src_offset: src.offset(),
            dst_offset: dst.offset(),
            size: src.size(),
        },
        get_hazards: move |device| {
            let current_epoch = device.pending_epoch();
            let task_idx = device.next_task_index();

            let mut hazards = HazardResult::default();

            hazards.merge_option(device.shared.page_tracker.check_and_update(
                src.bda(),
                src.size(),
                AccessType::DmaRead,
                current_epoch,
                task_idx,
            ));

            hazards.merge_option(device.shared.page_tracker.check_and_update(
                dst.bda(),
                dst.size(),
                AccessType::DmaWrite,
                current_epoch,
                task_idx,
            ));

            (true, hazards)
        },
    }
}

pub fn fill<'a, T: FillPattern>(
    dst: &'a mut GpuSlice<T>,
    pattern: T,
) -> DmaTask<impl Fn(&Device) -> (bool, HazardResult) + 'a> {
    assert!(dst.offset() % 4 == 0, "dst offset must be 4-byte aligned");
    assert!(
        dst.size() % 4 == 0,
        "dst total size must be a multiple of 4 bytes"
    );

    let pattern_u32 = pattern.into_u32_pattern();

    DmaTask {
        dma_op: DmaOp::Fill {
            dst: dst.vk_buffer(),
            dst_offset: dst.offset(),
            size: dst.size(),
            pattern: pattern_u32,
        },
        get_hazards: move |device| {
            let current_epoch = device.pending_epoch();
            let task_idx = device.next_task_index();

            let mut hazards = HazardResult::default();

            hazards.merge_option(device.shared.page_tracker.check_and_update(
                dst.bda(),
                dst.size(),
                AccessType::DmaWrite,
                current_epoch,
                task_idx,
            ));

            (true, hazards)
        },
    }
}

pub fn update<'a, T: Copy>(
    dst: &'a mut GpuSlice<T>,
    data: &[T],
) -> DmaTask<impl Fn(&Device) -> (bool, HazardResult) + 'a> {
    let byte_size = std::mem::size_of_val(data);

    const {
        let elem_size = std::mem::size_of::<T>();
        assert!(
            elem_size % 4 == 0 || elem_size == 1 || elem_size == 2,
            "Target type T must be 1, 2, or a multiple of 4 bytes to guarantee Vulkan alignment"
        );
    }

    assert!(
        byte_size <= 65536,
        "vkCmdUpdateBuffer payload exceeds 64KB limit (got {} bytes)",
        byte_size
    );
    assert!(
        byte_size as u64 <= dst.size(),
        "Update data size exceeds destination slice size"
    );
    assert!(dst.offset() % 4 == 0, "dst offset must be 4-byte aligned");
    assert!(
        byte_size % 4 == 0,
        "Total update byte size must be a multiple of 4 bytes due to Vulkan vkCmdUpdateBuffer spec (got {} bytes)",
        byte_size
    );

    let byte_slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_size) };

    DmaTask {
        dma_op: DmaOp::Update {
            dst: dst.vk_buffer(),
            dst_offset: dst.offset(),
            data: SmallVec::from_slice(byte_slice),
        },
        get_hazards: move |device| {
            let current_epoch = device.pending_epoch();
            let task_idx = device.next_task_index();

            let mut hazards = HazardResult::default();

            hazards.merge_option(device.shared.page_tracker.check_and_update(
                dst.bda(),
                byte_size as u64,
                AccessType::DmaWrite,
                current_epoch,
                task_idx,
            ));

            (true, hazards)
        },
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_buffer_share() {
        let device = Device::new().unwrap();
        let buffer_a = create_buffer(&vec![0.0]).with(&device).unwrap();
        let mut buffer_b = create_buffer(&vec![1.0]).on(Shared).with(&device).unwrap();
        copy(&buffer_a, &mut buffer_b).with(&device).unwrap();
        dbg!(buffer_b.try_as_mapped_slice());
    }
}
