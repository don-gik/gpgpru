use super::*;

/// Copies the entire contents of `src` into `dst` on the GPU (`vkCmdCopyBuffer`).
///
/// Both slices must have the exact same byte size, and their lifetimes are tied to the
/// returned [`DmaTask`] so neither buffer can vanish mid-transfer.
///
/// # Panics
///
/// Panics if `src.size() != dst.size()`.
///
/// # Examples
///
/// ```no_run
/// # fn main() -> gpgpru::Result<()> {
/// use gpgpru::prelude::*;
/// use gpgpru::{create_buffer, dma, Device};
///
/// let device = Device::new()?;
/// let src = create_buffer(&vec![1.0f32, 2.0, 3.0, 4.0]).with(&device)?;
/// let mut dst = create_buffer::<f32>(4).with(&device)?;
///
/// dma::copy(&src, &mut dst).with(&device)?;
/// # Ok(())
/// # }
/// ```
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
            let current_epoch = device.pending_epoch().raw();
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

/// Fills an entire [`GpuSlice`] with a repeating pattern (`vkCmdFillBuffer`).
///
/// Great for zeroing out buffers or setting initial sentinel values at hardware DMA speed
/// without spinning up a compute shader. Supports `u32`, `i32`, `f32`, and smaller types.
///
/// # Panics
///
/// Panics if `dst.offset()` or `dst.size()` is not a multiple of 4 bytes (a Vulkan hardware requirement).
///
/// # Examples
///
/// ```no_run
/// # fn main() -> gpgpru::Result<()> {
/// use gpgpru::{create_buffer, dma, Device};
///
/// let device = Device::new()?;
/// let mut buf = create_buffer::<f32>(1024).with(&device)?;
///
/// // Fill the entire buffer with `42.0f32`
/// dma::fill(&mut buf, 42.0).with(&device)?;
/// # Ok(())
/// # }
/// ```
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
            let current_epoch = device.pending_epoch().raw();
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

/// Performs a fast, inline upload of small host data directly into `dst` via the command buffer (`vkCmdUpdateBuffer`).
///
/// [`write_buffer`] internally uses dma operation when possible (64KB cap),
/// so it is preferred to use write_buffer utility instead.
///
/// # Panics
///
/// Due to strict Vulkan `vkCmdUpdateBuffer` hardware rules, this panics if:
/// - `data` is larger than **64 KB** (65,536 bytes)
/// - `data` exceeds `dst.size()`
/// - `dst.offset()` or the total byte size of `data` is not a multiple of **4 bytes**
/// - `size_of::<T>()` is not 1, 2, or a multiple of 4 bytes (checked at compile time)
///
/// # Examples
///
/// ```no_run
/// # fn main() -> gpgpru::Result<()> {
/// use gpgpru::{create_buffer, dma, Device};
///
/// let device = Device::new()?;
/// let mut buf = create_buffer::<u32>(16).with(&device)?;
///
/// dma::update(&mut buf, &[10, 20, 30, 40]).with(&device)?;
/// # Ok(())
/// # }
/// ```
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
            let current_epoch = device.pending_epoch().raw();
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
