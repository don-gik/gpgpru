// src/buffer/linear/gpubuffer.rs

use super::{AsGpuBuffer, BufferTrash, LocationInfo};
use super::{CpuToGpu, GpuOnly, GpuToCpu, Shared};
use crate::__private::UpdateTimeline;
use crate::device::{Collectable, SharedContext};

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk::Buffer;
use gpu_allocator::vulkan::Allocation;

pub struct GpuBuffer<T, Loc: LocationInfo = GpuOnly> {
    pub(crate) buffer: Buffer,
    pub(crate) allocation: Option<Allocation>,
    pub(crate) last_submitted_timeline: AtomicU64,

    device_address: u64,
    size: u64,

    shared: Arc<SharedContext>,
    _marker: PhantomData<(T, Loc)>,
}

unsafe impl<T, Loc: LocationInfo> Send for GpuBuffer<T, Loc> {}
unsafe impl<T, Loc: LocationInfo> Sync for GpuBuffer<T, Loc> {}

impl<T, Loc: LocationInfo> AsGpuBuffer<T> for GpuBuffer<T, Loc> {
    fn base_device_address(&self) -> u64 {
        self.device_address
    }

    fn vk_buffer(&self) -> Buffer {
        self.buffer
    }

    fn offset(&self) -> u64 {
        0
    }

    fn size(&self) -> u64 {
        self.size
    }
}

impl<T, Loc: LocationInfo> Drop for GpuBuffer<T, Loc> {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            let target_timeline = self.last_submitted_timeline.load(Ordering::Relaxed);

            let trash = BufferTrash {
                buffer: self.buffer,
                allocation,
                bda: self.base_device_address(),
                size: self.size(),
            };

            self.shared
                .gc
                .push(target_timeline, Collectable::Buffer(trash));
        }
    }
}

impl<T, Loc: LocationInfo> GpuBuffer<T, Loc> {
    fn get_mapped_slice(&self) -> &[T] {
        let alloc = self.allocation.as_ref().expect("Allocation missing");

        // gpu_allocator -> &[u8]
        let mapped_bytes = alloc.mapped_slice().expect("Memory is not host-mapped!");

        let count = (self.size as usize) / std::mem::size_of::<T>();

        // u8 -> *const T type pointer casting
        unsafe { std::slice::from_raw_parts(mapped_bytes.as_ptr() as *const T, count) }
    }

    fn get_mapped_slice_mut(&mut self) -> &mut [T] {
        let alloc = self.allocation.as_ref().expect("Allocation missing");

        // gpu_allocator -> NonNull<c_void>
        let mapped_ptr = alloc.mapped_ptr().expect("Memory is not host-mapped!");

        let count = (self.size as usize) / std::mem::size_of::<T>();

        // void -> *mut T casting
        unsafe { std::slice::from_raw_parts_mut(mapped_ptr.as_ptr() as *mut T, count) }
    }
}

impl<T> GpuBuffer<T, GpuToCpu> {
    pub fn as_slice(&self) -> &[T] {
        self.get_mapped_slice()
    }
}
impl<T> GpuBuffer<T, Shared> {
    pub fn as_slice(&self) -> &[T] {
        self.get_mapped_slice()
    }
}

impl<T> GpuBuffer<T, CpuToGpu> {
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.get_mapped_slice_mut()
    }
}
impl<T> GpuBuffer<T, Shared> {
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.get_mapped_slice_mut()
    }
}

impl<T, Loc: LocationInfo> GpuBuffer<T, Loc> {
    pub(crate) fn new(
        buffer: Buffer,
        allocation: Option<Allocation>,
        last_submitted_timeline: AtomicU64,
        device_address: u64,
        size: u64,
        shared: Arc<SharedContext>,
        _marker: PhantomData<(T, Loc)>,
    ) -> Self {
        Self {
            buffer,
            allocation,
            last_submitted_timeline,
            device_address,
            size,
            shared,
            _marker,
        }
    }
}

#[derive(Clone, Copy)]
pub struct BufferSlice<'a, T, Loc: LocationInfo = GpuOnly> {
    pub parent: &'a GpuBuffer<T, Loc>,
    pub offset: u64,
    pub size: u64,
}

unsafe impl<'a, T, Loc: LocationInfo> Send for BufferSlice<'a, T, Loc> {}
unsafe impl<'a, T, Loc: LocationInfo> Sync for BufferSlice<'a, T, Loc> {}

impl<'a, T> AsGpuBuffer<T> for BufferSlice<'a, T> {
    fn vk_buffer(&self) -> Buffer {
        self.parent.vk_buffer()
    }
    fn base_device_address(&self) -> u64 {
        self.parent.base_device_address()
    }
    fn offset(&self) -> u64 {
        self.offset
    }
    fn size(&self) -> u64 {
        self.size
    }
}

impl<'a, T> BufferSlice<'a, T, GpuToCpu> {
    pub fn as_slice(&self) -> &[T] {
        let parent_slice = self.parent.as_slice();
        let elem_size = std::mem::size_of::<T>() as u64;

        let start_idx = (self.offset / elem_size) as usize;
        let count = (self.size / elem_size) as usize;

        &parent_slice[start_idx..start_idx + count]
    }
}

impl<'a, T> BufferSlice<'a, T, Shared> {
    pub fn as_slice(&self) -> &[T] {
        let parent_slice = self.parent.as_slice();
        let elem_size = std::mem::size_of::<T>() as u64;

        let start_idx = (self.offset / elem_size) as usize;
        let count = (self.size / elem_size) as usize;

        &parent_slice[start_idx..start_idx + count]
    }
}

// ----------------------------------------------

impl<T, Loc: LocationInfo> UpdateTimeline for GpuBuffer<T, Loc> {
    fn __update_timeline(&self, signal_value: u64) {
        self.last_submitted_timeline
            .fetch_max(signal_value, Ordering::Relaxed);
    }
}

impl<'a, T, Loc: LocationInfo> UpdateTimeline for BufferSlice<'a, T, Loc> {
    fn __update_timeline(&self, signal_value: u64) {
        self.parent.__update_timeline(signal_value);
    }
}
