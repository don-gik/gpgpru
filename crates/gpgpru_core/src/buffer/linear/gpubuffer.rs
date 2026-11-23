use super::*;
use std::marker::PhantomData;
use std::ops::{Bound, Deref, DerefMut, RangeBounds};
use std::ptr::NonNull;

/// A lightweight, typed view into GPU memory—basically `&[T]`, except the bytes live in VRAM.
///
/// Unlike a normal Rust slice, you can't just index into a `GpuSlice` on the CPU (unless
/// it happens to live in host-visible memory—see [`GpuSlice::try_as_mapped_slice`]).
/// Instead, pass it to your `#[kernel]` functions or hand it to [`read_buffer`] / [`write_buffer`].
///
/// `GpuSlice` is `Copy` and cheap to sub-slice via [`GpuSlice::slice`], using Buffer Device
/// Address (BDA) pointer arithmetic under the hood.
pub struct GpuSlice<'a, T> {
    pub(crate) buffer: vk::Buffer,
    pub(crate) bda: u64,
    pub(crate) size: u64,
    pub(crate) offset: u64,
    pub(crate) mapped_ptr: Option<NonNull<u8>>,
    pub(crate) loc: MemoryLocation,
    _marker: PhantomData<(&'a (), T)>,
}

impl<'a, T> Copy for GpuSlice<'a, T> {}
impl<'a, T> Clone for GpuSlice<'a, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<'a, T> Send for GpuSlice<'a, T> {}
unsafe impl<'a, T> Sync for GpuSlice<'a, T> {}

impl<'a, T> GpuSlice<'a, T> {
    /// Returns the [`MemoryLocation`] where this slice's backing buffer resides.
    #[inline(always)]
    pub const fn loc(&self) -> MemoryLocation {
        self.loc
    }

    /// Returns the 64-bit Vulkan Buffer Device Address (BDA) pointing to the start of this slice.
    ///
    /// Do not use this, this is useless.
    #[inline]
    pub fn bda(&self) -> u64 {
        self.bda
    }

    /// Returns the size of this slice in **bytes** (not element count!).
    #[inline]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns the byte offset of this slice from the start of its underlying `vk::Buffer`.
    #[inline]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the raw Vulkan `vk::Buffer` handle backing this slice.
    ///
    /// Yeah, this is also useless for most cases.
    #[inline]
    pub fn vk_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    /// Returns `true` if this slice is mapped to CPU-accessible address space (e.g., [`Shared`], [`CpuToGpu`], [`GpuToCpu`]).
    #[inline]
    pub fn is_host_visible(&self) -> bool {
        self.mapped_ptr.is_some()
    }

    /// Creates a sub-slice viewing a subrange of elements (e.g., `slice.slice(10..20)`).
    ///
    /// Note that `range` is indexed by **elements of `T`**, not raw bytes.
    ///
    /// # Panics
    ///
    /// Panics if the requested element range goes out of bounds of the current slice.
    pub fn slice<R: RangeBounds<u64>>(&self, range: R) -> GpuSlice<'a, T> {
        let elem_size = std::mem::size_of::<T>() as u64;
        let start = match range.start_bound() {
            Bound::Included(&s) => s * elem_size,
            Bound::Excluded(&s) => (s + 1) * elem_size,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&e) => (e + 1) * elem_size,
            Bound::Excluded(&e) => e * elem_size,
            Bound::Unbounded => self.size,
        };

        assert!(
            start <= end && end <= self.size,
            "GpuSlice range out of bounds"
        );

        GpuSlice {
            buffer: self.buffer,
            bda: self.bda + start,
            size: end - start,
            offset: self.offset + start,
            mapped_ptr: self
                .mapped_ptr
                .map(|ptr| unsafe { NonNull::new_unchecked(ptr.as_ptr().add(start as usize)) }),
            loc: self.loc,
            _marker: PhantomData,
        }
    }

    /// Returns a copy of this `GpuSlice` view.
    #[inline]
    pub fn as_slice(&self) -> GpuSlice<'a, T> {
        *self
    }

    /// Directly exposes the underlying memory as a standard CPU slice `&[T]`, if host-mapped.
    ///
    /// Returns `None` if the buffer lives in [`GpuOnly`] memory (nice try, CPU).
    /// Make sure the GPU has actually finished writing to this memory before reading!
    #[inline]
    pub fn try_as_mapped_slice(&self) -> Option<&[T]> {
        let ptr = self.mapped_ptr?;
        let elem_size = std::mem::size_of::<T>();
        let count = if elem_size == 0 {
            0
        } else {
            (self.size as usize) / elem_size
        };
        Some(unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const T, count) })
    }

    /// Directly exposes the underlying memory as a mutable CPU slice `&mut [T]`, if host-mapped.
    ///
    /// Returns `None` if the buffer lives in [`GpuOnly`] memory.
    /// Beware of data races if the GPU is currently touching this buffer!
    #[inline]
    pub fn try_as_mut_slice(&mut self) -> Option<&mut [T]> {
        let ptr = self.mapped_ptr?;
        let elem_size = std::mem::size_of::<T>();
        let count = if elem_size == 0 {
            0
        } else {
            (self.size as usize) / elem_size
        };
        Some(unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, count) })
    }
}

/// An owned allocation of GPU memory—the VRAM equivalent of a `Vec<T>` or `Box<[T]>`.
///
/// Created via [`create_buffer`], `GpuBuffer` owns the underlying Vulkan buffer and memory
/// allocation. It automatically dereferences to [`GpuSlice<T>`], so you can pass `&buf`
/// or `&mut buf` straight into any `#[kernel]` function.
///
/// # Safe Destruction (No Use-After-Free)
///
/// Dropping a `GpuBuffer` won't immediately yank the rug out from under an in-flight GPU kernel.
/// Instead, it hands the allocation over to an internal garbage collector, which waits until
/// the GPU's timeline semaphore passes the buffer's last recorded access before freeing the VRAM.
///
/// # Type Parameters
///
/// - `T`: The element type stored in the buffer.
/// - `LOC`: A const-generic tag encoding the [`MemoryLocation`] (defaults to [`GpuOnly`]).
pub struct GpuBuffer<T, const LOC: u8 = { GpuOnly.to_u8() }> {
    inner: GpuSlice<'static, T>,
    pub(crate) allocation: Option<Allocation>,
    shared: Arc<SharedContext>,
}

unsafe impl<T, const LOC: u8> Send for GpuBuffer<T, LOC> {}
unsafe impl<T, const LOC: u8> Sync for GpuBuffer<T, LOC> {}

impl<T, const LOC: u8> GpuBuffer<T, LOC> {
    pub(crate) fn new(
        buffer: vk::Buffer,
        allocation: Option<Allocation>,
        device_address: u64,
        size: u64,
        shared: Arc<SharedContext>,
    ) -> Self {
        let mapped_ptr = allocation
            .as_ref()
            .and_then(|a| a.mapped_ptr())
            .map(|ptr| ptr.cast());

        Self {
            inner: GpuSlice {
                buffer,
                bda: device_address,
                size,
                offset: 0,
                mapped_ptr,
                loc: MemoryLocation::from_u8(LOC),
                _marker: PhantomData,
            },
            allocation,
            shared,
        }
    }
}

impl<T, const LOC: u8> Deref for GpuBuffer<T, LOC> {
    type Target = GpuSlice<'static, T>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T, const LOC: u8> DerefMut for GpuBuffer<T, LOC> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T, const LOC: u8> Drop for GpuBuffer<T, LOC> {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            let target_timeline = self
                .shared
                .page_tracker
                .get_last_access_epoch(self.inner.bda(), self.inner.size());

            let trash = BufferTrash {
                buffer: self.inner.buffer,
                allocation,
                bda: self.inner.bda,
                size: self.inner.size,
            };

            // dbg!(target_timeline);

            self.shared
                .gc
                .push(target_timeline, Collectable::Buffer(trash));
        }
    }
}
