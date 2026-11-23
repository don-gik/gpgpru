use super::*;

/// An opaque ticket pointing to a recorded timestamp slot inside a [`GpuTimer`].
///
/// Returned by [`GpuTimer::stamp`], you hand this back to [`GpuTimer::fetch`] once the GPU
/// has finished executing the batch to retrieve the actual [`GpuInstant`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Stamp(pub(crate) u32);

// ------------------------------------------

/// A point in time measured directly on the GPU's hardware clock.
///
/// Just like [`std::time::Instant`], you can subtract two `GpuInstant`s (`end - start`)
/// to get a standard [`std::time::Duration`].
///
/// # Note
///
/// Raw GPU ticks are converted to nanoseconds using the physical device's `timestamp_period`.
/// Only compare `GpuInstant` values captured on the same [`Device`].
#[derive(Copy, Clone, Debug)]
pub struct GpuInstant {
    pub(crate) ticks: u64,
    pub(crate) period_ns: f32,
}

impl std::ops::Sub<GpuInstant> for GpuInstant {
    type Output = std::time::Duration;

    /// Computes the elapsed GPU time between `rhs` (start) and `self` (end).
    ///
    /// Saturates at zero if `self` happened earlier than `rhs`.
    fn sub(self, rhs: GpuInstant) -> Self::Output {
        let diff_ticks = self.ticks.saturating_sub(rhs.ticks);
        let ns = (diff_ticks as f64) * (self.period_ns as f64);
        std::time::Duration::from_nanos(ns as u64)
    }
}

// ------------------------------------------

/// A hardware timestamp profiler backed by a Vulkan `vk::QueryPool`.
///
/// Measuring GPU kernels with `std::time::Instant::now()` on the CPU only tells you how fast
/// Vulkan queued the command—not how long the shader actually took. `GpuTimer` injects
/// hardware timestamp commands directly into your command stream so you get real silicon timings.
///
/// The const generic `N` sets the number of timestamp slots in the ring buffer (defaults to `2`,
/// enough for a single `start` and `end` pair). Slots wrap around automatically via modulo, so
/// be careful not to overwrite an un-fetched [`Stamp`] if you take more than `N` stamps in one batch!
///
/// # Examples
///
/// ```no_run
/// # fn main() -> gpgpru::Result<()> {
/// use gpgpru::Device;
/// use gpgpru::utils::GpuTimer;
///
/// let device = Device::new()?;
/// let timer = GpuTimer::new(&device)?;
///
/// let start = timer.stamp(&device);
/// // ... queue your kernels here ...
/// let end = timer.stamp(&device);
///
/// device.dispatch()?;
///
/// let elapsed = timer.fetch(&device, end)? - timer.fetch(&device, start)?;
/// println!("Kernel took: {:?}", elapsed);
/// # Ok(())
/// # }
/// ```
pub struct GpuTimer<const N: usize = 2> {
    query_pool: vk::QueryPool,
    timestamp_period: f32,
    next_slot: AtomicU32,
    shared: Arc<SharedContext>,
}

impl GpuTimer<2> {
    /// Creates a new [`GpuTimer`] with 2 timestamp slots (enough for one `start` and `end` measurement).
    ///
    /// Need to measure multiple stages in a single dispatch? Use [`GpuTimer::<N>::with_capacity`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Vulkan`] if the underlying Vulkan query pool fails to allocate.
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_capacity(device)
    }
}

impl<const N: usize> GpuTimer<N> {
    /// Creates a new [`GpuTimer`] with capacity for `N` timestamp slots.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Vulkan`] if the underlying Vulkan query pool fails to allocate.
    pub fn with_capacity(device: &Device) -> Result<Self> {
        let query_pool = device.create_query_pool(N as u32)?;
        Ok(Self {
            query_pool,
            timestamp_period: device.timestamp_period(),
            next_slot: AtomicU32::new(0),
            shared: device.shared.clone(),
        })
    }

    /// Queues a GPU timestamp command into `device`'s command stream and returns a [`Stamp`] ticket.
    ///
    /// Note that this doesn't read the clock immediately on the CPU—it records a timestamp
    /// instruction into the queue so the GPU captures the exact moment it reaches this point
    /// during [`Device::dispatch`].
    pub fn stamp(&self, device: &Device) -> Stamp {
        let slot = self.next_slot.fetch_add(1, Ordering::Relaxed) % (N as u32);
        device.push_command(Task::new_timestamp(self.query_pool, slot));
        Stamp(slot)
    }

    /// Reads back the resolved [`GpuInstant`] for a previously recorded [`Stamp`].
    ///
    /// # Warning
    ///
    /// This waits (`VK_QUERY_RESULT_WAIT_BIT`) until the GPU has actually executed the
    /// timestamp query. Make sure you have called [`Device::dispatch`] after [`GpuTimer::stamp`]
    /// before fetching—otherwise you'll block forever waiting for a query that was never submitted!
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Vulkan`] if querying the timestamp result from the driver fails.
    pub fn fetch(&self, device: &Device, stamp: Stamp) -> Result<GpuInstant> {
        let ticks = device.get_query_pool_result(self.query_pool, stamp.0)?;
        Ok(GpuInstant {
            ticks,
            period_ns: self.timestamp_period,
        })
    }
}

impl<const N: usize> Drop for GpuTimer<N> {
    fn drop(&mut self) {
        let target_timeline = self
            .shared
            .cmd_manager
            .next_semaphore_value
            .load(Ordering::Acquire);

        self.shared.gc.push(
            target_timeline,
            Collectable::QueryPool(QueryPoolTrash {
                query_pool: self.query_pool,
            }),
        );
    }
}

// ------------------------------------------

pub(crate) struct QueryPoolTrash {
    pub(crate) query_pool: vk::QueryPool,
}

impl Dispose for QueryPoolTrash {
    fn dispose(
        self,
        device: &LogicalDevice,
        _allocator: &AllocationManager,
        _page_tracker: &PageTracker,
    ) {
        unsafe {
            device.destroy_query_pool(self.query_pool, None);
        }
    }
}
