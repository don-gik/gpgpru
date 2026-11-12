use super::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Stamp(pub(crate) u32);

// ------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct GpuInstant {
    pub(crate) ticks: u64,
    pub(crate) period_ns: f32,
}

impl std::ops::Sub<GpuInstant> for GpuInstant {
    type Output = std::time::Duration;

    fn sub(self, rhs: GpuInstant) -> Self::Output {
        let diff_ticks = self.ticks.saturating_sub(rhs.ticks);
        let ns = (diff_ticks as f64) * (self.period_ns as f64);
        std::time::Duration::from_nanos(ns as u64)
    }
}

// ------------------------------------------

pub struct GpuTimer<const N: usize = 2> {
    query_pool: vk::QueryPool,
    timestamp_period: f32,
    next_slot: AtomicU32,
    shared: Arc<SharedContext>,
}

impl GpuTimer<2> {
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_capacity(device)
    }
}

impl<const N: usize> GpuTimer<N> {
    pub fn with_capacity(device: &Device) -> Result<Self> {
        let query_pool = device.create_query_pool(N as u32)?;
        Ok(Self {
            query_pool,
            timestamp_period: device.timestamp_period(),
            next_slot: AtomicU32::new(0),
            shared: device.shared.clone(),
        })
    }

    pub fn stamp(&self, device: &Device) -> Stamp {
        let slot = self.next_slot.fetch_add(1, Ordering::Relaxed) % (N as u32);
        device.push_command(Task::new_timestamp(self.query_pool, slot));
        Stamp(slot)
    }

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
