mod ops;
mod traits;

use super::*;

pub use ops::*;
use traits::*;

/// A builder pattern for DMA operations.
pub struct DmaTask<F>
where
    F: Fn(&Device) -> (bool, HazardResult),
{
    dma_op: DmaOp,
    get_hazards: F,
}

impl<F> DmaTask<F>
where
    F: Fn(&Device) -> (bool, HazardResult),
{
    pub const fn new(dma_op: DmaOp, get_hazards: F) -> Self {
        Self {
            dma_op,
            get_hazards,
        }
    }

    pub fn with(self, device: &Device) -> Result<u64> {
        let (has_write, hazards) = (self.get_hazards)(device);

        device.shared.cmd_manager.push(Task {
            payload: TaskPayload::Transfer(self.dma_op),
            target_epoch: 0,
            has_write,
            hazards,
        });
        let signal = device.dispatch()?;
        Ok(signal)
    }
}
