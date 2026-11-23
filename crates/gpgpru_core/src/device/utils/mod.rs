//! Helpful sidekicks and diagnostic utilities for `gpgpru`.
//!
//! Currently, this module houses hardware-backed GPU profiling tools ([`GpuTimer`], [`Stamp`],
//! and [`GpuInstant`]) so you can measure how fast your kernels *actually* run on the silicon—not
//! how fast the CPU queued them up.

use super::*;

mod time;

pub use time::*;
