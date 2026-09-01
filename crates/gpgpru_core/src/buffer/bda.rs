// src/buffer/bda.rs

use bytemuck::{Pod, Zeroable};
use std::marker::PhantomData;

pub struct Bda<T> {
    pub address: u64,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Copy for Bda<T> {}
impl<T> Clone for Bda<T> {
    fn clone(&self) -> Self {
        *self
    }
}
unsafe impl<T: 'static> Pod for Bda<T> {}
unsafe impl<T: 'static> Zeroable for Bda<T> {}

impl<T> Bda<T> {
    pub fn new(address: u64) -> Self {
        Self {
            address,
            _marker: PhantomData,
        }
    }

    pub fn null() -> Self {
        Self {
            address: 0,
            _marker: PhantomData,
        }
    }
}
