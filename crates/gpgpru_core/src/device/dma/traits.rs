// use super::*;

pub unsafe trait FillPattern: Copy {
    fn into_u32_pattern(self) -> u32;
}

macro_rules! impl_fill_1byte {
    ($($t:ty),*) => {
        $(
            unsafe impl FillPattern for $t {
                #[inline(always)]
                fn into_u32_pattern(self) -> u32 {
                    let b = unsafe { std::mem::transmute_copy::<Self, u8>(&self) };
                    u32::from_ne_bytes([b, b, b, b])
                }
            }
        )*
    };
}

macro_rules! impl_fill_2byte {
    ($($t:ty),*) => {
        $(
            unsafe impl FillPattern for $t {
                #[inline(always)]
                fn into_u32_pattern(self) -> u32 {
                    let b = unsafe { std::mem::transmute_copy::<Self, [u8; 2]>(&self) };
                    u32::from_ne_bytes([b[0], b[1], b[0], b[1]])
                }
            }
        )*
    };
}

macro_rules! impl_fill_4byte {
    ($($t:ty),*) => {
        $(
            unsafe impl FillPattern for $t {
                #[inline(always)]
                fn into_u32_pattern(self) -> u32 {
                    unsafe { std::mem::transmute_copy::<Self, u32>(&self) }
                }
            }
        )*
    };
}

impl_fill_1byte!(u8, i8, bool);
impl_fill_2byte!(u16, i16);
impl_fill_4byte!(u32, i32, f32);
