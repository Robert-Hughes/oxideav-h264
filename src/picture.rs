//! H.264 decoded-picture storage and reference metadata.
//!
//! Reconstructed samples are stored in their final unsigned representation:
//! `u8` for 8-bit pictures and little-endian `u16` containers for 9..=14-bit
//! pictures. Transform, prediction and filtering arithmetic remains `i32`; only
//! values already clipped to the legal sample range are committed here.
//!
//! A picture is mutable while reconstruction/deblocking is in progress.
//! [`Picture::freeze`] turns the exact same pooled allocation into an immutable
//! arena frame which can later be retained simultaneously by the H.264 DPB and
//! by an application [`oxideav_core::FrameLease`].

use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use oxideav_core::arena::sync::{
    Arena, ArenaPool, Frame as ArenaFrame, FrameHeader, VideoFrameBuilder,
};
use oxideav_core::{PixelFormat, Result};

/// §8.4.1.2.1 Table 8-7 — `PicCodingStruct( X )` of a decoded picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PicCodingStruct {
    #[default]
    Frm,
    Fld,
    Afrm,
}

/// H.264-only metadata associated with decoded samples.
///
/// Pixel ownership is deliberately absent: cloning this value never copies a
/// decoded picture. [`Picture`] dereferences to this type so existing codec
/// bookkeeping can continue to use `pic.pic_order_cnt`, `pic.mv_l0_grid`, etc.
#[derive(Debug, Clone)]
pub struct H264PictureMeta {
    pub width_in_samples: u32,
    pub height_in_samples: u32,
    pub chroma_array_type: u32,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub non_existing: bool,
    pub pic_order_cnt: i32,
    pub frame_num: u32,
    pub mb_width_in_picture: u32,
    pub mv_l0_grid: Vec<(i16, i16)>,
    pub mv_l1_grid: Vec<(i16, i16)>,
    pub ref_idx_l0_grid: Vec<i8>,
    pub ref_idx_l1_grid: Vec<i8>,
    pub is_intra_grid: Vec<bool>,
    pub ref_list_0_pocs: Vec<i32>,
    pub ref_list_1_pocs: Vec<i32>,
    pub ref_list_0_longterm: Vec<bool>,
    pub ref_list_1_longterm: Vec<bool>,
    pub coding_struct: PicCodingStruct,
    pub is_bottom_field: bool,
    pub top_field_order_cnt: i32,
    pub bottom_field_order_cnt: i32,
    pub mb_field_flags: Vec<bool>,
    pub view_of_frame_parity: Option<u8>,
    pub ref_list_0_keys: Vec<u32>,
    pub ref_list_0_parities: Vec<Option<u8>>,
    pub ref_list_0_unit_keys: Vec<u32>,
    pub ref_list_1_keys: Vec<u32>,
    pub ref_list_1_parities: Vec<Option<u8>>,
    pub ref_list_1_unit_keys: Vec<u32>,
}

impl H264PictureMeta {
    fn new(
        width_in_samples: u32,
        height_in_samples: u32,
        chroma_array_type: u32,
        bit_depth_luma: u32,
        bit_depth_chroma: u32,
    ) -> Self {
        Self {
            width_in_samples,
            height_in_samples,
            chroma_array_type,
            bit_depth_luma,
            bit_depth_chroma,
            non_existing: false,
            pic_order_cnt: 0,
            frame_num: 0,
            mb_width_in_picture: 0,
            mv_l0_grid: Vec::new(),
            mv_l1_grid: Vec::new(),
            ref_idx_l0_grid: Vec::new(),
            ref_idx_l1_grid: Vec::new(),
            is_intra_grid: Vec::new(),
            ref_list_0_pocs: Vec::new(),
            ref_list_1_pocs: Vec::new(),
            ref_list_0_longterm: Vec::new(),
            ref_list_1_longterm: Vec::new(),
            coding_struct: PicCodingStruct::default(),
            is_bottom_field: false,
            top_field_order_cnt: 0,
            bottom_field_order_cnt: 0,
            mb_field_flags: Vec::new(),
            view_of_frame_parity: None,
            ref_list_0_keys: Vec::new(),
            ref_list_0_parities: Vec::new(),
            ref_list_0_unit_keys: Vec::new(),
            ref_list_1_keys: Vec::new(),
            ref_list_1_parities: Vec::new(),
            ref_list_1_unit_keys: Vec::new(),
        }
    }
}

enum PictureStorage {
    Writable8(VideoFrameBuilder<u8>),
    Writable16(VideoFrameBuilder<u16>),
    Frozen(ArenaFrame),
}

/// Typed read-only plane used by motion-compensation kernels.
#[derive(Clone, Copy)]
pub(crate) enum SamplePlane<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
}

impl SamplePlane<'_> {
    pub(crate) fn len(self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
        }
    }

    #[inline]
    pub(crate) fn sample(self, index: usize) -> i32 {
        match self {
            Self::U8(v) => v[index] as i32,
            Self::U16(v) => u16::from_le(v[index]) as i32,
        }
    }
    pub(crate) fn offset(self, offset: usize) -> Self {
        match self {
            Self::U8(v) => Self::U8(&v[offset..]),
            Self::U16(v) => Self::U16(&v[offset..]),
        }
    }
}

/// Mutable-then-frozen decoded picture.
pub struct Picture {
    storage: Option<PictureStorage>,
    meta: H264PictureMeta,
}

impl fmt::Debug for Picture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Picture")
            .field("meta", &self.meta)
            .field("frozen", &self.is_frozen())
            .finish()
    }
}

impl Deref for Picture {
    type Target = H264PictureMeta;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl DerefMut for Picture {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meta
    }
}

impl Picture {
    /// Allocate a standalone picture. Primarily for tests and the approved
    /// PAFF/SCP copy fallbacks; production decode should use [`Self::new_in`].
    pub fn new(
        width_in_samples: u32,
        height_in_samples: u32,
        chroma_array_type: u32,
        bit_depth_luma: u32,
        bit_depth_chroma: u32,
    ) -> Self {
        let cap = Self::required_bytes(
            width_in_samples,
            height_in_samples,
            chroma_array_type,
            bit_depth_luma,
            bit_depth_chroma,
        )
        .saturating_add(3 * 64);
        let pool = ArenaPool::new(1, cap);
        Self::new_in(
            &pool,
            width_in_samples,
            height_in_samples,
            chroma_array_type,
            bit_depth_luma,
            bit_depth_chroma,
        )
        .expect("standalone H.264 picture allocation")
    }

    /// Allocate reconstruction planes from a reusable arena pool without
    /// blocking. Callers that are part of a producer/consumer pipeline and
    /// want pool pressure to become back-pressure should use
    /// [`Self::new_in_wait`].
    pub fn new_in(
        pool: &Arc<ArenaPool>,
        width_in_samples: u32,
        height_in_samples: u32,
        chroma_array_type: u32,
        bit_depth_luma: u32,
        bit_depth_chroma: u32,
    ) -> Result<Self> {
        Self::new_in_arena(
            pool.lease()?,
            width_in_samples,
            height_in_samples,
            chroma_array_type,
            bit_depth_luma,
            bit_depth_chroma,
        )
    }

    /// Allocate reconstruction planes from a reusable arena pool, waiting for
    /// an existing retained frame to release a slot when the pool is full.
    ///
    /// The software streaming decoder uses this path so temporary downstream
    /// back-pressure does not turn into a codec error and corrupt its picture
    /// state. The wait is fulfilled by [`ArenaPool`] when any arena from this
    /// pool is returned.
    pub fn new_in_wait(
        pool: &Arc<ArenaPool>,
        width_in_samples: u32,
        height_in_samples: u32,
        chroma_array_type: u32,
        bit_depth_luma: u32,
        bit_depth_chroma: u32,
    ) -> Result<Self> {
        Self::new_in_arena(
            pool.lease_wait()?,
            width_in_samples,
            height_in_samples,
            chroma_array_type,
            bit_depth_luma,
            bit_depth_chroma,
        )
    }

    fn new_in_arena(
        arena: Arena,
        width_in_samples: u32,
        height_in_samples: u32,
        chroma_array_type: u32,
        bit_depth_luma: u32,
        bit_depth_chroma: u32,
    ) -> Result<Self> {
        let (cw, ch) = chroma_dims(chroma_array_type, width_in_samples, height_in_samples);
        let luma_len = width_in_samples as usize * height_in_samples as usize;
        let chroma_len = cw as usize * ch as usize;
        let plane_elements = if chroma_array_type == 0 {
            vec![luma_len]
        } else {
            vec![luma_len, chroma_len, chroma_len]
        };
        let wide = bit_depth_luma.max(bit_depth_chroma) > 8;
        let bytes_per_sample = if wide { 2 } else { 1 };
        let mut strides = vec![width_in_samples as usize * bytes_per_sample];
        if chroma_array_type != 0 {
            strides.push(cw as usize * bytes_per_sample);
            strides.push(cw as usize * bytes_per_sample);
        }
        let storage = if wide {
            PictureStorage::Writable16(VideoFrameBuilder::<u16>::new(
                arena,
                &plane_elements,
                &strides,
            )?)
        } else {
            PictureStorage::Writable8(VideoFrameBuilder::<u8>::new(
                arena,
                &plane_elements,
                &strides,
            )?)
        };
        Ok(Self {
            storage: Some(storage),
            meta: H264PictureMeta::new(
                width_in_samples,
                height_in_samples,
                chroma_array_type,
                bit_depth_luma,
                bit_depth_chroma,
            ),
        })
    }

    pub fn required_bytes(
        width: u32,
        height: u32,
        chroma_array_type: u32,
        bit_depth_luma: u32,
        bit_depth_chroma: u32,
    ) -> usize {
        let (cw, ch) = chroma_dims(chroma_array_type, width, height);
        let samples = width as usize * height as usize + 2 * cw as usize * ch as usize;
        let bytes_per_sample = if bit_depth_luma.max(bit_depth_chroma) > 8 {
            2
        } else {
            1
        };
        samples.saturating_mul(bytes_per_sample)
    }

    pub fn is_frozen(&self) -> bool {
        matches!(self.storage, Some(PictureStorage::Frozen(_)))
    }

    fn sample_plane(&self, plane: usize) -> SamplePlane<'_> {
        match self.storage.as_ref().expect("picture storage") {
            PictureStorage::Writable8(b) => SamplePlane::U8(b.plane(plane).expect("picture plane")),
            PictureStorage::Writable16(b) => {
                SamplePlane::U16(b.plane(plane).expect("picture plane"))
            }
            PictureStorage::Frozen(frame) => {
                let bytes = frame.plane(plane).expect("picture plane");
                if self.bit_depth_luma.max(self.bit_depth_chroma) <= 8 {
                    SamplePlane::U8(bytes)
                } else {
                    debug_assert_eq!(bytes.len() % 2, 0);
                    debug_assert_eq!((bytes.as_ptr() as usize) % std::mem::align_of::<u16>(), 0);
                    // SAFETY: H.264 wide frames originate from
                    // `VideoFrameBuilder<u16>` and therefore retain u16 alignment.
                    SamplePlane::U16(unsafe {
                        std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), bytes.len() / 2)
                    })
                }
            }
        }
    }

    pub(crate) fn luma_plane(&self) -> SamplePlane<'_> {
        self.sample_plane(0)
    }

    pub(crate) fn cb_plane(&self) -> SamplePlane<'_> {
        self.sample_plane(1)
    }

    pub(crate) fn cr_plane(&self) -> SamplePlane<'_> {
        self.sample_plane(2)
    }

    #[inline]
    pub fn luma_sample(&self, index: usize) -> i32 {
        self.sample_plane(0).sample(index)
    }

    #[inline]
    pub fn cb_sample(&self, index: usize) -> i32 {
        self.sample_plane(1).sample(index)
    }

    #[inline]
    pub fn cr_sample(&self, index: usize) -> i32 {
        self.sample_plane(2).sample(index)
    }

    fn set_plane_sample(&mut self, plane: usize, index: usize, value: i32, bit_depth: u32) {
        let hi = (1i32 << bit_depth) - 1;
        let value = value.clamp(0, hi);
        match self.storage.as_mut().expect("picture storage") {
            PictureStorage::Writable8(b) => {
                b.plane_mut(plane).expect("picture plane")[index] = value as u8;
            }
            PictureStorage::Writable16(b) => {
                b.plane_mut(plane).expect("picture plane")[index] = (value as u16).to_le();
            }
            PictureStorage::Frozen(_) => panic!("attempted to mutate frozen H.264 picture"),
        }
    }

    pub(crate) fn set_luma_sample(&mut self, index: usize, value: i32) {
        self.set_plane_sample(0, index, value, self.bit_depth_luma);
    }

    pub(crate) fn set_cb_sample(&mut self, index: usize, value: i32) {
        self.set_plane_sample(1, index, value, self.bit_depth_chroma);
    }

    pub(crate) fn set_cr_sample(&mut self, index: usize, value: i32) {
        self.set_plane_sample(2, index, value, self.bit_depth_chroma);
    }

    pub(crate) fn chroma_sample(&self, plane: u8, index: usize) -> i32 {
        if plane == 0 {
            self.cb_sample(index)
        } else {
            self.cr_sample(index)
        }
    }

    pub(crate) fn set_chroma_sample(&mut self, plane: u8, index: usize, value: i32) {
        if plane == 0 {
            self.set_cb_sample(index, value);
        } else {
            self.set_cr_sample(index, value);
        }
    }

    fn copy_plane_to_i32(&self, plane: usize, start: usize, dst: &mut [i32]) {
        let len = dst.len();
        match self.sample_plane(plane) {
            SamplePlane::U8(src) => {
                for (out, &value) in dst.iter_mut().zip(&src[start..start + len]) {
                    *out = value as i32;
                }
            }
            SamplePlane::U16(src) => {
                for (out, &value) in dst.iter_mut().zip(&src[start..start + len]) {
                    *out = u16::from_le(value) as i32;
                }
            }
        }
    }

    fn copy_plane_from_i32(&mut self, plane: usize, start: usize, src: &[i32], bit_depth: u32) {
        let hi = (1i32 << bit_depth) - 1;
        match self.storage.as_mut().expect("picture storage") {
            PictureStorage::Writable8(builder) => {
                let dst =
                    &mut builder.plane_mut(plane).expect("picture plane")[start..start + src.len()];
                for (dst, &value) in dst.iter_mut().zip(src) {
                    *dst = value.clamp(0, hi) as u8;
                }
            }
            PictureStorage::Writable16(builder) => {
                let dst =
                    &mut builder.plane_mut(plane).expect("picture plane")[start..start + src.len()];
                for (dst, &value) in dst.iter_mut().zip(src) {
                    *dst = (value.clamp(0, hi) as u16).to_le();
                }
            }
            PictureStorage::Frozen(_) => panic!("attempted to mutate frozen H.264 picture"),
        }
    }

    pub(crate) fn copy_luma_range_to_i32(&self, start: usize, dst: &mut [i32]) {
        self.copy_plane_to_i32(0, start, dst);
    }

    pub(crate) fn copy_luma_range_from_i32(&mut self, start: usize, src: &[i32]) {
        self.copy_plane_from_i32(0, start, src, self.bit_depth_luma);
    }

    pub(crate) fn copy_chroma_range_to_i32(&self, plane: u8, start: usize, dst: &mut [i32]) {
        self.copy_plane_to_i32(usize::from(plane) + 1, start, dst);
    }

    pub(crate) fn copy_chroma_range_from_i32(&mut self, plane: u8, start: usize, src: &[i32]) {
        self.copy_plane_from_i32(usize::from(plane) + 1, start, src, self.bit_depth_chroma);
    }

    pub(crate) fn copy_luma_to_i32(&self, dst: &mut [i32]) {
        self.copy_plane_to_i32(0, 0, dst);
    }

    pub(crate) fn copy_cb_to_i32(&self, dst: &mut [i32]) {
        if self.chroma_array_type == 0 {
            debug_assert!(dst.is_empty());
            return;
        }
        self.copy_plane_to_i32(1, 0, dst);
    }

    pub(crate) fn copy_cr_to_i32(&self, dst: &mut [i32]) {
        if self.chroma_array_type == 0 {
            debug_assert!(dst.is_empty());
            return;
        }
        self.copy_plane_to_i32(2, 0, dst);
    }

    pub(crate) fn copy_luma_from_i32(&mut self, src: &[i32]) {
        self.copy_plane_from_i32(0, 0, src, self.bit_depth_luma);
    }

    pub(crate) fn copy_cb_from_i32(&mut self, src: &[i32]) {
        if self.chroma_array_type == 0 {
            debug_assert!(src.is_empty());
            return;
        }
        self.copy_plane_from_i32(1, 0, src, self.bit_depth_chroma);
    }

    pub(crate) fn copy_cr_from_i32(&mut self, src: &[i32]) {
        if self.chroma_array_type == 0 {
            debug_assert!(src.is_empty());
            return;
        }
        self.copy_plane_from_i32(2, 0, src, self.bit_depth_chroma);
    }

    pub(crate) fn fill_luma(&mut self, value: i32) {
        let len = self.luma_plane().len();
        for i in 0..len {
            self.set_luma_sample(i, value);
        }
    }

    pub(crate) fn fill_cb(&mut self, value: i32) {
        if self.chroma_array_type == 0 {
            return;
        }
        let len = self.cb_plane().len();
        for i in 0..len {
            self.set_cb_sample(i, value);
        }
    }

    pub(crate) fn fill_cr(&mut self, value: i32) {
        if self.chroma_array_type == 0 {
            return;
        }
        let len = self.cr_plane().len();
        for i in 0..len {
            self.set_cr_sample(i, value);
        }
    }

    /// Freeze the current sample allocation into the application/DPB format.
    pub fn freeze(&mut self, pts: Option<i64>) -> Result<ArenaFrame> {
        if let Some(PictureStorage::Frozen(frame)) = self.storage.as_ref() {
            return Ok(Arc::clone(frame));
        }
        let storage = self.storage.take().expect("picture storage");
        let mut header = FrameHeader::new(
            self.width_in_samples,
            self.height_in_samples,
            self.pixel_format(),
            pts,
        );
        if let Some(bits) = self.significant_bits_metadata() {
            header = header.with_significant_bits(&bits)?;
        }
        let frame = match storage {
            PictureStorage::Writable8(b) => b.freeze(header)?,
            PictureStorage::Writable16(b) => b.freeze(header)?,
            PictureStorage::Frozen(frame) => frame,
        };
        self.storage = Some(PictureStorage::Frozen(Arc::clone(&frame)));
        Ok(frame)
    }

    pub fn frozen_frame(&self) -> Option<&ArenaFrame> {
        match self.storage.as_ref()? {
            PictureStorage::Frozen(frame) => Some(frame),
            _ => None,
        }
    }

    pub fn pixel_format(&self) -> PixelFormat {
        use PixelFormat::*;
        let depth = self.bit_depth_luma.max(self.bit_depth_chroma);
        match (self.chroma_array_type, depth) {
            (0, 0..=8) => Gray8,
            (0, 9..=10) => Gray10Le,
            (0, 11..=12) => Gray12Le,
            (0, _) => Gray16Le,
            (1, 0..=8) => Yuv420P,
            (2, 0..=8) => Yuv422P,
            (3, 0..=8) => Yuv444P,
            (1, 9..=10) => Yuv420P10Le,
            (2, 9..=10) => Yuv422P10Le,
            (3, 9..=10) => Yuv444P10Le,
            (1, 11..=12) => Yuv420P12Le,
            (2, 11..=12) => Yuv422P12Le,
            (3, 11..=12) => Yuv444P12Le,
            (1, _) => Yuv420P16Le,
            (2, _) => Yuv422P16Le,
            (3, _) => Yuv444P16Le,
            _ => Gray8,
        }
    }

    /// Exact per-plane precision when the public pixel format's nominal depth
    /// does not fully describe this H.264 picture. Common 8/10/12-bit streams
    /// need no side channel; 9/11/13/14-bit or mixed-depth pictures do.
    pub(crate) fn significant_bits_metadata(&self) -> Option<Vec<u8>> {
        let format_depth = match self.bit_depth_luma.max(self.bit_depth_chroma) {
            0..=8 => 8,
            9..=10 => 10,
            11..=12 => 12,
            _ => 16,
        };
        let mut bits = vec![self.bit_depth_luma as u8];
        if self.chroma_array_type != 0 {
            bits.push(self.bit_depth_chroma as u8);
            bits.push(self.bit_depth_chroma as u8);
        }
        (!bits.iter().all(|&bits| bits == format_depth)).then_some(bits)
    }

    /// Explicit deep copy used only by the approved PAFF/SCP fallback paths
    /// and compatibility tests while the DPB hand-off is being migrated.
    pub fn deep_copy(&self) -> Picture {
        let mut out = Picture::new(
            self.width_in_samples,
            self.height_in_samples,
            self.chroma_array_type,
            self.bit_depth_luma,
            self.bit_depth_chroma,
        );
        for i in 0..self.luma_plane().len() {
            out.set_luma_sample(i, self.luma_sample(i));
        }
        if self.chroma_array_type != 0 {
            for i in 0..self.cb_plane().len() {
                out.set_cb_sample(i, self.cb_sample(i));
                out.set_cr_sample(i, self.cr_sample(i));
            }
        }
        out.meta = self.meta.clone();
        out
    }

    /// PAFF fallback: materialise one parity field into its own compact picture.
    pub fn field_view(&self, bottom: bool) -> Picture {
        let mut out = Picture::new(
            self.width_in_samples,
            self.height_in_samples / 2,
            self.chroma_array_type,
            self.bit_depth_luma,
            self.bit_depth_chroma,
        );
        let w = self.width_in_samples as usize;
        for (dst_row, src_row) in (usize::from(bottom)..self.height_in_samples as usize)
            .step_by(2)
            .enumerate()
        {
            for x in 0..w {
                out.set_luma_sample(dst_row * w + x, self.luma_sample(src_row * w + x));
            }
        }
        let cw = self.chroma_width() as usize;
        let ch = self.chroma_height() as usize;
        if ch > 0 {
            for (dst_row, src_row) in (usize::from(bottom)..ch).step_by(2).enumerate() {
                for x in 0..cw {
                    out.set_cb_sample(dst_row * cw + x, self.cb_sample(src_row * cw + x));
                    out.set_cr_sample(dst_row * cw + x, self.cr_sample(src_row * cw + x));
                }
            }
        }
        out.meta = self.meta.clone();
        out.height_in_samples /= 2;
        out.view_of_frame_parity = Some(u8::from(bottom));
        out
    }

    pub fn chroma_width(&self) -> u32 {
        chroma_dims(
            self.chroma_array_type,
            self.width_in_samples,
            self.height_in_samples,
        )
        .0
    }

    pub fn chroma_height(&self) -> u32 {
        chroma_dims(
            self.chroma_array_type,
            self.width_in_samples,
            self.height_in_samples,
        )
        .1
    }

    #[inline]
    pub fn luma_at(&self, x: i32, y: i32) -> i32 {
        if self.width_in_samples == 0 || self.height_in_samples == 0 {
            return 0;
        }
        let xi = clip3_i32(0, self.width_in_samples as i32 - 1, x) as usize;
        let yi = clip3_i32(0, self.height_in_samples as i32 - 1, y) as usize;
        self.luma_sample(yi * self.width_in_samples as usize + xi)
    }

    #[inline]
    pub fn cb_at(&self, x: i32, y: i32) -> i32 {
        let (cw, ch) = (self.chroma_width(), self.chroma_height());
        if cw == 0 || ch == 0 {
            return 0;
        }
        let xi = clip3_i32(0, cw as i32 - 1, x) as usize;
        let yi = clip3_i32(0, ch as i32 - 1, y) as usize;
        self.cb_sample(yi * cw as usize + xi)
    }

    #[inline]
    pub fn cr_at(&self, x: i32, y: i32) -> i32 {
        let (cw, ch) = (self.chroma_width(), self.chroma_height());
        if cw == 0 || ch == 0 {
            return 0;
        }
        let xi = clip3_i32(0, cw as i32 - 1, x) as usize;
        let yi = clip3_i32(0, ch as i32 - 1, y) as usize;
        self.cr_sample(yi * cw as usize + xi)
    }

    #[inline]
    pub fn set_luma(&mut self, x: i32, y: i32, v: i32) {
        if x < 0 || y < 0 || x as u32 >= self.width_in_samples || y as u32 >= self.height_in_samples
        {
            return;
        }
        self.set_luma_sample(y as usize * self.width_in_samples as usize + x as usize, v);
    }

    #[inline]
    pub fn set_cb(&mut self, x: i32, y: i32, v: i32) {
        let (cw, ch) = (self.chroma_width(), self.chroma_height());
        if x < 0 || y < 0 || x as u32 >= cw || y as u32 >= ch {
            return;
        }
        self.set_cb_sample(y as usize * cw as usize + x as usize, v);
    }

    #[inline]
    pub fn set_cr(&mut self, x: i32, y: i32, v: i32) {
        let (cw, ch) = (self.chroma_width(), self.chroma_height());
        if x < 0 || y < 0 || x as u32 >= cw || y as u32 >= ch {
            return;
        }
        self.set_cr_sample(y as usize * cw as usize + x as usize, v);
    }

    pub fn colocated_l0(&self, mb_addr: u32, blk4: usize) -> Option<((i16, i16), i8, bool)> {
        if self.mv_l0_grid.is_empty() {
            return None;
        }
        let mv_idx = (mb_addr as usize).checked_mul(16)?.checked_add(blk4)?;
        let mv = *self.mv_l0_grid.get(mv_idx)?;
        let ref_idx = *self
            .ref_idx_l0_grid
            .get((mb_addr as usize) * 4 + (blk4 / 4))?;
        let is_intra = self
            .is_intra_grid
            .get(mb_addr as usize)
            .copied()
            .unwrap_or(false);
        Some((mv, ref_idx, is_intra))
    }

    pub fn colocated_l1(&self, mb_addr: u32, blk4: usize) -> Option<((i16, i16), i8, bool)> {
        if self.mv_l1_grid.is_empty() {
            return None;
        }
        let mv_idx = (mb_addr as usize).checked_mul(16)?.checked_add(blk4)?;
        let mv = *self.mv_l1_grid.get(mv_idx)?;
        let ref_idx = *self
            .ref_idx_l1_grid
            .get((mb_addr as usize) * 4 + (blk4 / 4))?;
        let is_intra = self
            .is_intra_grid
            .get(mb_addr as usize)
            .copied()
            .unwrap_or(false);
        Some((mv, ref_idx, is_intra))
    }
    #[cfg(test)]
    pub(crate) fn luma_values(&self) -> Vec<i32> {
        let mut values = vec![0i32; self.luma_plane().len()];
        self.copy_luma_to_i32(&mut values);
        values
    }

    #[cfg(test)]
    pub(crate) fn cb_values(&self) -> Vec<i32> {
        if self.chroma_array_type == 0 {
            return Vec::new();
        }
        let mut values = vec![0i32; self.cb_plane().len()];
        self.copy_cb_to_i32(&mut values);
        values
    }

    #[cfg(test)]
    pub(crate) fn cr_values(&self) -> Vec<i32> {
        if self.chroma_array_type == 0 {
            return Vec::new();
        }
        let mut values = vec![0i32; self.cr_plane().len()];
        self.copy_cr_to_i32(&mut values);
        values
    }
}

fn chroma_dims(chroma_array_type: u32, w: u32, h: u32) -> (u32, u32) {
    match chroma_array_type {
        0 => (0, 0),
        1 => (w / 2, h / 2),
        2 => (w / 2, h),
        3 => (w, h),
        _ => (0, 0),
    }
}

#[inline]
fn clip3_i32(x: i32, y: i32, z: i32) -> i32 {
    if z < x {
        x
    } else if z > y {
        y
    } else {
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_monochrome() {
        let p = Picture::new(32, 16, 0, 8, 8);
        assert_eq!(p.luma_values().len(), 32 * 16);
        assert!(p.cb_values().is_empty());
        assert!(p.cr_values().is_empty());
        assert_eq!(p.chroma_width(), 0);
        assert_eq!(p.chroma_height(), 0);
    }

    #[test]
    fn allocation_yuv420() {
        let p = Picture::new(32, 16, 1, 8, 8);
        assert_eq!(p.luma_values().len(), 32 * 16);
        assert_eq!(p.cb_values().len(), 16 * 8);
        assert_eq!(p.cr_values().len(), 16 * 8);
        assert_eq!(p.chroma_width(), 16);
        assert_eq!(p.chroma_height(), 8);
    }

    #[test]
    fn allocation_yuv422() {
        let p = Picture::new(32, 16, 2, 8, 8);
        assert_eq!(p.luma_values().len(), 32 * 16);
        assert_eq!(p.cb_values().len(), 16 * 16);
        assert_eq!(p.cr_values().len(), 16 * 16);
        assert_eq!(p.chroma_width(), 16);
        assert_eq!(p.chroma_height(), 16);
    }

    #[test]
    fn allocation_yuv444() {
        let p = Picture::new(32, 16, 3, 10, 10);
        assert_eq!(p.luma_values().len(), 32 * 16);
        assert_eq!(p.cb_values().len(), 32 * 16);
        assert_eq!(p.cr_values().len(), 32 * 16);
        assert_eq!(p.chroma_width(), 32);
        assert_eq!(p.chroma_height(), 16);
    }

    #[test]
    fn freeze_8bit_uses_plain_image_planes_without_precision_side_channel() {
        let mut p = Picture::new(2, 2, 1, 8, 8);
        p.set_luma(0, 0, 123);
        p.set_cb(0, 0, 45);
        p.set_cr(0, 0, 67);

        let frame = p.freeze(Some(7)).expect("freeze 8-bit picture");
        assert_eq!(frame.header().pixel_format, PixelFormat::Yuv420P);
        assert_eq!(frame.header().presentation_timestamp, Some(7));
        assert_eq!(frame.header().significant_bits(), None);
        assert_eq!(frame.plane_count(), 3);
        assert_eq!(frame.plane(0).unwrap()[0], 123);
        assert_eq!(frame.plane(1).unwrap()[0], 45);
        assert_eq!(frame.plane(2).unwrap()[0], 67);
    }

    #[test]
    fn freeze_9bit_refines_10bit_container_with_exact_significant_bits() {
        let mut p = Picture::new(2, 2, 1, 9, 9);
        p.set_luma(0, 0, 0x101);
        p.set_cb(0, 0, 0x1ff);
        p.set_cr(0, 0, 0x155);

        let frame = p.freeze(None).expect("freeze 9-bit picture");
        assert_eq!(frame.header().pixel_format, PixelFormat::Yuv420P10Le);
        assert_eq!(frame.header().significant_bits(), Some(&[9, 9, 9][..]));
        assert_eq!(frame.plane_count(), 3);
        let y = frame.plane(0).unwrap();
        let cb = frame.plane(1).unwrap();
        let cr = frame.plane(2).unwrap();
        assert_eq!(u16::from_le_bytes([y[0], y[1]]), 0x101);
        assert_eq!(u16::from_le_bytes([cb[0], cb[1]]), 0x1ff);
        assert_eq!(u16::from_le_bytes([cr[0], cr[1]]), 0x155);
    }

    #[test]
    fn luma_set_and_get() {
        let mut p = Picture::new(16, 16, 1, 8, 8);
        p.set_luma(3, 4, 123);
        assert_eq!(p.luma_at(3, 4), 123);
        // Other samples untouched.
        assert_eq!(p.luma_at(0, 0), 0);
    }

    #[test]
    fn luma_edge_clamp() {
        let mut p = Picture::new(4, 4, 1, 8, 8);
        // Corner plant so we can detect the clamp.
        p.set_luma(0, 0, 42);
        p.set_luma(3, 3, 77);
        // Off the top-left clamps to (0, 0).
        assert_eq!(p.luma_at(-1, -1), 42);
        // Off the bottom-right clamps to (3, 3).
        assert_eq!(p.luma_at(10, 10), 77);
        // Off the top clamps to row 0.
        p.set_luma(2, 0, 55);
        assert_eq!(p.luma_at(2, -3), 55);
    }

    #[test]
    fn chroma_set_and_get() {
        let mut p = Picture::new(16, 16, 1, 8, 8);
        p.set_cb(2, 3, 10);
        p.set_cr(4, 5, 20);
        assert_eq!(p.cb_at(2, 3), 10);
        assert_eq!(p.cr_at(4, 5), 20);
    }

    #[test]
    fn chroma_edge_clamp() {
        let mut p = Picture::new(8, 8, 1, 8, 8);
        p.set_cb(0, 0, 1);
        p.set_cb(3, 3, 2);
        assert_eq!(p.cb_at(-1, -1), 1);
        assert_eq!(p.cb_at(5, 5), 2);
    }

    #[test]
    fn set_out_of_bounds_silently_ignored() {
        let mut p = Picture::new(4, 4, 1, 8, 8);
        // Should not panic.
        p.set_luma(100, 100, 42);
        p.set_luma(-1, -1, 42);
        assert_eq!(p.luma_values(), vec![0; 16]);
    }
}
