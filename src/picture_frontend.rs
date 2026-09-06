//! Optional shared H.264 picture parsing and reference-picture state.
//!
//! Hardware APIs such as VDPAU require the application to supply parsed
//! SPS/PPS/slice metadata, POC values and the live reference-picture set for
//! every decode submission. Other backends (notably NVIDIA NVDEC via
//! `cuvidParser`) already provide this state themselves and should not use this
//! helper. [`H264PictureFrontend`] is therefore an explicitly opt-in service,
//! not a mandatory decoder layer.

use oxideav_core::{Error, Result};

use crate::decoder::{Decoder as H264Parser, Event};
use crate::poc::{derive_poc, PocResult, PocSlice, PocSps, PocState};
use crate::pps::Pps;
use crate::ref_list::{perform_marking, DpbEntry, MmcoOp as RefMmcoOp, PicStructure, RefMarking};
use crate::slice_header::{DecRefPicMarking, MmcoOp as SliceMmcoOp, SliceHeader};
use crate::sps::Sps;

/// One complete primary coded picture, parsed and prepared for a hardware
/// backend but not yet committed to the reference-picture state.
#[derive(Debug, Clone)]
pub struct PreparedH264Picture {
    pub nal_unit_type: u8,
    pub nal_ref_idc: u8,
    pub header: SliceHeader,
    pub sps: Sps,
    pub pps: Pps,
    pub slice_count: u32,
    pub poc: PocResult,
    /// DPB snapshot to expose to the hardware decode call.
    pub references: Vec<DpbEntry>,
    /// True when this IDR starts a fresh reference-picture sequence.
    pub reference_reset: bool,
    /// IDR `no_output_of_prior_pics_flag`; output scheduling remains a
    /// backend/player concern, but the frontend surfaces the signal.
    pub no_output_of_prior_pics: bool,

    next_poc_state: PocState,
    marking: Option<DecRefPicMarking>,
}

impl PreparedH264Picture {
    #[must_use]
    pub fn is_idr(&self) -> bool {
        self.nal_unit_type == 5
    }

    #[must_use]
    pub fn is_reference(&self) -> bool {
        self.nal_ref_idc != 0
    }
}

/// Reference-store changes produced when a successfully decoded picture is
/// committed to [`H264PictureFrontend`]. Storage keys are deliberately opaque:
/// a hardware backend can map them to VDPAU/VA-API/Vulkan surfaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PictureCommit {
    /// Storage key assigned to the current picture when it is a reference.
    pub current_dpb_key: Option<u32>,
    /// Previously-live keys whose pictures became unused for reference.
    pub dead_dpb_keys: Vec<u32>,
    /// §8.2.5.4 MMCO-5 occurred; output-order state should reset after the
    /// pre-reset pictures have been drained.
    pub mmco5: bool,
}

/// Stateful shared H.264 parser + POC + reference-picture frontend.
#[derive(Default)]
pub struct H264PictureFrontend {
    parser: H264Parser,
    poc_state: PocState,
    prev_had_mmco5: bool,
    prev_reference_top_foc: i32,
    dpb: Vec<DpbEntry>,
    next_dpb_key: u32,
    saw_picture: bool,
}

impl H264PictureFrontend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse one complete Annex-B access unit and derive its POC/reference
    /// context. Parameter-set-only units return `Ok(None)`.
    ///
    /// The returned picture is transactional with respect to POC/DPB state:
    /// call [`commit`](Self::commit) only after the backend has successfully
    /// decoded the picture. SPS/PPS parser caches are updated while parsing.
    pub fn prepare_access_unit(&mut self, data: &[u8]) -> Result<Option<PreparedH264Picture>> {
        let mut slices = Vec::new();
        for event in self.parser.process_annex_b(data) {
            match event.map_err(|e| Error::invalid(format!("H.264 parse failed: {e}")))? {
                Event::Slice {
                    nal_unit_type,
                    nal_ref_idc,
                    header,
                    sps,
                    pps,
                    ..
                } => slices.push((nal_unit_type, nal_ref_idc, header, sps, pps)),
                Event::SliceDataPartitionA { .. }
                | Event::SliceDataPartitionBc { .. }
                | Event::SliceExtension { .. } => {
                    return Err(Error::unsupported(
                        "shared H.264 picture frontend does not yet support data partitioning/MVC slice extensions",
                    ));
                }
                _ => {}
            }
        }

        if slices.is_empty() {
            return Ok(None);
        }
        validate_same_picture(&slices)?;

        let (nal_unit_type, nal_ref_idc, header, sps, pps) = slices[0].clone();
        let is_idr = nal_unit_type == 5;
        let marking = header.dec_ref_pic_marking.clone();
        let no_output_of_prior_pics = marking
            .as_ref()
            .map(|m| m.no_output_of_prior_pics_flag)
            .unwrap_or(false);
        let reference_reset = is_idr && self.saw_picture;

        // IDR derives against an empty reference context. Keep the live DPB
        // untouched until commit so a hardware submission failure does not
        // partially mutate reference state.
        let references = if is_idr { Vec::new() } else { self.dpb.clone() };

        let prev_had_mmco5 = if is_idr { false } else { self.prev_had_mmco5 };
        let prev_reference_top_foc = if is_idr {
            0
        } else {
            self.prev_reference_top_foc
        };
        let mut next_poc_state = if is_idr {
            PocState::default()
        } else {
            self.poc_state.clone()
        };
        let poc = derive_poc(
            &make_poc_sps(&sps),
            &PocSlice {
                is_reference: nal_ref_idc != 0,
                is_idr,
                frame_num: header.frame_num,
                field_pic_flag: header.field_pic_flag,
                bottom_field_flag: header.bottom_field_flag,
                pic_order_cnt_lsb: header.pic_order_cnt_lsb,
                delta_pic_order_cnt_bottom: header.delta_pic_order_cnt_bottom,
                delta_pic_order_cnt: header.delta_pic_order_cnt,
                prev_had_mmco5,
                prev_reference_top_foc_for_mmco5: prev_reference_top_foc,
            },
            &mut next_poc_state,
        )
        .map_err(|e| Error::invalid(format!("H.264 POC derivation failed: {e}")))?;

        Ok(Some(PreparedH264Picture {
            nal_unit_type,
            nal_ref_idc,
            header,
            sps,
            pps,
            slice_count: slices.len() as u32,
            poc,
            references,
            reference_reset,
            no_output_of_prior_pics,
            next_poc_state,
            marking,
        }))
    }

    /// Commit a picture after successful backend reconstruction.
    pub fn commit(&mut self, picture: PreparedH264Picture) -> PictureCommit {
        // Capture storage keys before an IDR clears the logical DPB so the
        // backend can release the corresponding hardware surfaces.
        let old_keys: Vec<u32> = self.dpb.iter().map(|e| e.dpb_key).collect();
        if picture.is_idr() {
            self.dpb.clear();
            self.prev_had_mmco5 = false;
            self.prev_reference_top_foc = 0;
        }
        self.poc_state = picture.next_poc_state.clone();

        let mut current_dpb_key = None;
        let mut mmco5 = false;

        if picture.is_reference() {
            let key = self.next_dpb_key;
            self.next_dpb_key = self.next_dpb_key.wrapping_add(1);
            let mut current = DpbEntry {
                frame_num: picture.header.frame_num,
                top_field_order_cnt: picture.poc.top_field_order_cnt,
                bottom_field_order_cnt: picture.poc.bottom_field_order_cnt,
                pic_order_cnt: picture.poc.pic_order_cnt,
                structure: PicStructure::Frame,
                marking: RefMarking::Unused,
                long_term_frame_idx: 0,
                dpb_key: key,
                field_markings: [RefMarking::Unused; 2],
            };
            let converted_ops = picture
                .marking
                .as_ref()
                .and_then(|m| m.adaptive_marking.as_ref())
                .map(|ops| ops.iter().map(convert_mmco).collect::<Vec<_>>());
            mmco5 = perform_marking(
                &mut self.dpb,
                &mut current,
                picture.sps.max_num_ref_frames,
                picture.is_idr(),
                picture
                    .marking
                    .as_ref()
                    .map(|m| m.long_term_reference_flag)
                    .unwrap_or(false),
                picture.no_output_of_prior_pics,
                converted_ops.as_deref(),
                picture.header.frame_num,
                picture.sps.max_frame_num(),
            );

            self.dpb.retain(|e| e.marking != RefMarking::Unused);
            self.dpb.push(current);
            current_dpb_key = Some(key);

            if mmco5 {
                self.prev_had_mmco5 = true;
                self.prev_reference_top_foc = picture.poc.top_field_order_cnt;
            } else {
                self.prev_had_mmco5 = false;
                self.prev_reference_top_foc = 0;
            }
        } else {
            // The MMCO-5 hint is consumed by exactly the next picture.
            self.prev_had_mmco5 = false;
            self.prev_reference_top_foc = 0;
        }

        let live_keys: std::collections::HashSet<u32> =
            self.dpb.iter().map(|e| e.dpb_key).collect();
        let dead_dpb_keys = old_keys
            .into_iter()
            .filter(|key| !live_keys.contains(key))
            .collect();
        self.saw_picture = true;

        PictureCommit {
            current_dpb_key,
            dead_dpb_keys,
            mmco5,
        }
    }

    /// Reset parser, POC and DPB state (e.g. after seek).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Current live reference-picture descriptors.
    #[must_use]
    pub fn references(&self) -> &[DpbEntry] {
        &self.dpb
    }
}

fn validate_same_picture(slices: &[(u8, u8, SliceHeader, Sps, Pps)]) -> Result<()> {
    let first = &slices[0];
    for slice in &slices[1..] {
        if slice.0 != first.0
            || slice.1 != first.1
            || slice.2.frame_num != first.2.frame_num
            || slice.2.pic_parameter_set_id != first.2.pic_parameter_set_id
            || slice.2.pic_order_cnt_lsb != first.2.pic_order_cnt_lsb
            || slice.2.delta_pic_order_cnt_bottom != first.2.delta_pic_order_cnt_bottom
            || slice.2.delta_pic_order_cnt != first.2.delta_pic_order_cnt
            || slice.2.field_pic_flag != first.2.field_pic_flag
            || slice.2.bottom_field_flag != first.2.bottom_field_flag
            || slice.2.idr_pic_id != first.2.idr_pic_id
        {
            return Err(Error::invalid(
                "H.264 access unit contains slices from more than one primary coded picture",
            ));
        }
    }
    Ok(())
}

fn make_poc_sps(sps: &Sps) -> PocSps {
    PocSps {
        pic_order_cnt_type: sps.pic_order_cnt_type,
        log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        delta_pic_order_always_zero_flag: sps.delta_pic_order_always_zero_flag,
        offset_for_non_ref_pic: sps.offset_for_non_ref_pic,
        offset_for_top_to_bottom_field: sps.offset_for_top_to_bottom_field,
        num_ref_frames_in_pic_order_cnt_cycle: sps.num_ref_frames_in_pic_order_cnt_cycle,
        offset_for_ref_frame: sps.offset_for_ref_frame.clone(),
        frame_mbs_only_flag: sps.frame_mbs_only_flag,
    }
}

fn convert_mmco(op: &SliceMmcoOp) -> RefMmcoOp {
    match *op {
        SliceMmcoOp::MarkShortTermUnused(v) => RefMmcoOp::MarkShortTermUnused(v),
        SliceMmcoOp::MarkLongTermUnused(v) => RefMmcoOp::MarkLongTermUnused(v),
        SliceMmcoOp::AssignLongTerm(d, idx) => RefMmcoOp::AssignLongTerm(d, idx),
        SliceMmcoOp::SetMaxLongTermIdx(v) => RefMmcoOp::SetMaxLongTermIdx(v),
        SliceMmcoOp::MarkAllUnused => RefMmcoOp::MarkAllUnused,
        SliceMmcoOp::AssignCurrentLongTerm(v) => RefMmcoOp::AssignCurrentLongTerm(v),
    }
}
