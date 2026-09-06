//! Optional shared H.264 picture parsing and reference-picture state.
//!
//! Hardware APIs such as VDPAU require the application to supply parsed
//! SPS/PPS/slice metadata, POC values and the live reference-picture set for
//! every decode submission. Other backends (notably NVIDIA NVDEC via
//! `cuvidParser`) already provide this state themselves and should not use this
//! helper. [`H264PictureFrontend`] is therefore an explicitly opt-in service,
//! not a mandatory decoder layer.
//!
//! The frontend can either parse a complete Annex-B access unit itself via
//! [`H264PictureFrontend::prepare_access_unit`], or accept a picture header that
//! a backend has already parsed via [`H264PictureFrontend::prepare_parsed_picture`].
//! The latter lets the pure-Rust decoder reuse its existing rich slice parser
//! (data partitions, separate-colour-plane routing, etc.) while sharing the
//! cross-picture POC/DPB/MMCO machinery with hardware backends.

use std::collections::HashSet;

use oxideav_core::{Error, Result};

use crate::decoder::{Decoder as H264Parser, Event};
use crate::poc::{derive_poc, PocResult, PocSlice, PocSps, PocState};
use crate::pps::Pps;
use crate::ref_list::{
    perform_marking, sliding_window_marking, DpbEntry, MmcoOp as RefMmcoOp, PicStructure,
    RefMarking,
};
use crate::slice_header::{DecRefPicMarking, MmcoOp as SliceMmcoOp, SliceHeader};
use crate::sps::Sps;

/// One complete primary coded picture, parsed and prepared for a decoder
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
    pub structure: PicStructure,
    /// DPB snapshot that must be used while decoding this picture. For streams
    /// with allowed `frame_num` gaps this includes the synthetic non-existing
    /// reference entries required by §8.2.5.2.
    pub references: Vec<DpbEntry>,
    /// Synthetic §8.2.5.2 entries newly introduced while preparing this
    /// picture and still live in `references`. Software decoders can install
    /// neutral placeholder samples for these opaque keys; hardware backends
    /// may reject them if they cannot represent non-existing references.
    pub synthetic_references: Vec<DpbEntry>,
    /// True when this IDR starts a fresh reference-picture sequence.
    pub reference_reset: bool,
    /// IDR `no_output_of_prior_pics_flag`; output scheduling remains a
    /// backend/player concern, but the frontend surfaces the signal.
    pub no_output_of_prior_pics: bool,

    base_dpb: Vec<DpbEntry>,
    next_poc_state: PocState,
    next_dpb_key_after_gaps: u32,
    prev_ref_frame_num_after_gaps: Option<u32>,
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
/// a hardware backend can map them to VDPAU/VA-API/Vulkan surfaces and a
/// software backend can map them to reconstructed sample buffers.
#[derive(Debug, Clone)]
pub struct PictureCommit {
    /// Storage key assigned to the current picture when it is a reference.
    pub current_dpb_key: Option<u32>,
    /// Complete committed DPB descriptor for the current picture, including
    /// post-MMCO5 frame_num/POC identity when applicable.
    pub current_dpb_entry: Option<DpbEntry>,
    /// Previously-live keys whose pictures became unused for reference. This
    /// includes evictions caused by synthetic gap entries and by current-picture
    /// sliding-window/MMCO marking.
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
    prev_ref_frame_num: Option<u32>,
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
                        "shared H.264 Annex-B picture parser does not yet assemble data partitions/MVC slice extensions; use prepare_parsed_picture after backend-specific parsing",
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
        self.prepare_parsed_picture(
            nal_unit_type,
            nal_ref_idc,
            header,
            sps,
            pps,
            slices.len() as u32,
        )
        .map(Some)
    }

    /// Prepare one already-parsed primary coded picture for decode.
    ///
    /// This is the shared-state entry point for decoders that need richer NAL /
    /// slice handling than [`prepare_access_unit`](Self::prepare_access_unit)
    /// exposes. The caller retains ownership of slice-data parsing and pixel
    /// reconstruction; this frontend supplies the cross-picture state only.
    ///
    /// Preparation is transactional: POC, gap and DPB changes are simulated on
    /// cloned state. They become live only if [`commit`](Self::commit) is called.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_parsed_picture(
        &self,
        nal_unit_type: u8,
        nal_ref_idc: u8,
        header: SliceHeader,
        sps: Sps,
        pps: Pps,
        slice_count: u32,
    ) -> Result<PreparedH264Picture> {
        let is_idr = nal_unit_type == 5;
        let marking = header.dec_ref_pic_marking.clone();
        let no_output_of_prior_pics = marking
            .as_ref()
            .map(|m| m.no_output_of_prior_pics_flag)
            .unwrap_or(false);
        let reference_reset = is_idr && self.saw_picture;

        let mut base_dpb = if is_idr { Vec::new() } else { self.dpb.clone() };
        let mut next_poc_state = if is_idr {
            PocState::default()
        } else {
            self.poc_state.clone()
        };
        let mut next_dpb_key = self.next_dpb_key;
        let mut prev_ref_frame_num = if is_idr {
            None
        } else {
            self.prev_ref_frame_num
        };

        let first_gap_key = next_dpb_key;
        if !is_idr {
            simulate_frame_num_gap(
                &sps,
                header.frame_num,
                &mut base_dpb,
                &mut next_poc_state,
                &mut next_dpb_key,
                &mut prev_ref_frame_num,
            )?;
        }
        let synthetic_references = base_dpb
            .iter()
            .filter(|e| e.dpb_key >= first_gap_key && e.dpb_key < next_dpb_key)
            .cloned()
            .collect();

        let prev_had_mmco5 = if is_idr { false } else { self.prev_had_mmco5 };
        let prev_reference_top_foc = if is_idr {
            0
        } else {
            self.prev_reference_top_foc
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

        let references = base_dpb.clone();
        Ok(PreparedH264Picture {
            nal_unit_type,
            nal_ref_idc,
            structure: pic_structure_from_flags(header.field_pic_flag, header.bottom_field_flag),
            header,
            sps,
            pps,
            slice_count,
            poc,
            references,
            synthetic_references,
            reference_reset,
            no_output_of_prior_pics,
            base_dpb,
            next_poc_state,
            next_dpb_key_after_gaps: next_dpb_key,
            prev_ref_frame_num_after_gaps: prev_ref_frame_num,
            marking,
        })
    }

    /// Commit a picture after successful backend reconstruction.
    pub fn commit(&mut self, picture: PreparedH264Picture) -> PictureCommit {
        let old_keys: Vec<u32> = self.dpb.iter().map(|e| e.dpb_key).collect();

        self.dpb = picture.base_dpb.clone();
        self.poc_state = picture.next_poc_state.clone();
        self.next_dpb_key = picture.next_dpb_key_after_gaps;
        self.prev_ref_frame_num = picture.prev_ref_frame_num_after_gaps;
        if picture.is_idr() {
            self.prev_had_mmco5 = false;
            self.prev_reference_top_foc = 0;
        }

        let mut current_dpb_entry = None;
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
                structure: picture.structure,
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

            self.dpb.retain(DpbEntry::is_any_field_ref);

            if mmco5 {
                let temp = current.pic_order_cnt;
                current.frame_num = 0;
                current.top_field_order_cnt -= temp;
                current.bottom_field_order_cnt -= temp;
                current.pic_order_cnt = if current.structure.is_field() {
                    if current.structure.is_bottom() {
                        current.bottom_field_order_cnt
                    } else {
                        current.top_field_order_cnt
                    }
                } else {
                    current
                        .top_field_order_cnt
                        .min(current.bottom_field_order_cnt)
                };
            }

            self.dpb.push(current.clone());
            current_dpb_key = Some(key);
            current_dpb_entry = Some(current);

            if mmco5 {
                self.prev_had_mmco5 = true;
                self.prev_reference_top_foc =
                    picture.poc.top_field_order_cnt - picture.poc.pic_order_cnt;
                self.prev_ref_frame_num = Some(0);
            } else {
                self.prev_had_mmco5 = false;
                self.prev_reference_top_foc = 0;
                self.prev_ref_frame_num = Some(picture.header.frame_num);
            }
        } else {
            self.prev_had_mmco5 = false;
            self.prev_reference_top_foc = 0;
        }

        let live_keys: HashSet<u32> = self.dpb.iter().map(|e| e.dpb_key).collect();
        let dead_dpb_keys = old_keys
            .into_iter()
            .filter(|key| !live_keys.contains(key))
            .collect();
        self.saw_picture = true;

        PictureCommit {
            current_dpb_key,
            current_dpb_entry,
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
        if (slice.0 == 5) != (first.0 == 5)
            || (slice.1 == 0) != (first.1 == 0)
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

fn pic_structure_from_flags(field_pic_flag: bool, bottom_field_flag: bool) -> PicStructure {
    if !field_pic_flag {
        PicStructure::Frame
    } else if bottom_field_flag {
        PicStructure::BottomField
    } else {
        PicStructure::TopField
    }
}

fn simulate_frame_num_gap(
    sps: &Sps,
    current_frame_num: u32,
    dpb: &mut Vec<DpbEntry>,
    poc_state: &mut PocState,
    next_dpb_key: &mut u32,
    prev_ref_frame_num: &mut Option<u32>,
) -> Result<()> {
    let Some(prev) = *prev_ref_frame_num else {
        return Ok(());
    };
    let max_frame_num = sps.max_frame_num();
    let expected = (prev + 1) % max_frame_num;

    if current_frame_num == prev || current_frame_num == expected {
        return Ok(());
    }

    if !sps.gaps_in_frame_num_value_allowed_flag {
        return Err(Error::invalid(format!(
            "H.264 frame_num {} after PrevRefFrameNum {} requires {} when gaps_in_frame_num_value_allowed_flag is 0",
            current_frame_num, prev, expected
        )));
    }

    let mut missing = expected;
    let mut iterations = 0u32;
    while missing != current_frame_num && iterations < max_frame_num {
        if matches!(sps.pic_order_cnt_type, 1 | 2) {
            let prev_offset = poc_state.prev_frame_num_offset;
            if poc_state.prev_frame_num > missing {
                poc_state.prev_frame_num_offset = prev_offset + i64::from(max_frame_num);
            }
        }
        poc_state.prev_frame_num = missing;
        let (top_foc, bottom_foc, poc) = non_existing_poc(sps, poc_state, missing);

        sliding_window_marking(dpb, sps.max_num_ref_frames, missing, max_frame_num, None);
        dpb.retain(DpbEntry::is_any_field_ref);

        let key = *next_dpb_key;
        *next_dpb_key = next_dpb_key.wrapping_add(1);
        let mut entry = DpbEntry {
            frame_num: missing,
            top_field_order_cnt: top_foc,
            bottom_field_order_cnt: bottom_foc,
            pic_order_cnt: poc,
            structure: PicStructure::Frame,
            marking: RefMarking::ShortTerm,
            long_term_frame_idx: 0,
            dpb_key: key,
            field_markings: [RefMarking::Unused; 2],
        };
        entry.sync_field_markings();
        dpb.push(entry);
        *prev_ref_frame_num = Some(missing);

        missing = (missing + 1) % max_frame_num;
        iterations += 1;
    }

    if missing != current_frame_num {
        return Err(Error::invalid(
            "H.264 frame_num gap simulation exceeded MaxFrameNum iterations",
        ));
    }
    Ok(())
}

fn non_existing_poc(sps: &Sps, state: &PocState, frame_num: u32) -> (i32, i32, i32) {
    match sps.pic_order_cnt_type {
        2 => {
            let poc = 2 * (state.prev_frame_num_offset + i64::from(frame_num));
            let poc = poc.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
            (poc, poc, poc)
        }
        1 => (0, 0, 0),
        _ => (0, 0, 0),
    }
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
