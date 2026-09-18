//! H.264 decoder scaffold exposed through the `oxideav_core::Decoder`
//! trait so containers + players can route packets at us.
//!
//! What this scaffold currently does:
//! 1. Accepts Annex B byte-stream packets or AVCC-framed packets
//!    (length prefix size taken from `extradata` when present — an
//!    `AVCDecoderConfigurationRecord` per ISO/IEC 14496-15).
//! 2. Walks NAL units through [`crate::decoder::Decoder`] which
//!    captures SPS/PPS and emits parsed slice headers.
//! 3. Maintains a decoded picture store ([`crate::ref_store::RefPicStore`])
//!    across pictures so P/B slices have reference pictures for
//!    motion compensation (§8.2.4 / §8.2.5).
//! 4. For every slice it parses, the decoder derives POC (§8.2.1),
//!    builds the RefPicList0 / RefPicList1 that §8.4.2 inter
//!    prediction consumes, runs `reconstruct::reconstruct_slice`, and
//!    queues the reconstructed Picture as a `VideoFrame` for
//!    `receive_frame`.
//!
//! Known simplifications (see inline comments for details):
//! - **Access unit assembly**: §7.4.1.2.4 multi-slice assembly IS
//!   implemented — continuation slices land in the same Picture +
//!   MbGrid as the first slice, and we finalize the picture (push to
//!   DPB + output queue) when a slice opens a new primary coded picture
//!   or on AUD / EndOfSequence / flush. Caveat: the deblocking pass in
//!   `reconstruct::reconstruct_slice` runs per-slice and walks the
//!   whole grid, so continuation slices may re-filter earlier slices'
//!   interior edges. See the comment on `handle_slice` for details.
//! - **Output ordering**: frames are emitted in display (POC) order
//!   via [`crate::dpb_output::DpbOutput`] per Annex C §C.2.2 / §C.4
//!   bumping process.
//! - **Field pictures (PAFF) / MBAFF**: PAFF field pictures
//!   (`field_pic_flag == 1`) decode as half-height pictures and the
//!   §C.4.4 driver pairs complementary opposite-parity fields into one
//!   full-height output frame. MBAFF (`mb_adaptive_frame_field_flag ==
//!   1`, `field_pic_flag == 0`) intra reconstructs; MBAFF inter (P/B)
//!   field-coded pairs are still deferred.
//! - **Reference picture list modification (RPLM)**: the ops are
//!   applied via `ref_list::modify_ref_pic_list`, but the underlying
//!   short-term / long-term derivation is a first pass and may not
//!   be fully spec-accurate for all streams.
//!
//! The trait is what oxideplay/oxideav-pipeline consumes; registering
//! this decoder (via [`crate::register`]) stops the "codec not found"
//! error on the first h264 packet.

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use oxideav_core::arena::sync::{Arena, ArenaIdentity, ArenaPool, FrameHeader, VideoFrameBuilder};
use oxideav_core::Decoder;
use oxideav_core::{
    CancellationToken, CodecId, CodecParameters, Error, Frame, FrameLease, Packet, PixelFormat,
    Result, TimeBase, VideoColorInfo, VideoColorRange, VideoMatrixCoefficients,
};

use crate::access_unit::AnnexBAccessUnitAssembler;
use crate::decoder::{Decoder as H264Driver, Event};
use crate::dpb_output::{DpbOutput, OutputEntry};
use crate::mb_grid::MbGrid;
use crate::picture::Picture;
use crate::picture_frontend::{H264PictureFrontend, PreparedH264Picture};
use crate::poc::PocResult;
use crate::ref_list::{self, DpbEntry, PicStructure, RplmOp};
use crate::ref_store::{RefPicProvider, RefPicStore};
use crate::slice_header::{RefPicListModificationOp as SliceRplmOp, SliceHeader, SliceType};
use crate::sps::Sps;
use crate::vui::VideoSignalType;
use crate::{reconstruct, slice_data};

#[cfg(test)]
use oxideav_core::{VideoFrame, VideoPlane};

/// §7.4.1.2 / §7.4.1.2.4 — state carried forward across slices that
/// belong to the *same* primary coded picture.
///
/// Once a slice with first_mb_in_slice == 0 (and/or an AUD) opens a new
/// primary coded picture, a `PictureInProgress` is allocated. Each
/// subsequent slice that passes the §7.4.1.2.4 "same picture" test is
/// reconstructed into the *same* `pic` / `grid` so its macroblocks are
/// laid down alongside the earlier slices'. When a slice fails the test
/// (new primary coded picture) or an AUD / flush fires, the in-progress
/// picture is finalized (pushed into the DPB and output queue) and a
/// fresh one is started from the triggering slice.
struct PictureInProgress {
    /// Shared cross-picture state prepared transactionally from the first slice.
    prepared: PreparedH264Picture,
    /// Reconstructed samples.
    pic: Picture,
    /// MB metadata for the assembled picture. Carries §6.4.11 availability
    /// plus per-MB QP/CBP/etc. across slice boundaries so continuation
    /// slices see prior slices' MBs as neighbours during intra prediction
    /// and deblocking.
    grid: MbGrid,
    /// The `(nal_unit_type, nal_ref_idc, header)` of the *first* slice of
    /// this picture — the identity used for §7.4.1.2.4 comparisons against
    /// subsequent slices.
    first_nal_unit_type: u8,
    first_nal_ref_idc: u8,
    first_header: SliceHeader,
    /// True if any slice in the picture so far was a reference slice. A
    /// picture is a reference picture if *any* of its VCL NALs carries
    /// nal_ref_idc != 0 — §7.4.1.2.1 / §7.4.1.2.4 require all slices to
    /// share the zero-ness of nal_ref_idc, so this is effectively the
    /// first slice's is_reference bit, but we OR it to be defensive.
    is_reference: bool,
    /// True if this is an IDR picture (any slice has nal_unit_type == 5).
    is_idr: bool,
    /// §8.2.1 POC result derived at the first slice. All slices of the
    /// same picture share the same POC per §7.4.1.2.4.
    poc: PocResult,
    /// §7.4.3 picture structure for DPB bookkeeping.
    structure: PicStructure,
    /// Packet pts to stamp onto the finalized VideoFrame.
    pts: Option<i64>,
    /// Packet time_base for rescaling downstream.
    time_base: TimeBase,
    /// §8.7 deblocking state from the *first* slice of the picture.
    /// Multi-slice pictures that vary deblocking_filter_idc / alpha_off /
    /// beta_off per slice are a known simplification — the JVT
    /// conformance streams we target encode uniform per-picture
    /// deblock offsets. Kept so `finalize_in_progress_picture` can run
    /// the §8.7 pass exactly once, after every slice has populated the
    /// shared Picture + MbGrid.
    deblock_enabled: bool,
    deblock_alpha_off: i32,
    deblock_beta_off: i32,
    /// §7.4.4 — per-MB `mb_field_decoding_flag` aggregated across every
    /// slice of the picture, indexed by picture-level macroblock
    /// address. `slice_data.mb_field_decoding_flags` is slice-local so
    /// we copy it into this picture-wide vector using the raw
    /// CurrMbAddr walk the slice data parser produced.
    mb_field_flags: Vec<bool>,
    /// §7.4.1.2.1 — SPS + PPS snapshots captured at the first slice's
    /// header-parse time. Used by `finalize_in_progress_picture` for
    /// deblocking instead of reading the driver's current "active"
    /// parameter sets, which may have been overwritten by a later PPS
    /// NAL carrying the same id but different scaling / qp-offset
    /// values (JVT CACQP3 exercises this path).
    sps: Sps,
    pps: crate::pps::Pps,
    /// True iff at least one slice of this picture has been
    /// successfully reconstructed. When `false` at finalize time the
    /// picture is dropped instead of being pushed into the DPB +
    /// output queue: emitting a never-painted picture (zeroed luma /
    /// chroma plus garbage residue from neighbour MBs) is a
    /// strictness divergence from common H.264 decoders, which reject the
    /// access unit outright when every slice fails CABAC / CAVLC
    /// parse. Caught by fuzz target `ffmpeg_oracle_decode` on
    /// crash-2ad9589f… (3 slices, all fail "read past end of
    /// bitstream") — see commit message for details.
    any_slice_succeeded: bool,
}

/// §C.4.4 — a decoded PAFF field awaiting its complementary field so the
/// pair can be re-interleaved into a full-height output frame. Held
/// between the finalization of the first field of a complementary pair
/// and the arrival of the second field (opposite parity, same
/// access-unit `frame_num`).
struct PendingField {
    /// Reconstructed half-height field samples (field rows only).
    pic: Picture,
    /// `true` for a bottom field (the field occupies the odd output
    /// rows), `false` for a top field (even output rows).
    is_bottom: bool,
    /// `frame_num` of the field — a complementary pair shares it.
    frame_num: u32,
    /// The field's own PicOrderCnt (Top/BottomFieldOrderCnt). The frame's
    /// output POC is the minimum of the pair's two field POCs.
    field_poc: i32,
    /// Packet pts carried by whichever field opened the access unit.
    pts: Option<i64>,
}

/// §8.1 — separate-colour-plane decode state (round 448). When the
/// active SPS carries `separate_colour_plane_flag == 1`, "the decoding
/// process is invoked three times: … the decoding process of NAL units
/// with a particular value of colour_plane_id is specified as if only
/// a coded video sequence with monochrome colour format with that
/// particular value of colour_plane_id would be present in the
/// bitstream" — so the driver literally keeps three monochrome
/// sub-decoders and routes every coded slice to the one selected by
/// its §7.4.3 `colour_plane_id`. Each sub-decoder runs the complete
/// monochrome pipeline (POC, reference marking, DPB, reconstruction,
/// §8.7 deblocking, §C.4 output ordering) on its own plane; the
/// outputs are re-assembled into one three-plane picture per access
/// unit (plane 0 → S_L, 1 → S_Cb, 2 → S_Cr).
struct ScpState {
    subs: [Box<H264CodecDecoder>; 3],
    /// Per-plane decoded (monochrome) frames awaiting their two
    /// siblings. The three sub-decoders run identical §8.2.1 / §C.4
    /// machinery on identical slice-header fields, so their output
    /// streams pair 1:1 in emission order.
    queues: [VecDeque<FrameLease>; 3],
}

impl ScpState {
    fn new(codec_id: &CodecId, cancellation: Option<&CancellationToken>) -> Self {
        let mk = || {
            let mut d = H264CodecDecoder::new(codec_id.clone());
            d.scp_plane_mode = true;
            d.cancellation = cancellation.cloned();
            Box::new(d)
        };
        Self {
            subs: [mk(), mk(), mk()],
            queues: [VecDeque::new(), VecDeque::new(), VecDeque::new()],
        }
    }
}

/// Registry factory — called by the codec registry when a container
/// wants a decoder for H.264.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let mut dec = H264CodecDecoder::new(params.codec_id.clone());
    dec.output_params = params.clone();
    if !params.extradata.is_empty() {
        dec.consume_extradata(&params.extradata)?;
    }
    Ok(Box::new(dec))
}

/// Full per-slice decoder with DPB wiring.
/// §7.3.2.9 — a partition-A slice held while its partition-B/C
/// payloads arrive (the partitions of one slice are consecutive in
/// the NAL stream, §7.4.1.2.3). Flushed — parsed + reconstructed —
/// when any non-partition-B/C event follows, or at stream end.
struct PendingDpSlice {
    nal_ref_idc: u8,
    header: SliceHeader,
    rbsp_a: Vec<u8>,
    cursor_a: (usize, u8),
    slice_id: u32,
    sps: Sps,
    pps: crate::pps::Pps,
    part_b: Option<(Vec<u8>, (usize, u8))>,
    part_c: Option<(Vec<u8>, (usize, u8))>,
}

const H264_PICTURE_POOL_MAX_ARENAS: usize = 32;
const H264_PICTURE_ARENA_PADDING: usize = 3 * 64;

pub struct H264CodecDecoder {
    codec_id: CodecId,
    /// Decoder-discovered stream metadata published through `output_params()`.
    output_params: CodecParameters,
    /// NAL unit length-prefix size from `avcC`, when present. `None`
    /// means the input is treated as Annex B byte-stream.
    length_size: Option<u8>,
    driver: H264Driver,
    /// Last slice header we parsed — useful for probes / asserts.
    #[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
    pub last_slice: Option<SliceHeader>,
    /// Count of per-slice / per-picture reconstruction errors that were
    /// swallowed to keep the stream alive (the `h264 slice skipped: …`
    /// paths). Zero means every slice fed so far decoded cleanly; any
    /// frame emitted while this is non-zero may be partial/concealed
    /// output. Diagnostic only — see [`Self::decode_error_count`].
    decode_errors: u64,
    eof: bool,
    /// §C.2.2 / §C.4 — POC-ordered output DPB. The payload is an owned
    /// retainable lease so decoded CPU arenas can remain shared with the
    /// reference-picture store until the application releases them.
    output_dpb: DpbOutput<FrameLease>,
    /// Pictures that have already been "bumped" from the DPB and are waiting
    /// for the next receive call. The lease is preserved unchanged here.
    ready: VecDeque<FrameLease>,
    /// Packet-level pts passed on the most recent `send_packet`. We
    /// stamp the first frame produced from that packet with it.
    pending_pts: Option<i64>,
    /// §7.3.2.9 — in-flight data-partitioned slice (partition A held
    /// until its B/C payloads arrive).
    pending_dp: Option<PendingDpSlice>,
    /// Packet time_base so downstream consumers can rescale.
    pending_time_base: TimeBase,

    /// Long-lived reconstructed sample store keyed by shared DPB keys.
    ref_store: RefPicStore,
    /// Reusable final-format sample allocations for ordinary frame-coded
    /// software decode. The pool is recreated when the SPS changes geometry or
    /// storage width; old leases keep their former pool alive as needed.
    picture_pool: Arc<ArenaPool>,
    /// Reusable arena allocations for output that must be assembled from
    /// multiple decoded pictures (PAFF field pairs and separate colour planes).
    /// These paths copy during assembly but still expose native ArenaVideo leases.
    assembly_pool: Arc<ArenaPool>,
    /// Cooperative cancellation supplied by the pipeline. Blocking arena waits
    /// are only used when this token is available.
    cancellation: Option<CancellationToken>,
    /// Arena allocations owned by a containing decoder but retained outside this
    /// sub-decoder's own fields (currently separate-colour-plane queues). They
    /// count as decoder-internal for self-deadlock detection.
    extra_internal_arenas: HashSet<ArenaIdentity>,

    // ---- Shared H.264 byte/picture frontend --------------------------
    /// Optional Annex-B access-unit packetiser. AVCC retains its native
    /// length-prefixed packet path; Annex-B uses this helper so PES/container
    /// packet boundaries do not have to coincide with NAL/AU boundaries.
    au_assembler: AnnexBAccessUnitAssembler,
    /// Shared POC / DPB / MMCO / frame_num-gap state. Pixel samples remain in
    /// `ref_store`; this frontend owns only codec metadata and opaque keys.
    picture_frontend: H264PictureFrontend,

    /// §7.4.1.2 / §7.4.1.2.4 — picture currently being assembled across
    /// one-or-more slice NAL units. `None` means no slice of the current
    /// access unit has been processed yet (either we haven't started, or
    /// the last picture was just finalized). Populated by the first
    /// slice of a primary coded picture and consumed by `finalize_picture`
    /// when the picture boundary is detected.
    in_progress: Option<PictureInProgress>,

    /// §C.4.4 — the first decoded field of an as-yet-incomplete
    /// complementary field pair (PAFF). `None` when the decoder is not
    /// mid-pair. When the second field of the pair is finalized the two
    /// half-height field pictures are re-interleaved into one full-height
    /// output frame.
    pending_field: Option<PendingField>,

    /// §8.1 / §7.4.2.1.1 — separate-colour-plane routing state,
    /// created lazily at the first coded slice whose SPS carries
    /// `separate_colour_plane_flag == 1`. See [`ScpState`].
    scp: Option<Box<ScpState>>,
    /// True on the three [`ScpState`] sub-decoders: this instance
    /// decodes ONE colour plane of a `separate_colour_plane_flag == 1`
    /// stream as a monochrome picture (ChromaArrayType == 0) instead
    /// of routing — the recursion stop for the §8.1 three-invocation
    /// process.
    scp_plane_mode: bool,

    // ---- ISO/IEC 14496-15 §5.2.4.1.1 avcC diagnostic snapshot --------
    /// `AVCProfileIndication` from the last `consume_extradata` call.
    /// Optional because Annex B streams have no avcC.
    avcc_profile_idc: Option<u8>,
    /// `AVCLevelIndication` from the last `consume_extradata` call.
    avcc_level_idc: Option<u8>,
    /// `chroma_format` from the §5.2.4.1.1 High-profile extension
    /// (`0` = monochrome, `1` = 4:2:0, `2` = 4:2:2, `3` = 4:4:4).
    avcc_chroma_format: Option<u8>,
    /// `bit_depth_luma_minus8 + 8` — only populated when the avcC
    /// record carries the §5.2.4.1.1 High-profile extension.
    avcc_bit_depth_luma: Option<u8>,
    /// `bit_depth_chroma_minus8 + 8` — only populated when the avcC
    /// record carries the §5.2.4.1.1 High-profile extension.
    avcc_bit_depth_chroma: Option<u8>,
}

fn video_color_from_signal(signal: &VideoSignalType) -> VideoColorInfo {
    let (matrix, colour_primaries, transfer_characteristics) = signal
        .colour_description
        .as_ref()
        .map_or((None, None, None), |description| {
            let matrix = match description.matrix_coefficients {
                0 => VideoMatrixCoefficients::Identity,
                1 => VideoMatrixCoefficients::Bt709,
                2 => VideoMatrixCoefficients::Unspecified,
                4 => VideoMatrixCoefficients::Fcc,
                5 => VideoMatrixCoefficients::Bt470Bg,
                6 => VideoMatrixCoefficients::Smpte170M,
                7 => VideoMatrixCoefficients::Smpte240M,
                8 => VideoMatrixCoefficients::Ycgco,
                9 => VideoMatrixCoefficients::Bt2020Ncl,
                10 => VideoMatrixCoefficients::Bt2020Cl,
                value => VideoMatrixCoefficients::Unknown(value),
            };
            (
                Some(matrix),
                Some(description.colour_primaries),
                Some(description.transfer_characteristics),
            )
        });
    VideoColorInfo {
        range: Some(if signal.video_full_range_flag {
            VideoColorRange::Full
        } else {
            VideoColorRange::Limited
        }),
        matrix,
        colour_primaries,
        transfer_characteristics,
    }
}

fn video_color_from_sps(sps: &Sps) -> Option<VideoColorInfo> {
    sps.vui
        .as_ref()?
        .video_signal_type
        .as_ref()
        .map(video_color_from_signal)
}

impl H264CodecDecoder {
    pub fn new(codec_id: CodecId) -> Self {
        // Start with a "generous" placeholder sizing (16 frames, the
        // Annex A upper bound from §A.3.1 item h, `Min(…, 16)`). The
        // first slice updates the sizing in `ensure_output_dpb_sized`
        // from the active SPS.
        Self {
            output_params: CodecParameters::video(codec_id.clone()),
            codec_id,
            length_size: None,
            driver: H264Driver::new(),
            last_slice: None,
            decode_errors: 0,
            eof: false,
            output_dpb: DpbOutput::<FrameLease>::new(16, 16),
            ready: VecDeque::new(),
            pending_pts: None,
            pending_dp: None,
            pending_time_base: TimeBase::new(1, 1),
            ref_store: RefPicStore::new(),
            picture_pool: ArenaPool::new(H264_PICTURE_POOL_MAX_ARENAS, 1),
            assembly_pool: ArenaPool::new(H264_PICTURE_POOL_MAX_ARENAS, 1),
            cancellation: None,
            extra_internal_arenas: HashSet::new(),
            au_assembler: AnnexBAccessUnitAssembler::default(),
            picture_frontend: H264PictureFrontend::new(),
            in_progress: None,
            pending_field: None,
            scp: None,
            scp_plane_mode: false,
            avcc_profile_idc: None,
            avcc_level_idc: None,
            avcc_chroma_format: None,
            avcc_bit_depth_luma: None,
            avcc_bit_depth_chroma: None,
        }
    }

    /// §5.2.4.1.1 ISO/IEC 14496-15 — `AVCDecoderConfigurationRecord`.
    /// Picks up `lengthSizeMinusOne` and feeds the stored SPS + PPS NAL
    /// units through the driver. When `AVCProfileIndication` matches one
    /// of the High-family profiles whose §5.2.4.1.1 grammar extends the
    /// record (100 / 110 / 122 / 144), the trailing
    /// `chroma_format`/`bit_depth_luma_minus8`/`bit_depth_chroma_minus8`
    /// fields plus the `sequenceParameterSetExt` NAL list are also
    /// consumed (driver currently ignores SPS-Ext, but the parse must
    /// not silently leave bytes behind for downstream readers).
    ///
    /// Layout:
    /// ```text
    ///   u8  configurationVersion (= 1)
    ///   u8  AVCProfileIndication                 // §A.2 profile_idc
    ///   u8  profile_compatibility                // constraint set flags
    ///   u8  AVCLevelIndication                   // §A.3 level_idc
    ///   u8  reserved (6 bits, 111111) + lengthSizeMinusOne (2 bits)
    ///   u8  reserved (3 bits, 111) + numOfSequenceParameterSets (5 bits)
    ///     repeated numOfSequenceParameterSets times:
    ///       u16 sequenceParameterSetLength
    ///       <that many> sequenceParameterSetNALUnit
    ///   u8  numOfPictureParameterSets
    ///     repeated:
    ///       u16 pictureParameterSetLength
    ///       <that many> pictureParameterSetNALUnit
    ///   --- ISO/IEC 14496-15 §5.2.4.1.1 extension (profiles 100/110/122/144) ---
    ///   u8  reserved (6 bits, 111111) + chroma_format (2 bits)
    ///   u8  reserved (5 bits, 11111)  + bit_depth_luma_minus8 (3 bits)
    ///   u8  reserved (5 bits, 11111)  + bit_depth_chroma_minus8 (3 bits)
    ///   u8  numOfSequenceParameterSetExt
    ///     repeated:
    ///       u16 sequenceParameterSetExtLength
    ///       <that many> sequenceParameterSetExtNALUnit
    /// ```
    ///
    /// Per §5.2.4.1.1 `lengthSizeMinusOne` shall take the values 0, 1,
    /// or 3 (mapping to 1-, 2-, or 4-byte length prefixes). The value 2
    /// (3-byte prefix) is forbidden — reject it before storing.
    pub fn consume_extradata(&mut self, extra: &[u8]) -> Result<()> {
        if extra.len() < 7 {
            return Err(Error::invalid("h264: extradata shorter than avcC header"));
        }
        if extra[0] != 1 {
            return Err(Error::invalid("h264: avcC configurationVersion must be 1"));
        }
        let profile_idc = extra[1];
        // Capture, even though the driver picks them up again from the
        // SPS NAL — useful for diagnostics on streams where the avcC
        // header and in-band SPS disagree (the SPS wins).
        self.avcc_profile_idc = Some(profile_idc);
        self.avcc_level_idc = Some(extra[3]);
        let length_size_minus_one = extra[4] & 0x03;
        // §5.2.4.1.1 — lengthSizeMinusOne ∈ {0, 1, 3}. Value 2 (i.e. a
        // 3-byte length prefix) is forbidden by the spec; reject up
        // front so the AVCC framer never has to construct an illegal
        // splitter.
        if length_size_minus_one == 2 {
            return Err(Error::invalid(
                "h264: avcC lengthSizeMinusOne == 2 (3-byte prefix) forbidden by ISO/IEC 14496-15 §5.2.4.1.1",
            ));
        }
        self.length_size = Some(length_size_minus_one + 1);
        let num_sps = (extra[5] & 0x1f) as usize;
        let mut pos = 6;
        for _ in 0..num_sps {
            if pos + 2 > extra.len() {
                return Err(Error::invalid("h264: avcC truncated at SPS length"));
            }
            let len = u16::from_be_bytes([extra[pos], extra[pos + 1]]) as usize;
            pos += 2;
            if pos + len > extra.len() {
                return Err(Error::invalid("h264: avcC truncated at SPS body"));
            }
            let _ = self
                .driver
                .process_nal(&extra[pos..pos + len])
                .map_err(|e| Error::invalid(format!("h264 avcC SPS: {e}")))?;
            pos += len;
        }
        if pos >= extra.len() {
            return Err(Error::invalid("h264: avcC truncated at PPS count"));
        }
        let num_pps = extra[pos] as usize;
        pos += 1;
        for _ in 0..num_pps {
            if pos + 2 > extra.len() {
                return Err(Error::invalid("h264: avcC truncated at PPS length"));
            }
            let len = u16::from_be_bytes([extra[pos], extra[pos + 1]]) as usize;
            pos += 2;
            if pos + len > extra.len() {
                return Err(Error::invalid("h264: avcC truncated at PPS body"));
            }
            let _ = self
                .driver
                .process_nal(&extra[pos..pos + len])
                .map_err(|e| Error::invalid(format!("h264 avcC PPS: {e}")))?;
            pos += len;
        }

        // §5.2.4.1.1 extension: only present for the High family of
        // profile_idc values listed in the spec text. Older muxers that
        // ship a Baseline / Main / Extended record never append these
        // bytes, so the parse cleanly ends with the PPS list above.
        // For 244 (High 4:4:4 Predictive) the spec extends the list of
        // profiles that carry the extension — keep it here even though
        // the 14496-15:2013 edition only enumerated 100/110/122/144.
        if matches!(profile_idc, 100 | 110 | 122 | 144 | 244) {
            // Some real-world MP4 muxers truncate the extension entirely
            // even for these profiles; if we're already past the end,
            // accept the record as-is rather than hard-fail.
            if pos >= extra.len() {
                return Ok(());
            }
            if pos + 4 > extra.len() {
                return Err(Error::invalid(
                    "h264: avcC truncated at High-profile extension header",
                ));
            }
            let chroma_format = extra[pos] & 0x03;
            let bit_depth_luma_minus8 = extra[pos + 1] & 0x07;
            let bit_depth_chroma_minus8 = extra[pos + 2] & 0x07;
            let num_sps_ext = extra[pos + 3] as usize;
            // §7.4.2.1.1 caps both bit_depth fields at 6 (i.e. 14-bit
            // pixel samples). Treat values outside 0..=6 as malformed:
            // an unbounded value here would mislead any downstream code
            // that picks the bit-depth out of the avcC header before
            // the SPS is parsed.
            if bit_depth_luma_minus8 > 6 {
                return Err(Error::invalid(format!(
                    "h264: avcC bit_depth_luma_minus8 = {bit_depth_luma_minus8} exceeds §7.4.2.1.1 cap"
                )));
            }
            if bit_depth_chroma_minus8 > 6 {
                return Err(Error::invalid(format!(
                    "h264: avcC bit_depth_chroma_minus8 = {bit_depth_chroma_minus8} exceeds §7.4.2.1.1 cap"
                )));
            }
            // §7.4.2.1.1 — chroma_format_idc ∈ {0, 1, 2, 3}. The 2-bit
            // field already saturates at 3 so any value is grammatically
            // valid; record it for diagnostic exposure.
            self.avcc_chroma_format = Some(chroma_format);
            self.avcc_bit_depth_luma = Some(8 + bit_depth_luma_minus8);
            self.avcc_bit_depth_chroma = Some(8 + bit_depth_chroma_minus8);
            pos += 4;
            for _ in 0..num_sps_ext {
                if pos + 2 > extra.len() {
                    return Err(Error::invalid("h264: avcC truncated at SPS-Ext length"));
                }
                let len = u16::from_be_bytes([extra[pos], extra[pos + 1]]) as usize;
                pos += 2;
                if pos + len > extra.len() {
                    return Err(Error::invalid("h264: avcC truncated at SPS-Ext body"));
                }
                // Drive the SPS-Ext NAL through the same parse path; the
                // driver may or may not have an SPS-Ext handler today,
                // but it MUST NOT panic, and a parse failure here means
                // the avcC is corrupt.
                let _ = self
                    .driver
                    .process_nal(&extra[pos..pos + len])
                    .map_err(|e| Error::invalid(format!("h264 avcC SPS-Ext: {e}")))?;
                pos += len;
            }
        }
        Ok(())
    }

    /// `AVCProfileIndication` byte from the most recently consumed
    /// `avcC` record (ISO/IEC 14496-15 §5.2.4.1.1), if any.
    /// Returns `None` when this decoder was driven Annex B (no
    /// `consume_extradata` call) or before extradata is supplied.
    pub fn avcc_profile_idc(&self) -> Option<u8> {
        self.avcc_profile_idc
    }

    /// `AVCLevelIndication` byte from the most recently consumed `avcC`
    /// record, if any.
    pub fn avcc_level_idc(&self) -> Option<u8> {
        self.avcc_level_idc
    }

    /// `chroma_format` from the `avcC` High-profile extension
    /// (§5.2.4.1.1), if the record carried one. `0` = monochrome,
    /// `1` = 4:2:0, `2` = 4:2:2, `3` = 4:4:4.
    pub fn avcc_chroma_format(&self) -> Option<u8> {
        self.avcc_chroma_format
    }

    /// Luma bit depth from the `avcC` High-profile extension. The
    /// returned value is `bit_depth_luma_minus8 + 8` (so 8 ≤ x ≤ 14).
    pub fn avcc_bit_depth_luma(&self) -> Option<u8> {
        self.avcc_bit_depth_luma
    }

    /// Chroma bit depth from the `avcC` High-profile extension. The
    /// returned value is `bit_depth_chroma_minus8 + 8` (so 8 ≤ x ≤ 14).
    pub fn avcc_bit_depth_chroma(&self) -> Option<u8> {
        self.avcc_bit_depth_chroma
    }

    /// Number of per-slice / per-picture reconstruction errors that were
    /// swallowed so the stream could keep decoding (the paths that log
    /// `h264 slice skipped: …`). Zero means every slice fed so far
    /// decoded cleanly; when non-zero, frames produced by this decoder
    /// instance may be partial (missing slices) rather than a faithful
    /// reconstruction. Resets with [`Decoder::reset`].
    #[doc(hidden)] // internal diagnostic — exposed for tests/fuzz; not part of the stable API
    pub fn decode_error_count(&self) -> u64 {
        self.decode_errors
    }

    /// §7.4.1.2.1 — the currently active SPS, or `None` before any slice
    /// has been processed.
    #[doc(hidden)] // internal diagnostic — exposed for tests/fuzz; not part of the stable API
    pub fn active_sps(&self) -> Option<&Sps> {
        self.driver.active_sps()
    }

    /// Look up a stored SPS by `seq_parameter_set_id` (§7.4.1.2.1,
    /// 0..=31).
    #[doc(hidden)] // internal diagnostic — exposed for tests/fuzz; not part of the stable API
    pub fn stored_sps(&self, id: u32) -> Option<&Sps> {
        self.driver.sps(id)
    }

    /// Number of reference pictures whose samples the decoder is
    /// currently holding (§8.2.5 DPB contents). Bounded by the DPB
    /// size for the active SPS — regression hook for the round-430
    /// unbounded-store fix.
    #[doc(hidden)] // internal diagnostic — exposed for tests/fuzz; not part of the stable API
    pub fn ref_picture_count(&self) -> usize {
        self.ref_store.stored_count()
    }

    /// Handle a single emitted driver event.
    fn handle_event(&mut self, ev: Event) -> Result<()> {
        // §7.4.1.2.3 — the partitions of a data-partitioned slice are
        // consecutive: any event other than a partition-B/C payload
        // means the pending partitioned slice is complete — decode it
        // before processing the new event.
        if !matches!(ev, Event::SliceDataPartitionBc { .. }) {
            self.flush_pending_dp_slice()?;
        }
        match ev {
            Event::SpsStored(id) => {
                self.output_params.video_color = self.driver.sps(id).and_then(video_color_from_sps);
                Ok(())
            }
            Event::Slice {
                nal_unit_type,
                nal_ref_idc,
                header,
                rbsp,
                slice_data_cursor,
                pps,
                sps,
            } => {
                self.last_slice = Some(header.clone());
                // §8.1 — a coded slice of a separate-colour-plane
                // stream routes to the monochrome sub-decoder of its
                // §7.4.3 colour_plane_id (unless THIS instance already
                // is one of those sub-decoders).
                if sps.separate_colour_plane_flag && !self.scp_plane_mode {
                    return self.route_scp_slice(
                        nal_unit_type,
                        nal_ref_idc,
                        header,
                        rbsp,
                        slice_data_cursor,
                        sps,
                        pps,
                    );
                }
                self.handle_slice(
                    nal_unit_type,
                    nal_ref_idc,
                    header,
                    rbsp,
                    slice_data_cursor,
                    sps,
                    pps,
                )
            }
            // §7.4.1.2.3 — Access Unit Delimiter explicitly marks an
            // access unit boundary. Any picture we've been assembling is
            // finalized here so the next slice opens a fresh one.
            Event::AccessUnitDelimiter(_) => {
                self.forward_event_to_scp_subs(&ev)?;
                self.finalize_in_progress_picture()?;
                Ok(())
            }
            // §7.3.2.5 / §7.3.2.6 — end of sequence / stream close any
            // picture currently being assembled.
            Event::EndOfSequence | Event::EndOfStream => {
                self.forward_event_to_scp_subs(&ev)?;
                self.finalize_in_progress_picture()?;
                Ok(())
            }
            // §7.3.2.9.1 — partition A opens a pending partitioned
            // slice (any previous one was flushed above).
            Event::SliceDataPartitionA {
                nal_ref_idc,
                header,
                slice_id,
                rbsp,
                slice_data_cursor,
                pps,
                sps,
            } => {
                if sps.separate_colour_plane_flag {
                    return Err(Error::invalid(
                        "h264: separate_colour_plane_flag with slice data partitioning is not supported",
                    ));
                }
                self.last_slice = Some(header.clone());
                self.pending_dp = Some(PendingDpSlice {
                    nal_ref_idc,
                    header,
                    rbsp_a: rbsp,
                    cursor_a: slice_data_cursor,
                    slice_id,
                    sps,
                    pps,
                    part_b: None,
                    part_c: None,
                });
                Ok(())
            }
            // §7.3.2.9.2/.3 — partition B/C payloads attach to the
            // pending partition A by slice_id.
            Event::SliceDataPartitionBc {
                is_c,
                slice_id,
                redundant_pic_cnt,
                rbsp,
                slice_data_cursor,
            } => {
                // §7.4.2.9.2 — partitions of redundant coded pictures
                // (redundant_pic_cnt > 0) may be discarded; the
                // primary picture's data is what we decode.
                if redundant_pic_cnt > 0 {
                    return Ok(());
                }
                let Some(pending) = self.pending_dp.as_mut() else {
                    return Err(Error::invalid(format!(
                        "h264: slice data partition {} (slice_id {slice_id}) without a preceding partition A",
                        if is_c { 'C' } else { 'B' },
                    )));
                };
                if pending.slice_id != slice_id {
                    return Err(Error::invalid(format!(
                        "h264: slice data partition {} slice_id {slice_id} does not match partition A slice_id {}",
                        if is_c { 'C' } else { 'B' },
                        pending.slice_id,
                    )));
                }
                let slot = if is_c {
                    &mut pending.part_c
                } else {
                    &mut pending.part_b
                };
                if slot.is_some() {
                    return Err(Error::invalid(format!(
                        "h264: duplicate slice data partition {} for slice_id {slice_id}",
                        if is_c { 'C' } else { 'B' },
                    )));
                }
                *slot = Some((rbsp, slice_data_cursor));
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Decode a held §7.3.2.9 partitioned slice (no-op when none is
    /// pending). The partition A's slice header runs the ordinary
    /// picture-bookkeeping path; the slice-data parse routes each
    /// macroblock's residual to the partition-B/C payloads.
    fn flush_pending_dp_slice(&mut self) -> Result<()> {
        let Some(pending) = self.pending_dp.take() else {
            return Ok(());
        };
        self.handle_slice_with_dp(
            // §7.4.1 — partition A's NAL type is 2 (never IDR; an IDR
            // picture cannot be data-partitioned, §7.4.3 idr_pic_id
            // presence is keyed on nal_unit_type == 5).
            2,
            pending.nal_ref_idc,
            pending.header,
            pending.rbsp_a,
            pending.cursor_a,
            pending.sps,
            pending.pps,
            Some((pending.part_b, pending.part_c)),
        )
    }

    /// §8.1 — route one coded slice of a `separate_colour_plane_flag
    /// == 1` stream to the monochrome sub-decoder selected by its
    /// §7.4.3 `colour_plane_id`, then re-assemble any completed
    /// plane triples into three-plane output frames.
    #[allow(clippy::too_many_arguments)] // mirrors Event::Slice's flat layout
    fn route_scp_slice(
        &mut self,
        nal_unit_type: u8,
        nal_ref_idc: u8,
        header: SliceHeader,
        rbsp: Vec<u8>,
        slice_data_cursor: (usize, u8),
        sps: Sps,
        pps: crate::pps::Pps,
    ) -> Result<()> {
        let plane = header.colour_plane_id as usize;
        if plane > 2 {
            return Err(Error::invalid(format!(
                "h264 slice_header: colour_plane_id {} out of range (§7.4.3 requires 0..=2)",
                header.colour_plane_id
            )));
        }
        let codec_id = self.codec_id.clone();
        let cancellation = self.cancellation.clone();
        let scp = self
            .scp
            .get_or_insert_with(|| Box::new(ScpState::new(&codec_id, cancellation.as_ref())));
        // The packet pts belongs to the access unit; the plane-0
        // (luma) sub-decoder stamps it and the merged frame reuses it.
        if plane == 0 {
            if let Some(pts) = self.pending_pts.take() {
                scp.subs[0].pending_pts = Some(pts);
            }
            scp.subs[0].pending_time_base = self.pending_time_base;
        }
        scp.subs[plane].extra_internal_arenas = queue_arena_identities(&scp.queues[plane]);
        scp.subs[plane].handle_event(Event::Slice {
            nal_unit_type,
            nal_ref_idc,
            header,
            rbsp,
            slice_data_cursor,
            pps,
            sps,
        })?;
        self.drain_and_merge_scp()?;
        Ok(())
    }

    /// Forward a non-slice access-unit-boundary event (AUD /
    /// end-of-sequence / end-of-stream) to the three
    /// separate-colour-plane sub-decoders, when they exist.
    fn forward_event_to_scp_subs(&mut self, ev: &Event) -> Result<()> {
        if let Some(scp) = self.scp.as_mut() {
            for index in 0..scp.subs.len() {
                let internal = queue_arena_identities(&scp.queues[index]);
                scp.subs[index].extra_internal_arenas = internal;
                scp.subs[index].handle_event(ev.clone())?;
            }
            self.drain_and_merge_scp()?;
        }
        Ok(())
    }

    /// Pull every frame the separate-colour-plane sub-decoders have
    /// released into the per-plane queues, then emit one three-plane
    /// frame per completed (S_L, S_Cb, S_Cr) triple (§8.1: "the output
    /// of each of the three decoding processes is assigned to the 3
    /// sample arrays of the current picture").
    fn drain_and_merge_scp(&mut self) -> Result<()> {
        let mut completed = Vec::new();
        let dropped = {
            let Some(scp) = self.scp.as_mut() else {
                return Ok(());
            };
            for index in 0..scp.subs.len() {
                scp.subs[index].extra_internal_arenas = queue_arena_identities(&scp.queues[index]);
                while let Ok(lease) = scp.subs[index].receive_frame_lease() {
                    scp.queues[index].push_back(lease);
                }
                scp.subs[index].extra_internal_arenas = queue_arena_identities(&scp.queues[index]);
            }
            while scp.queues.iter().all(|q| !q.is_empty()) {
                completed.push((
                    scp.queues[0].pop_front().expect("checked non-empty"),
                    scp.queues[1].pop_front().expect("checked non-empty"),
                    scp.queues[2].pop_front().expect("checked non-empty"),
                ));
            }

            // Anti-OOM guard for NON-conforming streams: §7.4.1.2 requires
            // every access unit to carry all three colour planes, so the
            // per-plane queues stay shallow on legal input. A malformed
            // stream feeding only one colour_plane_id would otherwise grow
            // its queue without bound — drop the oldest unpairable plane
            // pictures past a generous cap and count them as decode errors.
            const SCP_QUEUE_CAP: usize = 64;
            let mut dropped = 0u64;
            for q in scp.queues.iter_mut() {
                while q.len() > SCP_QUEUE_CAP {
                    q.pop_front();
                    dropped += 1;
                }
            }
            dropped
        };
        self.decode_errors += dropped;

        let cancellation = self.cancellation.clone();
        for (y, cb, cr) in completed {
            let internal = self.internal_arena_identities(&self.assembly_pool);
            let merged = merge_separate_colour_planes(
                &mut self.assembly_pool,
                &y,
                &cb,
                &cr,
                cancellation.as_ref(),
                &internal,
            )?;
            self.ready.push_back(merged);
        }
        Ok(())
    }

    /// §7.4.1.2.4 — decide whether `header` opens a new primary coded
    /// picture, by comparing to the first slice of the
    /// `in_progress` picture. Returns true when any of the listed
    /// conditions in §7.4.1.2.4 differs (new picture) or when there is
    /// no picture currently in progress.
    ///
    /// The conditions enumerated in the spec (and used here):
    ///   * `frame_num` differs
    ///   * `pic_parameter_set_id` differs
    ///   * `field_pic_flag` differs
    ///   * `bottom_field_flag` differs (both being field pictures) — the
    ///     two fields of a complementary pair share `frame_num` but are
    ///     separate primary coded pictures
    ///   * `nal_ref_idc` is 0 for one and non-0 for the other
    ///   * `pic_order_cnt_lsb` differs  (pic_order_cnt_type == 0)
    ///   * `delta_pic_order_cnt_bottom` differs (pic_order_cnt_type == 0
    ///     and bottom_field_pic_order_in_frame_present_flag == 1)
    ///   * `delta_pic_order_cnt[0]` differs (pic_order_cnt_type == 1)
    ///   * `delta_pic_order_cnt[1]` differs (pic_order_cnt_type == 1 and
    ///     bottom_field_pic_order_in_frame_present_flag == 1)
    ///   * `IdrPicFlag` differs (one is IDR, the other isn't)
    ///   * `IdrPicFlag == 1` AND `idr_pic_id` differs
    fn is_first_vcl_of_new_picture(
        &self,
        nal_unit_type: u8,
        nal_ref_idc: u8,
        header: &SliceHeader,
    ) -> bool {
        let Some(in_progress) = self.in_progress.as_ref() else {
            return true;
        };
        let prev = &in_progress.first_header;
        let prev_idr = in_progress.first_nal_unit_type == 5;
        let curr_idr = nal_unit_type == 5;
        let prev_is_ref = in_progress.first_nal_ref_idc != 0;
        let curr_is_ref = nal_ref_idc != 0;

        if prev.frame_num != header.frame_num {
            return true;
        }
        if prev.pic_parameter_set_id != header.pic_parameter_set_id {
            return true;
        }
        if prev.field_pic_flag != header.field_pic_flag {
            return true;
        }
        // §7.4.1.2.4 — `bottom_field_flag` differs (when both are field
        // pictures). The top and bottom fields of a complementary pair
        // carry the same `frame_num` + `field_pic_flag` but are distinct
        // primary coded pictures; this is the condition that forces the
        // top field to finalize before the bottom field opens.
        if header.field_pic_flag && prev.bottom_field_flag != header.bottom_field_flag {
            return true;
        }
        if prev_is_ref != curr_is_ref {
            return true;
        }
        if prev.pic_order_cnt_lsb != header.pic_order_cnt_lsb {
            return true;
        }
        if prev.delta_pic_order_cnt_bottom != header.delta_pic_order_cnt_bottom {
            return true;
        }
        if prev.delta_pic_order_cnt[0] != header.delta_pic_order_cnt[0] {
            return true;
        }
        if prev.delta_pic_order_cnt[1] != header.delta_pic_order_cnt[1] {
            return true;
        }
        if prev_idr != curr_idr {
            return true;
        }
        if curr_idr && prev.idr_pic_id != header.idr_pic_id {
            return true;
        }
        false
    }

    /// Drive reconstruction for one slice NAL. Covers both the IDR
    /// and P/B paths.
    ///
    /// §7.4.1.2.4 multi-slice assembly: when this slice is a continuation
    /// of the in-progress picture (same frame_num / POC / IDR status etc.)
    /// we reconstruct straight into the existing Picture + MbGrid so the
    /// slice's macroblocks land alongside the earlier slices'. When this
    /// slice opens a new primary coded picture we first finalize the
    /// previous in-progress one (pushing it into the DPB + output queue)
    /// and then start a fresh Picture + MbGrid.
    ///
    /// Known limitation: reconstruct_slice runs a full-picture deblocking
    /// pass at the end (§8.7). For a continuation slice the earlier
    /// slices' macroblocks are still in the grid with `available == true`,
    /// so the deblocker may re-filter their interior edges — a slight
    /// over-filter. Proper behaviour would defer deblocking until all
    /// slices are in, but the deblocking helper is private to
    /// `reconstruct.rs` and cannot be invoked separately from here.
    /// In practice the artifact is minor compared to the coarse blocking
    /// you get with one-slice-per-picture assembly, which is the bug this
    /// replaces.
    #[allow(clippy::too_many_arguments)] // mirrors Event::Slice's flat layout
    fn handle_slice(
        &mut self,
        nal_unit_type: u8,
        nal_ref_idc: u8,
        header: SliceHeader,
        rbsp: Vec<u8>,
        cursor: (usize, u8),
        sps: Sps,
        pps: crate::pps::Pps,
    ) -> Result<()> {
        self.handle_slice_with_dp(
            nal_unit_type,
            nal_ref_idc,
            header,
            rbsp,
            cursor,
            sps,
            pps,
            None,
        )
    }

    /// [`handle_slice`] with optional §7.3.2.9 data-partition payloads
    /// (`(partition B, partition C)`, each `(rbsp, cursor)`).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn handle_slice_with_dp(
        &mut self,
        nal_unit_type: u8,
        nal_ref_idc: u8,
        header: SliceHeader,
        rbsp: Vec<u8>,
        cursor: (usize, u8),
        sps: Sps,
        pps: crate::pps::Pps,
        dp: Option<(
            Option<(Vec<u8>, (usize, u8))>,
            Option<(Vec<u8>, (usize, u8))>,
        )>,
    ) -> Result<()> {
        // §7.4.2.2 / §7.4.1.2 — a slice with `redundant_pic_cnt > 0`
        // belongs to a REDUNDANT coded picture: an approximation of
        // (part of) the primary picture that a decoder "may" use for
        // error recovery and may otherwise discard. We decode primary
        // data only — letting a redundant slice through here would
        // overwrite the primary picture's already-decoded macroblocks
        // with the approximation (the §7.4.1.2.4 first-VCL detection
        // deliberately keys on the PRIMARY picture's header fields, so
        // a redundant slice never opens a picture of its own).
        if header.redundant_pic_cnt > 0 {
            return Ok(());
        }

        // The SPS and PPS have been snapshotted at slice-header parse
        // time (see [`Event::Slice::pps`] for why — same-id PPS
        // re-transmission at access-unit boundaries, as in JVT CACQP3).
        let is_idr = nal_unit_type == 5;
        let is_reference = nal_ref_idc != 0;

        // §7.4.1.2.4 — first VCL of a primary coded picture?
        let starts_new_picture =
            self.is_first_vcl_of_new_picture(nal_unit_type, nal_ref_idc, &header);

        if starts_new_picture {
            // Finalize whatever was in progress before starting the new
            // picture with this slice.
            self.finalize_in_progress_picture()?;

            // §7.4.3 + Annex A — `first_mb_in_slice` of the first
            // coded slice of a coded picture must be 0 when arbitrary
            // slice order (ASO) is not allowed. §A.2.1 (Baseline) and
            // §A.2.3 (Extended) DO allow ASO — the slices of a coded
            // picture may arrive in any order, so the picture-opening
            // slice may legitimately cover any macroblock range
            // (round 451: previously rejected unconditionally, which
            // blocked every ASO stream). For every other profile the
            // slices form a contiguous raster walk and a non-zero
            // opener is a conformance violation.
            //
            // The hazard this rejection used to guard — a hostile
            // stream whose "first" slices never cover the leading MBs,
            // which would emit a Frame::Video with zero-initialised
            // luma (fuzz oracle `crash-957ac808…`: 440 B, four IDR
            // slices all with `first_mb_in_slice == 2`) — is closed
            // independently by `finalize_in_progress_picture`'s
            // full-coverage check: a picture whose MbGrid still has
            // unavailable entries at finalize time is dropped, never
            // emitted. ASO streams that DO cover the whole picture
            // pass that check whatever order their slices arrived in.
            if header.first_mb_in_slice != 0 && !matches!(sps.profile_idc, 66 | 88) {
                return Err(Error::invalid(format!(
                    "h264 slice_header: first_mb_in_slice = {} for first slice of a coded picture (§7.4.3 / Annex A require 0 when ASO is not allowed — profile_idc {})",
                    header.first_mb_in_slice, sps.profile_idc
                )));
            }

            // Shared §7.4.3/§8.2.1/§8.2.5 preparation: validates frame_num
            // discipline, simulates allowed non-existing references, derives
            // POC and snapshots the DPB that this picture must decode against.
            let prepared = self.picture_frontend.prepare_parsed_picture(
                nal_unit_type,
                nal_ref_idc,
                header.clone(),
                sps.clone(),
                pps.clone(),
                1,
            )?;
            let poc = prepared.poc;
            let structure = prepared.structure;

            self.ensure_picture_pool_sized(&sps);

            // §8.2.5.2 synthetic references carry metadata in the shared
            // frontend. The software backend supplies neutral pooled samples
            // for their opaque keys; conforming streams never sample them.
            if !prepared.synthetic_references.is_empty() {
                let bit_depth_y = sps.bit_depth_luma_minus8 + 8;
                let bit_depth_c = sps.bit_depth_chroma_minus8 + 8;
                for entry in &prepared.synthetic_references {
                    let mut gray = self.allocate_picture(
                        sps.pic_width_in_mbs() * 16,
                        sps.frame_height_in_mbs() * 16,
                        sps.chroma_array_type(),
                        bit_depth_y,
                        bit_depth_c,
                    )?;
                    gray.non_existing = true;
                    gray.fill_luma(1 << bit_depth_y.saturating_sub(1));
                    gray.fill_cb(1 << bit_depth_c.saturating_sub(1));
                    gray.fill_cr(1 << bit_depth_c.saturating_sub(1));
                    self.ref_store.insert(entry.dpb_key, gray);
                }
            }

            // §7.4.2.1.1 eq. (7-26) — a PAFF field picture
            // (`field_pic_flag == 1`) is decoded as a half-height picture
            // (`PicHeightInMbs = FrameHeightInMbs / 2`). The in-progress
            // `pic` + `grid` are sized to the coded picture's own height;
            // the two complementary fields are re-interleaved into the
            // full-height output frame at picture-pairing time.
            let pic_height_in_mbs = sps.pic_height_in_mbs(header.field_pic_flag);
            let width_samples = sps.pic_width_in_mbs() * 16;
            let height_samples = pic_height_in_mbs * 16;
            let chroma_array_type = sps.chroma_array_type();
            let pic = self.allocate_picture(
                width_samples,
                height_samples,
                chroma_array_type,
                sps.bit_depth_luma_minus8 + 8,
                sps.bit_depth_chroma_minus8 + 8,
            )?;
            let grid = MbGrid::new(sps.pic_width_in_mbs(), pic_height_in_mbs);

            // Consume the packet-level pts exactly once per access unit
            // — the first slice to open a picture gets it.
            let pts = self.pending_pts.take();
            let time_base = self.pending_time_base;

            let deblock_enabled = header.disable_deblocking_filter_idc != 1;
            let deblock_alpha_off = header.slice_alpha_c0_offset_div2 * 2;
            let deblock_beta_off = header.slice_beta_offset_div2 * 2;
            let mb_count = (sps.pic_width_in_mbs() * pic_height_in_mbs) as usize;
            let mb_field_flags = vec![false; mb_count];

            self.in_progress = Some(PictureInProgress {
                prepared,
                pic,
                grid,
                first_nal_unit_type: nal_unit_type,
                first_nal_ref_idc: nal_ref_idc,
                first_header: header.clone(),
                is_reference,
                is_idr,
                poc,
                structure,
                pts,
                time_base,
                deblock_enabled,
                deblock_alpha_off,
                deblock_beta_off,
                mb_field_flags,
                sps: sps.clone(),
                pps: pps.clone(),
                any_slice_succeeded: false,
            });
        }

        // Reconstruct this slice into the in-progress picture.
        self.reconstruct_slice_into_in_progress(
            nal_ref_idc,
            &header,
            &rbsp,
            cursor,
            &sps,
            &pps,
            dp,
        )?;

        Ok(())
    }

    /// Run `reconstruct::reconstruct_slice` against the currently
    /// in-progress picture's pic + grid. Also picks up any OR of
    /// `is_reference` so a picture is marked a reference picture as
    /// soon as any of its slices carries nal_ref_idc != 0 (§7.4.1.2.4
    /// requires this to be uniform across slices, but we're tolerant).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn reconstruct_slice_into_in_progress(
        &mut self,
        nal_ref_idc: u8,
        header: &SliceHeader,
        rbsp: &[u8],
        cursor: (usize, u8),
        sps: &Sps,
        pps: &crate::pps::Pps,
        dp: Option<(
            Option<(Vec<u8>, (usize, u8))>,
            Option<(Vec<u8>, (usize, u8))>,
        )>,
    ) -> Result<()> {
        // Parse slice_data — common to I / P / B paths. A §7.3.2.9
        // data-partitioned slice routes its residual reads to the
        // partition-B/C payloads.
        let sd = match &dp {
            None => slice_data::parse_slice_data(rbsp, cursor.0, cursor.1, header, sps, pps),
            Some((b, c)) => slice_data::parse_slice_data_partitioned(
                rbsp,
                cursor.0,
                cursor.1,
                b.as_ref().map(|(buf, cur)| (&buf[..], cur.0, cur.1)),
                c.as_ref().map(|(buf, cur)| (&buf[..], cur.0, cur.1)),
                header,
                sps,
                pps,
            ),
        }
        .map_err(|e| Error::invalid(format!("h264 slice_data: {e}")))?;

        let prepared_references = self
            .in_progress
            .as_ref()
            .expect("in_progress must have been seeded by handle_slice")
            .prepared
            .references
            .clone();
        let in_progress = self
            .in_progress
            .as_mut()
            .expect("in_progress must have been seeded by handle_slice");

        // Update the reference bit for the whole picture if any slice
        // is a reference slice.
        if nal_ref_idc != 0 {
            in_progress.is_reference = true;
        }

        let current_structure = in_progress.structure;
        let current_bottom = matches!(current_structure, PicStructure::BottomField);
        let current_is_field = header.field_pic_flag;
        let pic_order_cnt = in_progress.poc.pic_order_cnt;
        let is_idr = in_progress.is_idr;

        // Build per-slice RefPicList0 / RefPicList1.
        //
        // Round-416 PAFF: a coded FIELD picture (`field_pic_flag == 1`)
        // initialises its lists through the §8.2.4.2.2/.2.4 +
        // §8.2.4.2.5 field process — per-field entries interleaved by
        // alternating parity starting from the current field's own
        // parity — and the §8.2.4.3 field RPLM (eq. 8-30..8-33 PicNum
        // forms). The plain frame init below sorts by per-field PicNum
        // only, which puts the complementary field of the CURRENT frame
        // (highest PicNum but opposite parity) at index 0 for a second
        // field — the §8.2.4.2.5 alternation instead starts with the
        // same-parity field of the previous frame.
        let mut l0_overrides: Vec<Option<Picture>> = Vec::new();
        let mut l1_overrides: Vec<Option<Picture>> = Vec::new();
        let mut l0_parities: Vec<Option<u8>> = Vec::new();
        let mut l1_parities: Vec<Option<u8>> = Vec::new();
        let mut l0_unit_keys: Vec<u32> = Vec::new();
        let mut l1_unit_keys: Vec<u32> = Vec::new();
        // §8.4.1.2.3 — per-entry frame-level (TopFOC, BottomFOC) of
        // FRAME-slice list units (per-field tb/td + eq. 8-182).
        let mut l0_focs: Vec<(i32, i32)> = Vec::new();
        let mut l1_focs: Vec<(i32, i32)> = Vec::new();
        // §8.4.1.2.1 Table 8-6 — the (top, bottom) stored-field keys
        // behind complementary-PAIR units of a FRAME slice's lists.
        let mut l0_pair_keys: Vec<Option<(u32, u32)>> = Vec::new();
        let mut l1_pair_keys: Vec<Option<(u32, u32)>> = Vec::new();
        let mut field_pocs_lt: Option<FieldListPocsLt> = None;
        let (list0, list1) = if is_idr {
            (Vec::new(), Vec::new())
        } else if current_is_field {
            let max_frame_num = 1u32 << (sps.log2_max_frame_num_minus4 + 4);
            let (mut fl0, mut fl1) = match header.slice_type {
                SliceType::P | SliceType::SP => (
                    ref_list::init_ref_pic_list_p_field(
                        &prepared_references,
                        header.frame_num,
                        max_frame_num,
                        current_bottom,
                    ),
                    Vec::new(),
                ),
                SliceType::B => ref_list::init_ref_pic_lists_b_field(
                    &prepared_references,
                    pic_order_cnt,
                    current_bottom,
                ),
                SliceType::I | SliceType::SI => (Vec::new(), Vec::new()),
            };
            // §8.2.4 — same all-'no reference picture' refusal as the
            // frame path below.
            if header.slice_type.has_list_0() && fl0.is_empty() {
                return Err(Error::invalid(format!(
                    "h264 slice_header: {:?} field slice but the DPB holds no reference field (§8.2.4 RefPicList0 would be all 'no reference picture')",
                    header.slice_type
                )));
            }
            if header.slice_type.has_list_0() {
                let ops_l0: Vec<RplmOp> = header
                    .ref_pic_list_modification
                    .modifications_l0
                    .iter()
                    .map(slice_rplm_to_ref_rplm)
                    .collect();
                ref_list::modify_ref_pic_list_field(
                    &mut fl0,
                    &ops_l0,
                    &prepared_references,
                    header.num_ref_idx_l0_active_minus1 + 1,
                    header.frame_num,
                    max_frame_num,
                    current_bottom,
                );
            }
            if header.slice_type.has_list_1() {
                let ops_l1: Vec<RplmOp> = header
                    .ref_pic_list_modification
                    .modifications_l1
                    .iter()
                    .map(slice_rplm_to_ref_rplm)
                    .collect();
                ref_list::modify_ref_pic_list_field(
                    &mut fl1,
                    &ops_l1,
                    &prepared_references,
                    header.num_ref_idx_l1_active_minus1 + 1,
                    header.frame_num,
                    max_frame_num,
                    current_bottom,
                );
            }
            let r0 = Self::resolve_field_list(&prepared_references, &self.ref_store, &fl0);
            let r1 = Self::resolve_field_list(&prepared_references, &self.ref_store, &fl1);
            l0_overrides = r0.overrides;
            l1_overrides = r1.overrides;
            l0_parities = r0.parities;
            l1_parities = r1.parities;
            l0_unit_keys = r0.unit_keys;
            l1_unit_keys = r1.unit_keys;
            // Field lists: each entry is itself a field — its tb/td
            // POC is the per-field POC already carried in `pocs`.
            l0_focs = r0.pocs.iter().map(|&p| (p, p)).collect();
            l1_focs = r1.pocs.iter().map(|&p| (p, p)).collect();
            l0_pair_keys = vec![None; r0.keys.len()];
            l1_pair_keys = vec![None; r1.keys.len()];
            field_pocs_lt = Some((r0.pocs, r0.longterm, r1.pocs, r1.longterm));
            (r0.keys, r1.keys)
        } else {
            let max_frame_num = 1u32 << (sps.log2_max_frame_num_minus4 + 4);
            // §8.2.4.1 / §8.2.4.2.1 / §8.2.4.2.3 — a FRAME slice's
            // reference lists range over frame-level UNITS: decoded
            // reference frames and complementary reference field
            // PAIRS (two stored coded-field entries collapse into one
            // unit; non-paired reference fields are excluded).
            let (frame_units, pairings) = ref_list::collapse_field_pairs(&prepared_references);
            let (mut l0, mut l1) = match header.slice_type {
                SliceType::P | SliceType::SP => (
                    ref_list::init_ref_pic_list_p(
                        &frame_units,
                        header.frame_num,
                        max_frame_num,
                        current_structure,
                        current_bottom,
                    ),
                    Vec::new(),
                ),
                SliceType::B => ref_list::init_ref_pic_lists_b(
                    &frame_units,
                    pic_order_cnt,
                    current_structure,
                    current_bottom,
                ),
                SliceType::I | SliceType::SI => (Vec::new(), Vec::new()),
            };

            // §8.2.4 — an inter-predicted (P/SP/B) slice with NO usable
            // reference picture in the DPB at all. §8.2.4.2.1 pads a
            // too-short RefPicList with "no reference picture" entries,
            // and referring to one of those is barred by conformance —
            // so when the initial list is completely empty every
            // ref_idx the slice could code is invalid before a single
            // macroblock is parsed. A stream can only be entered at an
            // IDR (or with the references it needs); decoding such a
            // slice — even one whose macroblocks all happen to be
            // intra-coded — silently fabricates a picture the encoder
            // never meant to exist on its own, so refuse it up front
            // like reference decoders do.
            if header.slice_type.has_list_0() && l0.is_empty() {
                return Err(Error::invalid(format!(
                    "h264 slice_header: {:?} slice but the DPB holds no reference picture (§8.2.4 RefPicList0 would be all 'no reference picture')",
                    header.slice_type
                )));
            }

            if header.slice_type.has_list_0() {
                let ops_l0: Vec<RplmOp> = header
                    .ref_pic_list_modification
                    .modifications_l0
                    .iter()
                    .map(slice_rplm_to_ref_rplm)
                    .collect();
                ref_list::modify_ref_pic_list(
                    &mut l0,
                    &ops_l0,
                    &frame_units,
                    header.num_ref_idx_l0_active_minus1 + 1,
                    header.frame_num,
                    max_frame_num,
                    current_is_field,
                    current_bottom,
                );
            }
            if header.slice_type.has_list_1() {
                let ops_l1: Vec<RplmOp> = header
                    .ref_pic_list_modification
                    .modifications_l1
                    .iter()
                    .map(slice_rplm_to_ref_rplm)
                    .collect();
                ref_list::modify_ref_pic_list(
                    &mut l1,
                    &ops_l1,
                    &frame_units,
                    header.num_ref_idx_l1_active_minus1 + 1,
                    header.frame_num,
                    max_frame_num,
                    current_is_field,
                    current_bottom,
                );
            }

            // §8.4.2.1 — resolve each unit key: a complementary-pair
            // unit materialises the full-height reference frame by
            // re-interleaving its two stored half-height fields (top →
            // even rows, bottom → odd rows), stamped with the pair's
            // eq. 8-1 PicOrderCnt. Plain frame units resolve through
            // the store as before. POC / long-term metadata come from
            // the unit entries (a pair's POC is Min of its field POCs
            // — the stored top FIELD picture alone carries only its
            // own field POC).
            type FrameListResolution = (
                Vec<Option<Picture>>,
                Vec<i32>,
                Vec<bool>,
                Vec<(i32, i32)>,
                Vec<Option<(u32, u32)>>,
            );
            let resolve_frame_list = |keys: &[u32]| -> FrameListResolution {
                let mut overrides = Vec::with_capacity(keys.len());
                let mut pocs = Vec::with_capacity(keys.len());
                let mut lts = Vec::with_capacity(keys.len());
                let mut focs = Vec::with_capacity(keys.len());
                let mut pair_keys = Vec::with_capacity(keys.len());
                for &key in keys {
                    let unit = frame_units.iter().find(|u| u.dpb_key == key);
                    let pairing = pairings.iter().find(|p| p.unit_key == key);
                    let ov = pairing.and_then(|p| {
                        let top = self.ref_store.get_by_key(p.top_key)?;
                        let bottom = self.ref_store.get_by_key(p.bottom_key)?;
                        let mut merged = interleave_fields(top, bottom);
                        if let Some(u) = unit {
                            merged.pic_order_cnt = u.pic_order_cnt;
                            merged.frame_num = u.frame_num;
                        }
                        Some(merged)
                    });
                    overrides.push(ov);
                    pocs.push(unit.map(|u| u.pic_order_cnt).unwrap_or_else(|| {
                        self.ref_store
                            .get_by_key(key)
                            .map(|p| p.pic_order_cnt)
                            .unwrap_or(0)
                    }));
                    lts.push(unit.map(|u| u.is_long_term()).unwrap_or(false));
                    focs.push(
                        unit.map(|u| (u.top_field_order_cnt, u.bottom_field_order_cnt))
                            .unwrap_or((0, 0)),
                    );
                    pair_keys.push(pairing.map(|p| (p.top_key, p.bottom_key)));
                }
                (overrides, pocs, lts, focs, pair_keys)
            };
            let (ov0, pocs0, lt0, focs0, pk0) = resolve_frame_list(&l0);
            let (ov1, pocs1, lt1, focs1, pk1) = resolve_frame_list(&l1);
            l0_overrides = ov0;
            l1_overrides = ov1;
            l0_unit_keys = l0.clone();
            l1_unit_keys = l1.clone();
            l0_parities = vec![None; l0.len()];
            l1_parities = vec![None; l1.len()];
            l0_focs = focs0;
            l1_focs = focs1;
            l0_pair_keys = pk0;
            l1_pair_keys = pk1;
            field_pocs_lt = Some((pocs0, lt0, pocs1, lt1));

            (l0, l1)
        };

        // §8.4.* — pixel reconstruction into the in-progress picture.
        // Stamp the current picture's POC + frame_num so §8.4.1.2.3
        // temporal-direct derivation can consult it. Idempotent across
        // the slices of a coded picture (§7.4.1.2.4 requires POC
        // consistency).
        in_progress.pic.pic_order_cnt = in_progress.poc.pic_order_cnt;
        in_progress.pic.frame_num = header.frame_num;
        // §8.4.1.2.1 Table 8-7 + §8.4.1.2.3 — the current picture's
        // coding structure, field parity and per-field order counts
        // feed the temporal-direct co-located derivation.
        in_progress.pic.coding_struct = if header.field_pic_flag {
            crate::picture::PicCodingStruct::Fld
        } else if sps.mb_adaptive_frame_field_flag {
            crate::picture::PicCodingStruct::Afrm
        } else {
            crate::picture::PicCodingStruct::Frm
        };
        in_progress.pic.is_bottom_field = header.bottom_field_flag;
        in_progress.pic.top_field_order_cnt = in_progress.poc.top_field_order_cnt;
        in_progress.pic.bottom_field_order_cnt = in_progress.poc.bottom_field_order_cnt;

        // §8.4.1.2.3 — precompute POCs + long-term flags for the
        // slice's RefPicList0. A later B-slice uses this picture as
        // the colocated picture and invokes MapColToList0 which
        // requires picture-identity lookup (by POC) back into the
        // list that was active when this picture was decoded.
        // Round-416 PAFF: field slices already computed per-FIELD POCs
        // (top/bottom field order counts) during list resolution — a
        // stored frame's two fields share a dpb_key but carry distinct
        // field POCs, so the key-based lookup below would be ambiguous.
        let (list_0_pocs, list_0_longterm, list_1_pocs, list_1_longterm) =
            if let Some(t) = field_pocs_lt {
                t
            } else {
                let list_0_pocs: Vec<i32> = list0
                    .iter()
                    .map(|&key| {
                        self.ref_store
                            .get_by_key(key)
                            .map(|p| p.pic_order_cnt)
                            .unwrap_or(0)
                    })
                    .collect();
                let list_0_longterm: Vec<bool> = list0
                    .iter()
                    .map(|&key| {
                        prepared_references
                            .iter()
                            .find(|e| e.dpb_key == key)
                            .map(|e| e.is_long_term())
                            .unwrap_or(false)
                    })
                    .collect();
                let list_1_pocs: Vec<i32> = list1
                    .iter()
                    .map(|&key| {
                        self.ref_store
                            .get_by_key(key)
                            .map(|p| p.pic_order_cnt)
                            .unwrap_or(0)
                    })
                    .collect();
                let list_1_longterm: Vec<bool> = list1
                    .iter()
                    .map(|&key| {
                        prepared_references
                            .iter()
                            .find(|e| e.dpb_key == key)
                            .map(|e| e.is_long_term())
                            .unwrap_or(false)
                    })
                    .collect();
                (list_0_pocs, list_0_longterm, list_1_pocs, list_1_longterm)
            };
        // Idempotent across slices: once set for a picture, only
        // update if still empty (shared lists across slices of one
        // primary coded picture have the same POCs — but RPLM may
        // differ. Using the first non-empty snapshot matches the
        // common "first slice wins" convention for primary MB 0).
        if in_progress.pic.ref_list_0_pocs.is_empty() {
            in_progress.pic.ref_list_0_pocs = list_0_pocs.clone();
            in_progress.pic.ref_list_0_longterm = list_0_longterm.clone();
            in_progress.pic.ref_list_1_pocs = list_1_pocs.clone();
            in_progress.pic.ref_list_1_longterm = list_1_longterm.clone();
            // §8.4.1.2.3 MapColToList0 — picture-identity snapshot: a
            // later B slice using this picture as colPic resolves the
            // colocated block's refIdxCol to a concrete DPB unit.
            in_progress.pic.ref_list_0_keys = list0.clone();
            in_progress.pic.ref_list_1_keys = list1.clone();
            in_progress.pic.ref_list_0_parities = l0_parities.clone();
            in_progress.pic.ref_list_1_parities = l1_parities.clone();
            in_progress.pic.ref_list_0_unit_keys = l0_unit_keys.clone();
            in_progress.pic.ref_list_1_unit_keys = l1_unit_keys.clone();
        }
        let _ = list_1_pocs;
        let provider = BorrowedRefProvider {
            store: &self.ref_store,
            list_0: &list0,
            list_1: &list1,
            list_0_pocs,
            list_0_longterm,
            list_1_longterm,
            list_0_overrides: l0_overrides,
            list_1_overrides: l1_overrides,
            list_0_parities: l0_parities,
            list_1_parities: l1_parities,
            list_0_unit_keys: l0_unit_keys,
            list_1_unit_keys: l1_unit_keys,
            list_0_focs: l0_focs,
            list_1_focs: l1_focs,
            list_0_pair_keys: l0_pair_keys,
            list_1_pair_keys: l1_pair_keys,
        };
        reconstruct::reconstruct_slice_no_deblock(
            &sd,
            header,
            sps,
            pps,
            &provider,
            &mut in_progress.pic,
            &mut in_progress.grid,
        )
        .map_err(|e| Error::invalid(format!("h264 reconstruct: {e}")))?;

        // §7.4.4 — copy this slice's per-MB mb_field_decoding_flag
        // values into the picture-wide array so `finalize_in_progress_picture`
        // can hand the whole picture's flags to the deblocker in one shot.
        // `sd.macroblocks[i]` corresponds to macroblock address
        // `first_mb_in_slice * (1 + MbaffFrameFlag) + i` in the raster /
        // slice-group-0 walk.
        let mbaff_frame_flag = sps.mb_adaptive_frame_field_flag && !header.field_pic_flag;
        let mut curr_addr = header.first_mb_in_slice * (1 + u32::from(mbaff_frame_flag));
        // §8.2.2 — FMO slice group map (round-453).
        let mb_map = crate::mb_address::slice_mb_to_slice_group_map(sps, pps, header);
        for flag in sd.mb_field_decoding_flags.iter().copied() {
            if let Some(slot) = in_progress.mb_field_flags.get_mut(curr_addr as usize) {
                *slot = flag;
            }
            curr_addr = crate::mb_address::advance_mb_addr(curr_addr, mb_map.as_deref());
        }

        // §7.4.1.2.4 — record that this slice fully reconstructed.
        // `finalize_in_progress_picture` consults this flag and
        // discards pictures whose every slice failed parse /
        // reconstruction (rather than emitting a never-painted
        // frame). Reaching this line means `parse_slice_data` and
        // `reconstruct_slice_no_deblock` both returned Ok.
        in_progress.any_slice_succeeded = true;

        Ok(())
    }

    /// Round-416 PAFF — resolve a §8.2.4.2.5 per-field reference list
    /// into the provider's shape: parallel vectors of (dpb_key,
    /// optional field-view override, per-FIELD POC, long-term flag).
    ///
    /// * an entry naming a stored coded field maps to that field's own
    ///   dpb_key (the stored picture already IS the half-height field);
    /// * an entry naming one parity field of a picture stored as a
    ///   FRAME materialises a [`Picture::field_view`] override stamped
    ///   with the field's own order count (top/bottom FOC);
    /// * the §8.2.4.2 "no reference picture" sentinel (`u32::MAX`)
    ///   stays unresolvable — the provider returns `None` and motion
    ///   compensation refuses the reference, per conformance.
    fn resolve_field_list(
        dpb_entries: &[DpbEntry],
        ref_store: &RefPicStore,
        entries: &[ref_list::RefFieldEntry],
    ) -> ResolvedFieldList {
        let mut keys = Vec::with_capacity(entries.len());
        let mut overrides = Vec::with_capacity(entries.len());
        let mut pocs = Vec::with_capacity(entries.len());
        let mut lts = Vec::with_capacity(entries.len());
        let mut parities = Vec::with_capacity(entries.len());
        let mut unit_keys = Vec::with_capacity(entries.len());
        // §8.2.4.1 — frame-level unit key of a coded-field entry: the
        // complementary pair's key (top field's storage key) when the
        // opposite-parity partner exists, else the entry's own key.
        let unit_key_of = |dpb: &DpbEntry| -> u32 {
            if dpb.structure.is_field() {
                let partner = dpb_entries.iter().find(|d| {
                    d.structure.is_field()
                        && d.frame_num == dpb.frame_num
                        && d.structure.is_bottom() != dpb.structure.is_bottom()
                });
                match partner {
                    Some(p) => {
                        if dpb.structure.is_bottom() {
                            p.dpb_key
                        } else {
                            dpb.dpb_key
                        }
                    }
                    None => dpb.dpb_key,
                }
            } else {
                dpb.dpb_key
            }
        };
        for e in entries {
            let Some(dpb) = dpb_entries.iter().find(|d| d.dpb_key == e.dpb_key) else {
                keys.push(u32::MAX);
                overrides.push(None);
                pocs.push(0);
                lts.push(false);
                parities.push(None);
                unit_keys.push(u32::MAX);
                continue;
            };
            unit_keys.push(unit_key_of(dpb));
            let bottom = e.parity == ref_list::FieldParity::Bottom;
            let field_poc = if bottom {
                dpb.bottom_field_order_cnt
            } else {
                dpb.top_field_order_cnt
            };
            match dpb.structure {
                PicStructure::TopField | PicStructure::BottomField => {
                    keys.push(e.dpb_key);
                    overrides.push(None);
                }
                PicStructure::Frame | PicStructure::FieldPair => {
                    let ov = ref_store.get_by_key(e.dpb_key).map(|p| {
                        let mut v = p.field_view(bottom);
                        v.pic_order_cnt = field_poc;
                        v
                    });
                    keys.push(e.dpb_key);
                    overrides.push(ov);
                }
            }
            pocs.push(field_poc);
            lts.push(dpb.is_long_term());
            parities.push(Some(u8::from(bottom)));
        }
        ResolvedFieldList {
            keys,
            overrides,
            pocs,
            longterm: lts,
            parities,
            unit_keys,
        }
    }

    /// Complete the picture currently held in `self.in_progress`: run the
    /// §8.2.5 decoded reference picture marking, insert the picture into
    /// the DPB if it's a reference, and push the finalized `VideoFrame`
    /// through the §C.4 output bumping process. Clears `in_progress`.
    ///
    /// No-op when no picture is in progress.
    fn finalize_in_progress_picture(&mut self) -> Result<()> {
        let Some(in_progress) = self.in_progress.take() else {
            return Ok(());
        };
        // §7.4.1.2.4 — a primary coded picture whose every slice
        // failed parse / reconstruction is dropped here, with no DPB
        // update and no `VideoFrame` emitted. Pushing the
        // never-painted picture (zeroed planes plus stale neighbour
        // residue) would diverge from common H.264 decoders, which reject the
        // access unit outright when every slice fails. Caught by
        // fuzz oracle on `crash-2ad9589f…` (3 non-IDR slices, all
        // fail "CABAC read past end of bitstream").
        if !in_progress.any_slice_succeeded {
            let live: Vec<u32> = self
                .picture_frontend
                .references()
                .iter()
                .map(|e| e.dpb_key)
                .collect();
            self.ref_store.retain_keys(&live);
            return Ok(());
        }
        // §7.4.2.1 / Annex A — a coded picture must cover every
        // macroblock 0..PicSizeInMbs in decoding order. Each
        // successful slice marks its walked MBs as
        // `MbInfo::available = true`; an in-progress picture whose
        // MbGrid still has unavailable entries at finalize time means
        // the slice walk stopped short (CABAC end_of_slice_flag fired
        // before reaching the picture's last MB, or no later slice
        // resumed the walk) and the picture is incomplete. common H.264 decoders
        // refuses to emit such pictures (it either conceals or returns
        // Invalid Data from `avcodec_send_packet`); we mirror that by
        // dropping the in-progress picture here rather than emitting a
        // `Frame::Video` with the missing-MB remainder still
        // zero-initialised. Caught by the `ffmpeg_oracle_decode` fuzz
        // target on `crash-b20f4127…` (round 91): a 48x2048
        // (3×128 = 384-MB) picture whose two non-IDR slices each
        // walked ~4 MBs before CABAC-end then handed off to the next
        // slice — total coverage ≪ PicSizeInMbs, leaving most of the
        // luma + chroma planes zero on output.
        if in_progress.grid.info.iter().any(|m| !m.available) {
            let live: Vec<u32> = self
                .picture_frontend
                .references()
                .iter()
                .map(|e| e.dpb_key)
                .collect();
            self.ref_store.retain_keys(&live);
            return Ok(());
        }
        let PictureInProgress {
            prepared,
            mut pic,
            grid,
            first_nal_unit_type: _,
            first_nal_ref_idc: _,
            first_header,
            is_reference: _,
            is_idr,
            poc,
            structure: _,
            pts,
            time_base,
            deblock_enabled,
            deblock_alpha_off,
            deblock_beta_off,
            mb_field_flags,
            sps,
            pps,
            any_slice_succeeded: _,
        } = in_progress;

        // §8.4.1.2.3 temporal direct needs the colocated block's MVs
        // of any B slice that references this picture. Snapshot the
        // decoded MV grid into the Picture so `ref_store` carries it
        // forward. Idempotent: overwrites whatever was in the Picture
        // before.
        snapshot_grid_into_picture(&mut pic, &grid);

        // SPS and PPS were snapshotted at the first slice's header-parse
        // time (via [`Event::Slice`]). Using the driver's current
        // `active_pps()` here would be wrong whenever a later NAL
        // overwrites `pps_by_id[id]` before the picture is finalized
        // — as happens in JVT CACQP3 where every access unit re-sends
        // PPS id 0 with a different `chroma_qp_index_offset`.

        // §8.7 — one picture-level deblocking pass, AFTER every slice of
        // this primary coded picture has populated the shared
        // Picture + MbGrid. Running it per-slice (the old behaviour)
        // would re-filter already-deblocked edges when a later slice's
        // pass revisits them with more MBs marked `available`, corrupting
        // the pixels near slice boundaries. Multi-slice pictures are
        // exercised by the JVT SVA_Base_B (CAVLC IP, 3 slices/pic) and
        // SL1_SVA_B (CAVLC IPB, 3 slices/pic) conformance streams.
        let bit_depth_y = 8 + sps.bit_depth_luma_minus8;
        let bit_depth_c = 8 + sps.bit_depth_chroma_minus8;
        let mbaff_frame_flag = sps.mb_adaptive_frame_field_flag && !first_header.field_pic_flag;
        if deblock_enabled {
            reconstruct::deblock_picture_full(
                &mut pic,
                &grid,
                deblock_alpha_off,
                deblock_beta_off,
                bit_depth_y,
                bit_depth_c,
                &pps,
                mbaff_frame_flag,
                first_header.field_pic_flag,
                &mb_field_flags,
            );
        }

        // Ordinary frame-coded pictures freeze before the shared frontend is
        // committed. This keeps the handoff transactional: a (theoretical)
        // arena/header failure cannot advance POC/DPB state. PAFF deliberately
        // stays mutable because its approved fallback interleaves/copies fields.
        let output_arena = if first_header.field_pic_flag {
            None
        } else {
            Some(pic.freeze(pts)?)
        };

        // §8.2.5 / §8.2.1 cross-picture state is committed only after
        // reconstruction, deblocking and the normal-path arena freeze succeed.
        let commit = self.picture_frontend.commit(prepared);
        let mmco5_triggered = commit.mmco5;
        let current_ref_key = commit.current_dpb_entry.as_ref().map(|entry| {
            if mmco5_triggered {
                // Keep the stored Picture's identity aligned with the
                // frontend's post-MMCO5 DPB descriptor.
                pic.pic_order_cnt = entry.pic_order_cnt;
                pic.frame_num = entry.frame_num;
            }
            entry.dpb_key
        });
        let live_keys: Vec<u32> = self
            .picture_frontend
            .references()
            .iter()
            .map(|e| e.dpb_key)
            .collect();

        // `time_base` was previously stamped onto the VideoFrame for
        // downstream rescaling; the slim VideoFrame shape only carries
        // pts + planes now, so the time base lives on the stream's
        // CodecParameters instead. Bind to `_` to keep the destructure
        // total and document the intent.
        let _ = time_base;

        // §C.4 — at IDR / MMCO-5 drain the prior sequence. A leftover
        // unpaired field from the previous sequence can never be paired
        // now, so emit it as a half-height frame before the drain.
        if is_idr || mmco5_triggered {
            self.flush_pending_field()?;
            for drained in self.output_dpb.flush() {
                self.ready.push_back(drained.picture);
            }
            self.output_dpb.reset();
        }

        self.ensure_output_dpb_sized(&sps);

        // §8.2.1 NOTE 1 — after decoding of the MMCO-5 picture:
        //   tempPicOrderCnt = PicOrderCnt(CurrPic);
        //   TopFieldOrderCnt -= tempPicOrderCnt;
        //   BottomFieldOrderCnt -= tempPicOrderCnt;
        // For a frame (field_pic_flag == 0 and TopFOC == BotFOC) this
        // zeros both, so PicOrderCnt(CurrPic) for output ordering
        // becomes 0. Downstream POC-ordered bumping MUST see the
        // post-reset POC — otherwise the MMCO-5 picture is sorted by
        // its pre-reset POC (ordinarily the largest in the outgoing
        // CVS) and later CVS pictures, which use POC starting from 0
        // again, are bumped ahead of it.
        let output_poc = commit
            .current_dpb_entry
            .as_ref()
            .filter(|_| mmco5_triggered)
            .map(|e| e.pic_order_cnt)
            .unwrap_or(poc.pic_order_cnt);

        // §C.4.4 — PAFF remains the explicitly approved copy fallback. A
        // field can be referenced before its complementary partner arrives, so
        // keep an independent compact copy in RefPicStore and let the pairing
        // path materialise/interleave output separately.
        if first_header.field_pic_flag {
            if let Some(key) = current_ref_key {
                self.ref_store.insert(key, pic.deep_copy());
            }
            self.ref_store.retain_keys(&live_keys);
            self.handle_field_output(
                pic,
                first_header.bottom_field_flag,
                first_header.frame_num,
                output_poc,
                pts,
            )?;
            return Ok(());
        }

        // Normal frame-coded path: the picture was frozen exactly once above.
        // The output lease and RefPicStore now retain the same arena allocation;
        // only H.264 metadata is duplicated/owned separately from the samples.
        let arena = output_arena.expect("non-field picture was frozen");
        let output_lease = FrameLease::from_arena_video(Arc::clone(&arena));
        if let Some(key) = current_ref_key {
            self.ref_store.insert(key, pic);
        }
        self.ref_store.retain_keys(&live_keys);
        self.push_output_lease(output_lease, output_poc, first_header.frame_num);

        Ok(())
    }

    /// §C.4.4 — accept a finalized PAFF field. If it completes a
    /// complementary pair with a previously-held field (same `frame_num`,
    /// opposite parity), interleave the two half-height field pictures
    /// into one full-height output frame and push it to the §C.4 output
    /// DPB. Otherwise hold the field as the pending half of a pair.
    fn handle_field_output(
        &mut self,
        pic: Picture,
        is_bottom: bool,
        frame_num: u32,
        field_poc: i32,
        pts: Option<i64>,
    ) -> Result<()> {
        if let Some(prev) = self.pending_field.take() {
            // Complete the pair only when the two fields are genuinely
            // complementary (opposite parity, same frame_num). A second
            // same-parity field, or a field with a different frame_num,
            // means the first field was unpaired — emit it on its own and
            // start a fresh pending pair with the current field.
            if prev.is_bottom != is_bottom && prev.frame_num == frame_num {
                let (top, bottom) = if prev.is_bottom {
                    (&pic, &prev.pic)
                } else {
                    (&prev.pic, &pic)
                };
                let internal = self.internal_arena_identities(&self.assembly_pool);
                let cancellation = self.cancellation.clone();
                let mut frame = interleave_fields_in_pool(
                    &mut self.assembly_pool,
                    top,
                    bottom,
                    cancellation.as_ref(),
                    &internal,
                )?;
                // §8.2.1 eq. 8-1 — PicOrderCnt(frame) =
                // Min(TopFieldOrderCnt, BottomFieldOrderCnt).
                let frame_poc = prev.field_poc.min(field_poc);
                let frame_pts = prev.pts.or(pts);
                let arena = frame.freeze(frame_pts)?;
                self.push_output_lease(FrameLease::from_arena_video(arena), frame_poc, frame_num);
                return Ok(());
            }
            // Not complementary — flush the orphaned previous field.
            self.emit_unpaired_field(prev)?;
        }
        self.pending_field = Some(PendingField {
            pic,
            is_bottom,
            frame_num,
            field_poc,
            pts,
        });
        Ok(())
    }

    /// Emit a leftover (unpaired) field as a standalone half-height frame,
    /// pushed through the §C.4 output DPB in POC order. The original field
    /// allocation is frozen directly; no legacy VideoFrame is materialised.
    fn emit_unpaired_field(&mut self, mut field: PendingField) -> Result<()> {
        let arena = field.pic.freeze(field.pts)?;
        self.push_output_lease(
            FrameLease::from_arena_video(arena),
            field.field_poc,
            field.frame_num,
        );
        Ok(())
    }

    /// §C.4.4 — flush any pending unpaired field (e.g. at IDR / EOF). The
    /// field is emitted as a standalone half-height arena frame.
    fn flush_pending_field(&mut self) -> Result<()> {
        if let Some(field) = self.pending_field.take() {
            self.emit_unpaired_field(field)?;
        }
        Ok(())
    }

    fn push_output_lease(&mut self, picture: FrameLease, pic_order_cnt: i32, frame_num: u32) {
        let entry = OutputEntry {
            picture,
            pic_order_cnt,
            frame_num,
            needed_for_output: true,
        };
        if let Some(bumped) = self.output_dpb.push(entry) {
            self.ready.push_back(bumped.picture);
        }
    }

    /// Resize the output DPB capacity from the active SPS's VUI
    /// `bitstream_restriction` (§E.2.1) when present, else fall back
    /// to the Annex A Table A-1 per-level default derived from
    /// `MaxDpbMbs` (§A.3.1 item h). Only rebuilds the internal queue
    /// when the capacity would actually change — the common case of a
    /// steady SPS is a cheap no-op.
    fn ensure_output_dpb_sized(&mut self, sps: &Sps) {
        let (reorder, buffering) = output_dpb_sizing(sps);
        if self.output_dpb.max_num_reorder_frames != reorder
            || self.output_dpb.max_dec_frame_buffering != buffering
        {
            // The DpbOutput has no "resize" primitive, so move any
            // entries currently queued into a fresh DpbOutput with the
            // new capacity. Using push() on the new queue preserves
            // §C.4 bumping semantics — if the new cap is smaller, the
            // excess is pushed to `ready` in POC order.
            let pending = self.output_dpb.flush();
            let mut new_dpb = DpbOutput::<FrameLease>::new(reorder, buffering);
            // Iterate in the POC-ascending order flush() produced.
            for e in pending {
                if let Some(bumped) = new_dpb.push(e) {
                    self.ready.push_back(bumped.picture);
                }
            }
            self.output_dpb = new_dpb;
        }
    }
    /// Keep the reusable sample pool compatible with the active SPS. A new
    /// pool is cheap because arenas are allocated lazily; frames already
    /// leased from the previous pool retain their storage independently.
    fn ensure_picture_pool_sized(&mut self, sps: &Sps) {
        let required = Picture::required_bytes(
            sps.pic_width_in_mbs() * 16,
            sps.frame_height_in_mbs() * 16,
            sps.chroma_array_type(),
            sps.bit_depth_luma_minus8 + 8,
            sps.bit_depth_chroma_minus8 + 8,
        )
        .saturating_add(H264_PICTURE_ARENA_PADDING)
        .max(1);
        if self.picture_pool.cap_per_arena() != required
            || self.picture_pool.max_arenas() != H264_PICTURE_POOL_MAX_ARENAS
        {
            self.picture_pool = ArenaPool::new(H264_PICTURE_POOL_MAX_ARENAS, required);
        }
    }

    fn internal_arena_identities(&self, pool: &Arc<ArenaPool>) -> HashSet<ArenaIdentity> {
        let mut ids = HashSet::new();

        if let Some(in_progress) = self.in_progress.as_ref() {
            insert_picture_arena_identity(&mut ids, pool, &in_progress.pic);
        }
        if let Some(field) = self.pending_field.as_ref() {
            insert_picture_arena_identity(&mut ids, pool, &field.pic);
        }
        for picture in self.ref_store.pictures() {
            insert_picture_arena_identity(&mut ids, pool, picture);
        }
        for entry in self.output_dpb.iter() {
            insert_lease_arena_identity(&mut ids, pool, &entry.picture);
        }
        for lease in &self.ready {
            insert_lease_arena_identity(&mut ids, pool, lease);
        }
        for id in &self.extra_internal_arenas {
            if pool.owns_identity(*id) {
                ids.insert(*id);
            }
        }
        ids
    }

    fn lease_with_backpressure(&self, pool: &Arc<ArenaPool>, purpose: &str) -> Result<Arena> {
        let internal = self.internal_arena_identities(pool);
        lease_pool_with_backpressure(pool, self.cancellation.as_ref(), &internal, purpose)
    }

    fn allocate_picture(
        &mut self,
        width_samples: u32,
        height_samples: u32,
        chroma_array_type: u32,
        bit_depth_y: u32,
        bit_depth_c: u32,
    ) -> Result<Picture> {
        let arena = self.lease_with_backpressure(&self.picture_pool, "picture")?;
        Picture::new_in_arena(
            arena,
            width_samples,
            height_samples,
            chroma_array_type,
            bit_depth_y,
            bit_depth_c,
        )
    }
}

/// §8.2.5.2 — mid-grey placeholder picture for a synthetic non-existing
/// reference frame. Samples are set to `2^(bit_depth - 1)` per plane so
/// that accidental motion-compensation references produce neutral
/// output instead of zeroes (which would bias the residual).
#[cfg(test)]
fn gray_picture(
    width_samples: u32,
    height_samples: u32,
    chroma_array_type: u32,
    bit_depth_y: u32,
    bit_depth_c: u32,
) -> Picture {
    let mut p = Picture::new(
        width_samples,
        height_samples,
        chroma_array_type,
        bit_depth_y,
        bit_depth_c,
    );
    p.non_existing = true;
    let grey_y: i32 = 1 << (bit_depth_y.saturating_sub(1));
    let grey_c: i32 = 1 << (bit_depth_c.saturating_sub(1));
    p.fill_luma(grey_y);
    p.fill_cb(grey_c);
    p.fill_cr(grey_c);
    p
}

/// Per-slice [`RefPicProvider`] that borrows pictures from a long-running
/// [`RefPicStore`] but carries its own RefPicList0 / RefPicList1 key
/// arrays. Avoids cloning every DPB Picture on every slice.
/// Round-416 PAFF — per-field-list POC / long-term metadata:
/// (list0 per-FIELD POCs, list0 long-term flags, list1 POCs, list1
/// long-term flags), produced by `resolve_field_list`.
type FieldListPocsLt = (Vec<i32>, Vec<bool>, Vec<i32>, Vec<bool>);

/// Round-416 PAFF — one resolved §8.2.4.2.5 field reference list:
/// parallel per-index vectors (see `resolve_field_list`).
struct ResolvedFieldList {
    keys: Vec<u32>,
    overrides: Vec<Option<Picture>>,
    pocs: Vec<i32>,
    longterm: Vec<bool>,
    /// Parity of each reference FIELD (0 = top, 1 = bottom) for the
    /// §8.4.1.4 Table 8-10 chroma-MV adjustment.
    parities: Vec<Option<u8>>,
    /// §8.4.1.2.3 MapColToList0 — the frame-level UNIT key containing
    /// each field entry (a coded field of a complementary pair maps to
    /// the pair's unit key = the top field's storage key; a field of a
    /// stored frame maps to the frame's key).
    unit_keys: Vec<u32>,
}

struct BorrowedRefProvider<'a> {
    store: &'a RefPicStore,
    list_0: &'a [u32],
    list_1: &'a [u32],
    /// §8.4.1.2.3 — precomputed POCs of the pictures in `list_0`,
    /// supplied to the temporal-direct MapColToList0 derivation.
    list_0_pocs: Vec<i32>,
    /// §8.4.1.2.3 — long-term flag parallel to `list_0_pocs`.
    list_0_longterm: Vec<bool>,
    /// §8.4.1.2.2 — long-term flag for RefPicList1. Spatial-direct
    /// mode suppresses colZeroFlag when `RefPicList1[0]` is long-term.
    list_1_longterm: Vec<bool>,
    /// Round-416 PAFF — per-index owned field-view pictures for field
    /// slices whose §8.2.4.2.5 list entry names one parity field of a
    /// picture stored as a FRAME (`Picture::field_view` materialised at
    /// slice setup, stamped with the field's own POC). `Some` entries
    /// shadow the key-based `ref_store` lookup at that index; entries
    /// resolving to stored coded fields stay `None` and read the
    /// half-height stored picture directly.
    list_0_overrides: Vec<Option<Picture>>,
    list_1_overrides: Vec<Option<Picture>>,
    /// Round-416 PAFF — §8.4.1.4 Table 8-10: per-index parity of the
    /// reference FIELD (0 = top, 1 = bottom) for field slices; empty
    /// for frame slices.
    list_0_parities: Vec<Option<u8>>,
    list_1_parities: Vec<Option<u8>>,
    /// §8.4.1.2.3 MapColToList0 — frame-level unit key per entry.
    list_0_unit_keys: Vec<u32>,
    list_1_unit_keys: Vec<u32>,
    /// §8.4.1.2.3 — per-entry (TopFOC, BottomFOC) of the entry's unit
    /// for FRAME slices (per-field pocs duplicated for field slices).
    list_0_focs: Vec<(i32, i32)>,
    list_1_focs: Vec<(i32, i32)>,
    /// §8.4.1.2.1 Table 8-6 — (top, bottom) stored-field keys of
    /// complementary-PAIR units in a FRAME slice's lists.
    list_0_pair_keys: Vec<Option<(u32, u32)>>,
    list_1_pair_keys: Vec<Option<(u32, u32)>>,
}

impl RefPicProvider for BorrowedRefProvider<'_> {
    fn ref_pic(&self, list: u8, idx: u32) -> Option<&Picture> {
        let (keys, overrides) = match list {
            0 => (self.list_0, &self.list_0_overrides),
            1 => (self.list_1, &self.list_1_overrides),
            _ => return None,
        };
        if let Some(Some(p)) = overrides.get(idx as usize) {
            return Some(p);
        }
        let key = *keys.get(idx as usize)?;
        self.store.get_by_key(key)
    }

    fn ref_field_parity(&self, list: u8, idx: u32) -> Option<u8> {
        let parities = match list {
            0 => &self.list_0_parities,
            1 => &self.list_1_parities,
            _ => return None,
        };
        parities.get(idx as usize).copied().flatten()
    }

    fn ref_list_0_pocs(&self) -> &[i32] {
        &self.list_0_pocs
    }

    fn ref_list_0_longterm(&self) -> &[bool] {
        &self.list_0_longterm
    }

    fn ref_list_1_longterm(&self) -> &[bool] {
        &self.list_1_longterm
    }

    fn ref_list_0_keys(&self) -> &[u32] {
        self.list_0
    }

    fn ref_list_0_unit_keys(&self) -> &[u32] {
        &self.list_0_unit_keys
    }

    fn ref_list_0_parities(&self) -> &[Option<u8>] {
        &self.list_0_parities
    }

    fn ref_entry_identity(&self, list: u8, idx: u32) -> Option<(u32, Option<u8>, u32)> {
        let (keys, parities, unit_keys) = match list {
            0 => (self.list_0, &self.list_0_parities, &self.list_0_unit_keys),
            1 => (self.list_1, &self.list_1_parities, &self.list_1_unit_keys),
            _ => return None,
        };
        let key = *keys.get(idx as usize)?;
        let parity = parities.get(idx as usize).copied().flatten();
        let unit = unit_keys.get(idx as usize).copied().unwrap_or(key);
        Some((key, parity, unit))
    }

    fn ref_entry_unit_focs(&self, list: u8, idx: u32) -> Option<(i32, i32)> {
        let focs = match list {
            0 => &self.list_0_focs,
            1 => &self.list_1_focs,
            _ => return None,
        };
        focs.get(idx as usize).copied()
    }

    fn ref_pair_field(&self, list: u8, idx: u32, bottom: bool) -> Option<&Picture> {
        let pair_keys = match list {
            0 => &self.list_0_pair_keys,
            1 => &self.list_1_pair_keys,
            _ => return None,
        };
        let (top_key, bottom_key) = (*pair_keys.get(idx as usize)?)?;
        self.store
            .get_by_key(if bottom { bottom_key } else { top_key })
    }
}

/// Derive the `(max_num_reorder_frames, max_dec_frame_buffering)`
/// pair for sizing [`DpbOutput`] from the active SPS.
///
/// Two sources of sizing information:
///   * **VUI `bitstream_restriction`** (§E.2.1) — encoder's explicit
///     claim about how many pictures the decoder must hold.
///   * **Level-derived cap** (§A.3.1 item h, Table A-1) —
///     `Min(MaxDpbMbs / (PicWidthInMbs * FrameHeightInMbs), 16)`,
///     the upper bound the decoder is *capable* of holding for the
///     declared profile/level + picture size.
///
/// Selection policy
/// ----------------
/// Real-world encoders routinely emit `max_num_reorder_frames`
/// values smaller than the actual reorder depth their stream uses
/// — solana-ad's High@L3.1 720p run is a textbook case (claims
/// reorder=2 in VUI, decodes a 4-frame B-pyramid that needs
/// reorder=4 to bump in POC order). Honoring the encoder's
/// undersized claim forces §C.4 to bump pictures before later POCs
/// arrive, emitting frames in something close to *decode* order
/// rather than display order.
///
/// Per §A.3.1 the level-derived cap is always a valid decoder
/// commitment — the spec lets a decoder hold up to that many
/// pictures regardless of what `bitstream_restriction` claims. We
/// therefore use it as a *floor* on the reorder window: trust the
/// VUI when it asks us to hold *more*, but raise to the level cap
/// when it asks for *fewer*. This is exactly the common-decoder
/// "max_num_reorder_frames is a lower bound on what we can buffer"
/// real-world reading and what makes B-pyramid streams from
/// permissive encoders decode in display order.
///
/// `max_dec_frame_buffering` follows the same logic: it must be at
/// least as large as the reorder window we settle on, and never
/// below the level-derived buffering cap.
///
/// Both values floor at 1 so [`DpbOutput::push`] can always bump
/// when the queue is full (cap of 0 would deadlock the queue).
fn output_dpb_sizing(sps: &Sps) -> (u32, u32) {
    // Level-derived cap (§A.3.1 item h, Annex A Table A-1).
    // §A.3.4.1 — bit 3 of `constraint_set_flags` is constraint_set3_flag,
    // which in combination with `level_idc == 11` signals Level 1b
    // (MaxDpbMbs = 396, same as Level 1) rather than Level 1.1
    // (MaxDpbMbs = 900). For all other level_idc values the flag is
    // ignored.
    let constraint_set3_flag = (sps.constraint_set_flags & 0b0000_1000) != 0;
    let max_dpb_mbs = max_dpb_mbs_for_level(sps.level_idc, constraint_set3_flag);
    let pic_size_mbs = sps
        .pic_width_in_mbs()
        .saturating_mul(sps.frame_height_in_mbs())
        .max(1);
    let level_cap = (max_dpb_mbs / pic_size_mbs).clamp(1, 16);

    if let Some(br) = sps
        .vui
        .as_ref()
        .and_then(|v| v.bitstream_restriction.as_ref())
    {
        // §E.2.1 — encoder claim, but raise to the level cap when
        // it's smaller. See doc-comment for the rationale.
        let reorder = br.max_num_reorder_frames.max(level_cap);
        let buffering = br
            .max_dec_frame_buffering
            .max(reorder)
            .max(level_cap)
            .max(1);
        return (reorder, buffering);
    }

    // §A.3.1 item j — bitstream_restriction absent → both default
    // to the level-derived buffering cap.
    (level_cap, level_cap)
}

/// Annex A Table A-1 — `MaxDpbMbs` per `level_idc`.
///
/// `level_idc` is the raw `u8` from §7.4.2.1.1; intermediate levels
/// (e.g. 2.1) are encoded as `10 * <level>` so `21 → 2.1`.
///
/// **Level 1b disambiguation**: `level_idc == 11` is shared between
/// Level 1.1 (MaxDpbMbs = 900) and Level 1b (MaxDpbMbs = 396, same as
/// Level 1). The two are distinguished by `constraint_set3_flag`
/// (§A.3.4.1) — when set, the stream signals Level 1b. For Baseline /
/// Constrained Baseline the spec also allows `level_idc == 9` to mean
/// Level 1b directly; we honour that path too.
///
/// Unknown `level_idc` values fall through to the lowest bucket
/// (MaxDpbMbs = 396) to minimise over-allocation while still
/// permitting at least one reference picture.
fn max_dpb_mbs_for_level(level_idc: u8, constraint_set3_flag: bool) -> u32 {
    match level_idc {
        9 => 396,  // Level 1b (Baseline / Constrained Baseline shorthand)
        10 => 396, // level 1
        11 => {
            if constraint_set3_flag {
                396 // Level 1b
            } else {
                900 // Level 1.1
            }
        }
        12 => 2_376, // level 1.2
        13 => 2_376, // level 1.3
        20 => 2_376, // level 2
        21 => 4_752, // level 2.1
        22 => 8_100, // level 2.2
        30 => 8_100, // level 3
        31 => 18_000,
        32 => 20_480,
        40 => 32_768,
        41 => 32_768,
        42 => 34_816,
        50 => 110_400,
        51 => 184_320,
        52 => 184_320,
        60 => 696_320,
        61 => 696_320,
        62 => 696_320,
        _ => 396, // conservative fallback
    }
}

/// Convert §7.3.3.1 RPLM op to the §8.2.4.3 ref_list equivalent.
fn slice_rplm_to_ref_rplm(op: &SliceRplmOp) -> RplmOp {
    match *op {
        SliceRplmOp::Subtract(v) => RplmOp::Subtract(v),
        SliceRplmOp::Add(v) => RplmOp::Add(v),
        SliceRplmOp::LongTerm(v) => RplmOp::LongTerm(v),
    }
}

impl H264CodecDecoder {
    /// Feed one complete Annex-B access unit through the software parser.
    /// Timing comes from the packet that began this assembled AU rather than
    /// from whichever container/PES packet happened to complete it.
    fn feed_annex_b_access_unit(&mut self, packet: &Packet) -> Result<()> {
        self.pending_pts = packet.pts;
        self.pending_time_base = packet.time_base;

        // Collect events first so the parser borrow ends before handle_event
        // re-borrows decoder state during reconstruction.
        let events: Vec<_> = self.driver.process_annex_b(&packet.data).collect();
        for ev in events {
            match ev {
                Ok(ev) => {
                    if let Err(e) = self.handle_event(ev) {
                        if e.is_cancelled() || e.is_resource_exhausted() {
                            return Err(e);
                        }
                        self.decode_errors += 1;
                        log::warn!("h264 slice skipped: {e}");
                    }
                }
                Err(e) => return Err(Error::invalid(format!("h264 NAL parse: {e}"))),
            }
        }
        Ok(())
    }
    fn pop_frame_lease(&mut self) -> Result<FrameLease> {
        // §C.4 — bumped / already-released pictures come out first in the
        // order the bumping process produced them.
        if let Some(lease) = self.ready.pop_front() {
            return Ok(lease);
        }
        // Conservative mid-stream bumping.
        if let Some(bumped) = self.output_dpb.pop_ready() {
            return Ok(bumped.picture);
        }
        // EOF drains the remaining output DPB exactly once into `ready`.
        if self.eof {
            let drained = self.output_dpb.flush();
            if drained.is_empty() {
                return Err(Error::Eof);
            }
            for entry in drained {
                self.ready.push_back(entry.picture);
            }
            return self.ready.pop_front().ok_or(Error::Eof);
        }
        Err(Error::NeedMore)
    }
}

impl Decoder for H264CodecDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> Option<&CodecParameters> {
        Some(&self.output_params)
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        match self.length_size {
            Some(n) => {
                // AVCC framing is already packetised as explicit NAL lengths;
                // keep the native path and do not insert Annex-B preprocessing.
                self.pending_pts = packet.pts;
                self.pending_time_base = packet.time_base;
                let data = &packet.data;
                let mut i = 0usize;
                let n = n as usize;
                while i < data.len() {
                    if i + n > data.len() {
                        return Err(Error::invalid("h264: AVCC length prefix truncated"));
                    }
                    let mut len = 0usize;
                    for k in 0..n {
                        len = (len << 8) | data[i + k] as usize;
                    }
                    i += n;
                    if i + len > data.len() {
                        return Err(Error::invalid("h264: AVCC NAL payload truncated"));
                    }
                    let ev = self
                        .driver
                        .process_nal(&data[i..i + len])
                        .map_err(|e| Error::invalid(format!("h264 NAL parse: {e}")))?;
                    if let Err(e) = self.handle_event(ev) {
                        if e.is_cancelled() || e.is_resource_exhausted() {
                            return Err(e);
                        }
                        self.decode_errors += 1;
                        log::warn!("h264 slice skipped: {e}");
                    }
                    i += len;
                }
                Ok(())
            }
            None => {
                // Annex-B is a byte stream, not a container-packet format.
                // Opt into the shared assembler so a PES boundary may split a
                // NAL/access unit without forcing every decoder to duplicate
                // this buffering logic. Decoders with their own streaming
                // parser (e.g. cuvidParser) simply do not use this helper.
                let completed = self.au_assembler.push(packet)?;
                for access_unit in completed {
                    self.feed_annex_b_access_unit(&access_unit)?;
                }
                Ok(())
            }
        }
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        self.pop_frame_lease()?.into_frame()
    }

    fn receive_frame_lease(&mut self) -> Result<FrameLease> {
        self.pop_frame_lease()
    }

    fn set_cancellation_token(&mut self, token: CancellationToken) {
        self.cancellation = Some(token.clone());
        if let Some(scp) = self.scp.as_mut() {
            for sub in scp.subs.iter_mut() {
                sub.set_cancellation_token(token.clone());
            }
        }
    }

    fn flush(&mut self) -> Result<()> {
        if self.length_size.is_none() {
            if let Some(access_unit) = self.au_assembler.flush() {
                self.feed_annex_b_access_unit(&access_unit)?;
            }
        }
        // §7.3.2.9 — decode any partitioned slice still waiting for
        // (possibly absent) partition-B/C payloads at EOF.
        self.flush_pending_dp_slice()?;
        // §7.4.1.2 — close any picture we've been assembling so it reaches
        // the DPB + output queue before the caller drains at EOF.
        if let Err(e) = self.finalize_in_progress_picture() {
            if e.is_cancelled() || e.is_resource_exhausted() {
                return Err(e);
            }
            self.decode_errors += 1;
            log::warn!("h264 flush: final picture skipped: {e}");
        }
        // §C.4.4 — a trailing unpaired PAFF field at EOF can never gain a
        // complementary partner; emit it as a standalone half-height
        // frame so it is not silently dropped.
        self.flush_pending_field()?;
        // §8.1 — flush the three separate-colour-plane sub-decoders and
        // merge their drained plane pictures into three-plane frames.
        if let Some(scp) = self.scp.as_mut() {
            for index in 0..scp.subs.len() {
                scp.subs[index].extra_internal_arenas = queue_arena_identities(&scp.queues[index]);
                scp.subs[index].flush()?;
            }
            self.drain_and_merge_scp()?;
        }
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.driver = H264Driver::new();
        self.last_slice = None;
        self.decode_errors = 0;
        self.eof = false;
        // §C.4 — wipe the output queue and any picture that was
        // already bumped but not yet consumed.
        self.output_dpb.reset();
        self.ready.clear();
        self.pending_pts = None;
        self.ref_store = RefPicStore::new();
        self.au_assembler.reset();
        self.picture_frontend.reset();
        // Drop any picture currently being assembled — reset implies we
        // discard in-flight state, not deliver it.
        self.in_progress = None;
        self.pending_field = None;
        // Detach the next decoding session from arena buffers retained by
        // application leases obtained before reset(). Those old leases remain
        // valid through their old Arc<ArenaPool>; the fresh pool can make
        // progress independently even if every old slot is still retained.
        let arena_cap = self.picture_pool.cap_per_arena().max(1);
        self.picture_pool = ArenaPool::new(H264_PICTURE_POOL_MAX_ARENAS, arena_cap);
        let assembly_cap = self.assembly_pool.cap_per_arena().max(1);
        self.assembly_pool = ArenaPool::new(H264_PICTURE_POOL_MAX_ARENAS, assembly_cap);
        self.extra_internal_arenas.clear();
        // §8.1 — drop the separate-colour-plane sub-decoders wholesale;
        // a post-reset stream re-creates them at its first SCP slice.
        self.scp = None;
        Ok(())
    }
}

/// §8.4.1.2.3 — snapshot the per-4x4-block motion data from the
/// freshly-decoded picture's [`MbGrid`] into its [`Picture`] so
/// subsequent B slices can consult the colocated block for temporal
/// direct mode.
///
/// The `Picture`'s sample buffer already contains the decoded samples;
/// this only populates the optional mv / refIdx / intra grids. Called
/// by `finalize_in_progress_picture` whether the picture is a
/// reference or not — non-reference pictures never feed a later B
/// slice, but the per-picture cost of the copy is tiny and it keeps
/// the code path uniform.
fn snapshot_grid_into_picture(pic: &mut Picture, grid: &MbGrid) {
    let w = grid.width_in_mbs as usize;
    let h = grid.height_in_mbs as usize;
    let nmb = w * h;
    pic.mb_width_in_picture = grid.width_in_mbs;
    pic.mv_l0_grid = vec![(0i16, 0i16); nmb * 16];
    pic.mv_l1_grid = vec![(0i16, 0i16); nmb * 16];
    pic.ref_idx_l0_grid = vec![-1i8; nmb * 4];
    pic.ref_idx_l1_grid = vec![-1i8; nmb * 4];
    pic.is_intra_grid = vec![false; nmb];
    // §6.4.12.2 / Table 8-8 — `fieldDecodingFlagX` of every MB, for
    // AFRM pictures serving as colPic.
    pic.mb_field_flags = vec![false; nmb];
    for (addr, info) in grid.info.iter().enumerate() {
        let base_mv = addr * 16;
        let base_r = addr * 4;
        for blk4 in 0..16 {
            pic.mv_l0_grid[base_mv + blk4] = info.mv_l0[blk4];
            pic.mv_l1_grid[base_mv + blk4] = info.mv_l1[blk4];
        }
        for blk8 in 0..4 {
            pic.ref_idx_l0_grid[base_r + blk8] = info.ref_idx_l0[blk8];
            pic.ref_idx_l1_grid[base_r + blk8] = info.ref_idx_l1[blk8];
        }
        pic.is_intra_grid[addr] = info.is_intra;
        pic.mb_field_flags[addr] = info.mb_field_decoding_flag;
    }
}

fn ensure_assembly_pool_sized(pool: &mut Arc<ArenaPool>, required: usize) {
    if pool.cap_per_arena() != required || pool.max_arenas() != H264_PICTURE_POOL_MAX_ARENAS {
        *pool = ArenaPool::new(H264_PICTURE_POOL_MAX_ARENAS, required);
    }
}

fn insert_picture_arena_identity(
    ids: &mut HashSet<ArenaIdentity>,
    pool: &Arc<ArenaPool>,
    picture: &Picture,
) {
    let id = picture.arena_identity();
    if pool.owns_identity(id) {
        ids.insert(id);
    }
}

fn insert_lease_arena_identity(
    ids: &mut HashSet<ArenaIdentity>,
    pool: &Arc<ArenaPool>,
    lease: &FrameLease,
) {
    if let Some(frame) = lease.as_arena_video() {
        let id = frame.arena_identity();
        if pool.owns_identity(id) {
            ids.insert(id);
        }
    }
}

fn queue_arena_identities(queue: &VecDeque<FrameLease>) -> HashSet<ArenaIdentity> {
    queue
        .iter()
        .filter_map(FrameLease::as_arena_video)
        .map(|frame| frame.arena_identity())
        .collect()
}

fn lease_pool_with_backpressure(
    pool: &Arc<ArenaPool>,
    cancellation: Option<&CancellationToken>,
    internal_ids: &HashSet<ArenaIdentity>,
    purpose: &str,
) -> Result<Arena> {
    match pool.lease() {
        Ok(arena) => return Ok(arena),
        Err(Error::ResourceExhausted(_)) => {}
        Err(error) => return Err(error),
    }

    if cancellation.is_some_and(CancellationToken::is_cancelled) {
        return Err(Error::cancelled(format!(
            "h264 {purpose} arena wait cancelled"
        )));
    }

    // `ArenaPool::lease()` only reports exhaustion when every one of its
    // `max_arenas` slots is checked out. If the decoder itself retains all of
    // those unique allocations, no other thread can make the wait progress:
    // the decoder would be sleeping on resources only it can release by
    // continuing. Reject that state rather than deadlocking indefinitely.
    let internal = internal_ids
        .iter()
        .filter(|id| pool.owns_identity(**id))
        .count();
    if internal >= pool.max_arenas() {
        return Err(Error::resource_exhausted(format!(
            "h264 {purpose} arena wait would self-deadlock: decoder retains all \
             {} pool arenas",
            pool.max_arenas()
        )));
    }

    let Some(cancellation) = cancellation else {
        return Err(Error::resource_exhausted(format!(
            "h264 {purpose} arena pool exhausted without a cancellation-aware \
             execution context; refusing an indefinite wait"
        )));
    };
    pool.lease_wait_cancellable(cancellation)
}

fn merge_separate_colour_planes(
    pool: &mut Arc<ArenaPool>,
    y: &FrameLease,
    cb: &FrameLease,
    cr: &FrameLease,
    cancellation: Option<&CancellationToken>,
    internal_ids: &HashSet<ArenaIdentity>,
) -> Result<FrameLease> {
    let y = y.as_arena_video().ok_or_else(|| {
        Error::other("h264: separate-colour-plane sub-decoder emitted non-arena luma")
    })?;
    let cb = cb.as_arena_video().ok_or_else(|| {
        Error::other("h264: separate-colour-plane sub-decoder emitted non-arena Cb")
    })?;
    let cr = cr.as_arena_video().ok_or_else(|| {
        Error::other("h264: separate-colour-plane sub-decoder emitted non-arena Cr")
    })?;

    let yh = y.header();
    let cbh = cb.header();
    let crh = cr.header();
    if yh.width != cbh.width
        || yh.width != crh.width
        || yh.height != cbh.height
        || yh.height != crh.height
        || yh.pixel_format != cbh.pixel_format
        || yh.pixel_format != crh.pixel_format
    {
        return Err(Error::invalid(
            "h264: separate colour planes have mismatched geometry or sample format",
        ));
    }
    let (pixel_format, nominal_bits) = match yh.pixel_format {
        PixelFormat::Gray8 => (PixelFormat::Yuv444P, 8),
        PixelFormat::Gray10Le => (PixelFormat::Yuv444P10Le, 10),
        PixelFormat::Gray12Le => (PixelFormat::Yuv444P12Le, 12),
        PixelFormat::Gray16Le => (PixelFormat::Yuv444P16Le, 16),
        other => {
            return Err(Error::unsupported(format!(
                "h264: unsupported separate-colour-plane arena format {other:?}"
            )));
        }
    };

    let y_plane = y
        .plane(0)
        .ok_or_else(|| Error::invalid("h264: missing SCP luma plane"))?;
    let cb_plane = cb
        .plane(0)
        .ok_or_else(|| Error::invalid("h264: missing SCP Cb plane"))?;
    let cr_plane = cr
        .plane(0)
        .ok_or_else(|| Error::invalid("h264: missing SCP Cr plane"))?;
    let strides = [
        y.plane_stride(0)
            .ok_or_else(|| Error::invalid("h264: missing SCP luma stride"))?,
        cb.plane_stride(0)
            .ok_or_else(|| Error::invalid("h264: missing SCP Cb stride"))?,
        cr.plane_stride(0)
            .ok_or_else(|| Error::invalid("h264: missing SCP Cr stride"))?,
    ];
    let lengths = [y_plane.len(), cb_plane.len(), cr_plane.len()];
    let required = lengths
        .iter()
        .try_fold(0usize, |sum, len| sum.checked_add(*len))
        .and_then(|sum| sum.checked_add(H264_PICTURE_ARENA_PADDING))
        .ok_or_else(|| Error::resource_exhausted("h264: SCP output arena size overflow"))?
        .max(1);
    ensure_assembly_pool_sized(pool, required);
    let arena = lease_pool_with_backpressure(pool, cancellation, internal_ids, "SCP assembly")?;
    let mut builder = VideoFrameBuilder::<u8>::new(arena, &lengths, &strides)?;
    builder
        .plane_mut(0)
        .expect("three output planes")
        .copy_from_slice(y_plane);
    builder
        .plane_mut(1)
        .expect("three output planes")
        .copy_from_slice(cb_plane);
    builder
        .plane_mut(2)
        .expect("three output planes")
        .copy_from_slice(cr_plane);

    let mut header = FrameHeader::new(yh.width, yh.height, pixel_format, yh.presentation_timestamp);
    let precision = [
        yh.significant_bits().and_then(|bits| bits.first()).copied(),
        cbh.significant_bits()
            .and_then(|bits| bits.first())
            .copied(),
        crh.significant_bits()
            .and_then(|bits| bits.first())
            .copied(),
    ];
    if precision.iter().any(Option::is_some) {
        let bits = precision.map(|bits| bits.unwrap_or(nominal_bits));
        header = header.with_significant_bits(&bits)?;
    }
    Ok(FrameLease::from_arena_video(builder.freeze(header)?))
}

/// §C.4.4 / §8.4.2 — re-interleave a complementary pair of half-height
/// field pictures into a single full-height frame.
///
/// The `top` field's row `r` becomes the frame's even row `2*r`; the
/// `bottom` field's row `r` becomes the frame's odd row `2*r + 1`. The
/// two fields are decoded independently (each as a half-height picture)
/// so their luma + chroma plane geometries are identical apart from
/// occupying alternate output lines. The frame inherits the fields' bit
/// depth, chroma format and width.
fn interleave_fields(top: &Picture, bottom: &Picture) -> Picture {
    let required = Picture::required_bytes(
        top.width_in_samples,
        top.height_in_samples.saturating_mul(2),
        top.chroma_array_type,
        top.bit_depth_luma,
        top.bit_depth_chroma,
    )
    .saturating_add(H264_PICTURE_ARENA_PADDING)
    .max(1);
    let mut pool = ArenaPool::new(1, required);
    interleave_fields_in_pool(&mut pool, top, bottom, None, &HashSet::new())
        .expect("standalone PAFF interleave allocation")
}

fn interleave_fields_in_pool(
    pool: &mut Arc<ArenaPool>,
    top: &Picture,
    bottom: &Picture,
    cancellation: Option<&CancellationToken>,
    internal_ids: &HashSet<ArenaIdentity>,
) -> Result<Picture> {
    let w = top.width_in_samples;
    let field_h = top.height_in_samples;
    let frame_h = field_h * 2;
    let required = Picture::required_bytes(
        w,
        frame_h,
        top.chroma_array_type,
        top.bit_depth_luma,
        top.bit_depth_chroma,
    )
    .saturating_add(H264_PICTURE_ARENA_PADDING)
    .max(1);
    ensure_assembly_pool_sized(pool, required);
    let arena = lease_pool_with_backpressure(pool, cancellation, internal_ids, "PAFF assembly")?;
    let mut frame = Picture::new_in_arena(
        arena,
        w,
        frame_h,
        top.chroma_array_type,
        top.bit_depth_luma,
        top.bit_depth_chroma,
    )?;

    // Luma: approved PAFF fallback copy. Widen one source row into scratch,
    // then compact it directly into the parity-selected destination row.
    let wl = w as usize;
    let mut luma_row = vec![0i32; wl];
    for r in 0..field_h as usize {
        top.copy_luma_range_to_i32(r * wl, &mut luma_row);
        frame.copy_luma_range_from_i32((2 * r) * wl, &luma_row);
        bottom.copy_luma_range_to_i32(r * wl, &mut luma_row);
        frame.copy_luma_range_from_i32((2 * r + 1) * wl, &luma_row);
    }

    // Chroma: same explicit PAFF copy on each chroma plane.
    if top.chroma_array_type != 0 {
        let cw = top.chroma_width() as usize;
        let cfh = top.chroma_height() as usize;
        let mut chroma_row = vec![0i32; cw];
        for r in 0..cfh {
            for plane in 0..2u8 {
                top.copy_chroma_range_to_i32(plane, r * cw, &mut chroma_row);
                frame.copy_chroma_range_from_i32(plane, (2 * r) * cw, &chroma_row);
                bottom.copy_chroma_range_to_i32(plane, r * cw, &mut chroma_row);
                frame.copy_chroma_range_from_i32(plane, (2 * r + 1) * cw, &chroma_row);
            }
        }
    }

    frame.pic_order_cnt = top.pic_order_cnt.min(bottom.pic_order_cnt);
    frame.frame_num = top.frame_num;
    Ok(frame)
}

#[cfg(test)]
mod tests {
    //! Unit tests for the §C.2.2 / §C.4 wiring through
    //! [`H264CodecDecoder`]. These tests bypass `handle_slice` and push
    //! fabricated [`VideoFrame`] + POC pairs straight into the output
    //! path so we exercise only the DPB-output plumbing without
    //! needing a real H.264 bitstream with B-frames in the samples
    //! dir. The bumping-logic correctness itself is already covered by
    //! `crate::dpb_output::tests`; what we assert here is that the
    //! wrapper delivers entries in POC order through `receive_frame`,
    //! honours IDR resets (§C.4), and flushes at EOF.
    //!
    //! Spec references:
    //! * §C.2.2 — "Storage and output of decoded pictures"
    //! * §C.4   — "Bumping process"
    //! * §8.2.5.4 — MMCO op 5 resets the DPB
    //! * Annex A Table A-1 — level-derived MaxDpbMbs defaults
    use super::*;
    use crate::dpb_output::OutputEntry;

    /// Build a tiny VideoFrame so tests can track individual pictures
    /// without carrying real pixel data.
    fn vf(tag: u8) -> VideoFrame {
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 1,
                data: vec![tag],
            }],
        }
    }

    /// Test access: pull the single-byte "tag" out of a VideoFrame
    /// planted by `vf`.
    fn vf_tag(f: &Frame) -> u8 {
        match f {
            Frame::Video(v) => v.planes[0].data[0],
            _ => panic!("non-video frame"),
        }
    }

    fn push_entry(dec: &mut H264CodecDecoder, tag: u8, poc: i32, frame_num: u32) {
        let entry = OutputEntry {
            picture: FrameLease::from_frame(Frame::Video(vf(tag))),
            pic_order_cnt: poc,
            frame_num,
            needed_for_output: true,
        };
        if let Some(bumped) = dec.output_dpb.push(entry) {
            dec.ready.push_back(bumped.picture);
        }
    }

    #[test]
    fn frame_coded_decode_shares_arena_between_reference_store_and_output() {
        let data = include_bytes!("../tests/fixtures/mbaff_iframe_128x96.h264");
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let packet = Packet::new(0, TimeBase::new(1, 25), data.to_vec()).with_pts(7);
        dec.send_packet(&packet).expect("send fixture");
        dec.flush().expect("flush fixture");

        let lease = dec.receive_frame_lease().expect("decoded arena lease");
        let output = lease
            .as_arena_video()
            .expect("ordinary frame-coded H.264 must stay arena-backed");
        assert_eq!(output.header().presentation_timestamp, Some(7));

        let reference = dec
            .picture_frontend
            .references()
            .first()
            .expect("IDR retained as a reference");
        let stored = dec
            .ref_store
            .get_by_key(reference.dpb_key)
            .expect("reference samples retained");
        let stored_frame = stored
            .frozen_frame()
            .expect("reference picture shares frozen arena storage");

        assert_eq!(
            output.plane(0).expect("output luma").as_ptr(),
            stored_frame.plane(0).expect("stored luma").as_ptr(),
            "output and DPB reference must retain the exact same pixel allocation",
        );
    }

    /// §C.4 — with `max_num_reorder_frames == 2`, feeding a decode
    /// order that reorders POC mid-stream must eventually deliver
    /// every picture in POC-ascending order. Mid-stream order depends
    /// on when the bumping process fires (queue-full threshold); the
    /// end-of-stream flush cleans up the tail.
    ///
    /// Trace (cap = 2, bump runs BEFORE each insertion when len == cap):
    ///   push (POC 0, tag 10): queue=[0]
    ///   push (POC 4, tag 11): queue=[0,4]
    ///   push (POC 2, tag 12): bump lowest POC 0 (tag 10) → ready;
    ///                         queue=[4,2]
    ///   push (POC 1, tag 13): bump lowest POC 2 (tag 12) → ready;
    ///                         queue=[4,1]
    ///   push (POC 3, tag 14): bump lowest POC 1 (tag 13) → ready;
    ///                         queue=[4,3]
    ///   flush: sorted ascending → [POC 3 (tag 14), POC 4 (tag 11)]
    ///
    /// Ready drain: 10, 12, 13. Flush: 14, 11. Total: 10, 12, 13, 14, 11.
    /// This is NOT strictly POC-ascending because the bumping process
    /// is conservative — it emits the lowest-POC entry *currently
    /// queued* when capacity is reached, not the lowest POC across
    /// the whole stream. That matches §C.4's real-decoder behaviour.
    /// For a genuinely ascending output the bitstream must give the
    /// decoder enough slack (i.e. a `max_num_reorder_frames` large
    /// enough to hold every picture that could reorder past the one
    /// being bumped).
    #[test]
    fn reorder_with_small_dpb_matches_conservative_bumping() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        dec.output_dpb = DpbOutput::<FrameLease>::new(2, 3);

        push_entry(&mut dec, 10, 0, 0); // IDR
        push_entry(&mut dec, 11, 4, 1); // P
        push_entry(&mut dec, 12, 2, 2); // B
        push_entry(&mut dec, 13, 1, 3); // B
        push_entry(&mut dec, 14, 3, 4); // B

        dec.flush().expect("flush");

        let mut tags = Vec::new();
        while let Ok(f) = dec.receive_frame() {
            tags.push(vf_tag(&f));
        }

        assert_eq!(tags, vec![10, 12, 13, 14, 11]);
    }

    /// §C.4 — with a reorder window that *is* large enough to hold
    /// every out-of-order picture, the flush at EOF produces the
    /// fully POC-ascending output order expected for display.
    #[test]
    fn reorder_with_sufficient_dpb_yields_strictly_ascending_poc() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        // Cap = 5 ≥ number of pictures → nothing bumps mid-stream,
        // every picture goes through the end-of-stream flush in POC
        // order.
        dec.output_dpb = DpbOutput::<FrameLease>::new(5, 5);

        // Same IPBBB decode order as above.
        push_entry(&mut dec, 10, 0, 0); // IDR
        push_entry(&mut dec, 11, 4, 1); // P
        push_entry(&mut dec, 12, 2, 2); // B
        push_entry(&mut dec, 13, 1, 3); // B
        push_entry(&mut dec, 14, 3, 4); // B

        dec.flush().expect("flush");

        let mut tags = Vec::new();
        let mut pocs = Vec::new();
        while let Ok(f) = dec.receive_frame() {
            tags.push(vf_tag(&f));
            // Reconstruct POC from our tagging scheme (tag -> poc):
            // 10->0, 11->4, 12->2, 13->1, 14->3.
            let poc = match vf_tag(&f) {
                10 => 0,
                11 => 4,
                12 => 2,
                13 => 1,
                14 => 3,
                _ => unreachable!(),
            };
            pocs.push(poc);
        }

        // Expected POC-ascending order: 0, 1, 2, 3, 4 → tags 10, 13, 12, 14, 11.
        assert_eq!(tags, vec![10, 13, 12, 14, 11]);
        assert_eq!(pocs, vec![0, 1, 2, 3, 4]);
    }

    /// §C.4 — a full flush drains the DPB in POC order at EOF.
    #[test]
    fn flush_at_eof_drains_in_poc_order() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        dec.output_dpb = DpbOutput::<FrameLease>::new(8, 8);

        // Decode order: [POC 3, POC 1, POC 2] — nothing bumped mid-stream
        // because we stay below capacity.
        push_entry(&mut dec, 0xA0, 3, 0);
        push_entry(&mut dec, 0xA1, 1, 1);
        push_entry(&mut dec, 0xA2, 2, 2);

        // Before flush(), nothing is over capacity → receive_frame gets
        // NeedMore.
        assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));

        dec.flush().expect("flush");

        let tags: Vec<u8> =
            std::iter::from_fn(|| dec.receive_frame().ok().map(|f| vf_tag(&f))).collect();
        // POC ascending: 1, 2, 3 → tags 0xA1, 0xA2, 0xA0.
        assert_eq!(tags, vec![0xA1, 0xA2, 0xA0]);
    }

    /// §C.4 — at EOF with an empty queue, `receive_frame` returns
    /// `Error::Eof` not `NeedMore`.
    #[test]
    fn eof_on_empty_queue_after_flush() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        dec.flush().expect("flush");
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    }

    /// §C.4 — `receive_frame` returns `NeedMore` mid-stream when the
    /// queue is under capacity.
    #[test]
    fn need_more_before_any_push() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    }

    /// §C.4 — the pre-IDR drain: when a new IDR lands, any pictures
    /// still pending from the previous coded video sequence must be
    /// delivered in POC order before the IDR itself.
    ///
    /// We simulate this by pushing a few entries, then doing the same
    /// flush-into-ready + reset dance `handle_slice` does on an IDR
    /// boundary, then pushing the new IDR. The POC counter restarts
    /// from 0 on IDR, but the old sequence's pictures still need to
    /// come out first.
    #[test]
    fn idr_drains_pending_pictures_in_poc_order() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        dec.output_dpb = DpbOutput::<FrameLease>::new(4, 4);

        // Sequence 1: POCs 0, 4, 2, 1 — four frames queued, none bumped.
        push_entry(&mut dec, 1, 0, 0);
        push_entry(&mut dec, 2, 4, 1);
        push_entry(&mut dec, 3, 2, 2);
        push_entry(&mut dec, 4, 1, 3);

        // Mimic `handle_slice`'s IDR branch: drain pending into `ready`
        // then reset the output DPB.
        for drained in dec.output_dpb.flush() {
            dec.ready.push_back(drained.picture);
        }
        dec.output_dpb.reset();

        // IDR at POC 0 kicks off sequence 2.
        push_entry(&mut dec, 100, 0, 0);

        // EOF drains the IDR itself too.
        dec.flush().expect("flush");

        let tags: Vec<u8> =
            std::iter::from_fn(|| dec.receive_frame().ok().map(|f| vf_tag(&f))).collect();
        // Sequence 1 in POC order (0, 1, 2, 4) → tags [1, 4, 3, 2]
        // followed by sequence 2's IDR (100).
        assert_eq!(tags, vec![1, 4, 3, 2, 100]);
    }

    /// §C.4 — `reset()` wipes both the output DPB and the "already
    /// bumped" queue.
    #[test]
    fn reset_clears_output_dpb_and_ready() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        dec.output_dpb = DpbOutput::<FrameLease>::new(2, 2);
        let old_pool = Arc::clone(&dec.picture_pool);
        // Fill + overflow so one entry lands in `ready`.
        push_entry(&mut dec, 1, 0, 0);
        push_entry(&mut dec, 2, 1, 1);
        push_entry(&mut dec, 3, 2, 2); // bumps lowest POC (0) → ready.
        assert_eq!(dec.ready.len(), 1);
        assert_eq!(dec.output_dpb.len(), 2);

        dec.reset().expect("reset");
        assert_eq!(dec.output_dpb.len(), 0);
        assert!(dec.ready.is_empty());
        assert!(
            !Arc::ptr_eq(&old_pool, &dec.picture_pool),
            "reset must detach new decode from pools retained by old leases"
        );
        assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    }

    /// VUI policy: when the encoder claims a `max_num_reorder_frames`
    /// at-or-above the level-derived cap we honour it (the encoder
    /// has correctly characterised its stream); when it's below
    /// the level cap we raise to the cap. solana-ad's High@L3.1 720p
    /// run is the textbook case for the floor — the VUI claims 2
    /// but the stream uses a 4-frame B-pyramid that needs 4. See
    /// `output_dpb_sizing` doc-comment.
    #[test]
    fn dpb_sizing_honours_vui_when_at_or_above_level_cap() {
        use crate::sps::Sps;
        use crate::vui::{BitstreamRestriction, VuiParameters};

        // 176x144 (PicSize=99 mb) at level 3.0 → MaxDpbMbs/PicSize =
        // 8100/99 = 81 → clamp to 16. VUI claims reorder=16, buffer=16
        // → both honoured exactly because they match the cap.
        let sps = Sps {
            profile_idc: 66,
            constraint_set_flags: 0,
            level_idc: 30,
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: false,
            seq_scaling_lists: None,
            log2_max_frame_num_minus4: 0,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 4,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs_minus1: 10,
            pic_height_in_map_units_minus1: 8,
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: true,
            frame_cropping: None,
            vui_parameters_present_flag: true,
            vui: Some(VuiParameters {
                bitstream_restriction: Some(BitstreamRestriction {
                    motion_vectors_over_pic_boundaries_flag: true,
                    max_bytes_per_pic_denom: 0,
                    max_bits_per_mb_denom: 0,
                    log2_max_mv_length_horizontal: 0,
                    log2_max_mv_length_vertical: 0,
                    max_num_reorder_frames: 16,
                    max_dec_frame_buffering: 16,
                }),
                ..Default::default()
            }),
        };
        assert_eq!(output_dpb_sizing(&sps), (16, 16));
    }

    /// Real-world encoder quirk: VUI claims reorder=2 but the level
    /// cap is 5 (level 3.1, 720p, PicSize=3600 mb → MaxDpbMbs/PicSize
    /// = 18000/3600 = 5). The fix raises the reorder window to the
    /// level cap so a 4-deep B-pyramid (which `max_num_reorder_frames=2`
    /// would prematurely bump out of order) decodes in display order.
    #[test]
    fn dpb_sizing_raises_undersized_vui_to_level_cap() {
        use crate::sps::Sps;
        use crate::vui::{BitstreamRestriction, VuiParameters};

        let sps = Sps {
            profile_idc: 100, // High
            constraint_set_flags: 0,
            level_idc: 31, // 3.1 → MaxDpbMbs 18000
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: false,
            seq_scaling_lists: None,
            log2_max_frame_num_minus4: 0,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 5,
            gaps_in_frame_num_value_allowed_flag: false,
            // 1280x720 → 80x45 mbs → PicSize = 3600.
            pic_width_in_mbs_minus1: 79,
            pic_height_in_map_units_minus1: 44,
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: true,
            frame_cropping: None,
            vui_parameters_present_flag: true,
            vui: Some(VuiParameters {
                bitstream_restriction: Some(BitstreamRestriction {
                    motion_vectors_over_pic_boundaries_flag: true,
                    max_bytes_per_pic_denom: 0,
                    max_bits_per_mb_denom: 0,
                    log2_max_mv_length_horizontal: 0,
                    log2_max_mv_length_vertical: 0,
                    max_num_reorder_frames: 2, // encoder's undersized claim
                    max_dec_frame_buffering: 5,
                }),
                ..Default::default()
            }),
        };
        // level_cap = 18000/3600 = 5; reorder raised from 2 → 5,
        // buffering = max(5, 5, 5) = 5.
        assert_eq!(output_dpb_sizing(&sps), (5, 5));
    }

    /// §C.4 regression — B-pyramid output ordering on a stream
    /// matching solana-ad's High@L3.1 720p shape.
    ///
    /// Decode order for a 4-deep B-pyramid following an IDR is
    /// `I0, P8, P4, B2, B6, P16, P12, B10, B14, ...` — POC values
    /// in parentheses; the encoder feeds the decoder anchor
    /// references first then fills the in-between B-frames. Before
    /// the fix, `output_dpb_sizing` honoured the encoder's
    /// `max_num_reorder_frames = 2` which forced §C.4 bumps before
    /// the in-between Bs arrived, scrambling the output. With the
    /// reorder window sized to the level-derived 5 (level 3.1 720p
    /// MaxDpbMbs/PicSize = 5), the bumping process emits frames in
    /// strict POC-ascending order.
    ///
    /// `tag` here is `display_index + 1` (1..=9) so `vf_tag` can
    /// recover monotonic display indices; the POC scheme uses the
    /// usual ×2 spacing.
    #[test]
    fn b_pyramid_emits_in_poc_order_at_level_31_720p() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        // Mirror what the production fix derives for level 3.1 720p.
        dec.output_dpb = DpbOutput::<FrameLease>::new(5, 5);

        // Decode order, with (tag = display_index + 1, POC).
        push_entry(&mut dec, 1, 0, 0); // I0   display 0
        push_entry(&mut dec, 5, 8, 1); // P8   display 4
        push_entry(&mut dec, 3, 4, 2); // P4   display 2 (ref-B / pyramid anchor)
        push_entry(&mut dec, 2, 2, 3); // B2   display 1
        push_entry(&mut dec, 4, 6, 3); // B6   display 3
        push_entry(&mut dec, 9, 16, 4); // P16  display 8
        push_entry(&mut dec, 7, 12, 5); // P12  display 6 (ref-B)
        push_entry(&mut dec, 6, 10, 5); // B10  display 5
        push_entry(&mut dec, 8, 14, 5); // B14  display 7

        dec.flush().expect("flush");

        let mut tags = Vec::new();
        while let Ok(f) = dec.receive_frame() {
            tags.push(vf_tag(&f));
        }

        // Must emerge in display order — i.e. tags monotonically 1..=9.
        assert_eq!(tags, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    /// Annex A Table A-1 fallback — when VUI is absent, use the
    /// level-derived MaxDpbMbs cap as `max_dec_frame_buffering`, and
    /// (per §A.3.1 item j) infer `max_num_reorder_frames` equal to
    /// that cap (maximum reorder window). Level 3.0 (level_idc == 30)
    /// has MaxDpbMbs = 8100; for a 176x144 (11x9 MB) picture that's
    /// Min(8100/99, 16) = 16 on both values.
    #[test]
    fn dpb_sizing_falls_back_to_level_default() {
        use crate::sps::Sps;

        let sps = Sps {
            profile_idc: 66,
            constraint_set_flags: 0,
            level_idc: 30, // Level 3.0 → MaxDpbMbs 8100.
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: false,
            seq_scaling_lists: None,
            log2_max_frame_num_minus4: 0,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 4,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs_minus1: 10, // width_in_mbs = 11 (176 px)
            pic_height_in_map_units_minus1: 8, // height_in_mbs = 9 (144 px)
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: true,
            frame_cropping: None,
            vui_parameters_present_flag: false,
            vui: None,
        };
        // PicSize = 11 * 9 = 99; 8100 / 99 = 81 → capped at 16
        // (max_dec_frame_buffering). Reorder inferred to match
        // buffering per §A.3.1 item j.
        assert_eq!(output_dpb_sizing(&sps), (16, 16));
    }

    /// Annex A Table A-1 — per-level MaxDpbMbs sanity. Levels that
    /// share the same MaxDpbMbs row in Table A-1 must yield the same
    /// value.
    #[test]
    fn max_dpb_mbs_per_level_table() {
        // For all level_idc values except 11, constraint_set3_flag is
        // ignored — pass `false` for the typical case.
        assert_eq!(max_dpb_mbs_for_level(10, false), 396);
        assert_eq!(max_dpb_mbs_for_level(12, false), 2_376);
        assert_eq!(max_dpb_mbs_for_level(21, false), 4_752);
        assert_eq!(max_dpb_mbs_for_level(30, false), 8_100);
        assert_eq!(max_dpb_mbs_for_level(31, false), 18_000);
        assert_eq!(max_dpb_mbs_for_level(40, false), 32_768);
        assert_eq!(max_dpb_mbs_for_level(42, false), 34_816);
        assert_eq!(max_dpb_mbs_for_level(50, false), 110_400);
        assert_eq!(max_dpb_mbs_for_level(51, false), 184_320);
        assert_eq!(max_dpb_mbs_for_level(60, false), 696_320);
        // Unknown level → conservative fallback.
        assert_eq!(max_dpb_mbs_for_level(200, false), 396);
    }

    /// §A.3.4.1 + Annex A Table A-1 — Level 1b vs Level 1.1.
    ///
    /// Both share `level_idc == 11`; `constraint_set3_flag` is the
    /// disambiguator. Level 1b's MaxDpbMbs (396) matches Level 1, NOT
    /// Level 1.1 (900). The shorthand `level_idc == 9` (Baseline /
    /// Constrained Baseline) also signals Level 1b directly.
    #[test]
    fn max_dpb_mbs_level_1b_versus_level_1_1() {
        // level_idc == 11 alone → Level 1.1 (the historical default
        // when constraint_set3_flag is unset).
        assert_eq!(max_dpb_mbs_for_level(11, false), 900);
        // level_idc == 11 with constraint_set3_flag → Level 1b.
        assert_eq!(max_dpb_mbs_for_level(11, true), 396);
        // level_idc == 9 → Level 1b shorthand (Baseline path).
        assert_eq!(max_dpb_mbs_for_level(9, false), 396);
        assert_eq!(max_dpb_mbs_for_level(9, true), 396);
    }

    /// §A.3.1 / §C.4 — sizing a Level 1b QCIF stream must use the
    /// Level 1b MaxDpbMbs (396), not the Level 1.1 value (900). For a
    /// 176x144 frame (PicSize = 11x9 = 99 MBs) that gives a
    /// level-derived reorder cap of 4, not 9.
    ///
    /// This is the textbook bug-fix case: a real Level 1b encoder
    /// signals `level_idc=11`, `constraint_set3_flag=1`, and our
    /// previous code over-buffered by treating the stream as Level 1.1.
    #[test]
    fn level_1b_qcif_stream_sized_at_level_1b_cap_not_level_1_1() {
        let mut sps = Sps {
            profile_idc: 66,
            constraint_set_flags: 0b0000_1000, // constraint_set3_flag = 1
            level_idc: 11,
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: false,
            seq_scaling_lists: None,
            log2_max_frame_num_minus4: 0,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 1,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs_minus1: 10,       // 11 MBs wide → 176 px
            pic_height_in_map_units_minus1: 8, // 9 map units high
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: true,
            frame_cropping: None,
            vui_parameters_present_flag: false,
            vui: None,
        };
        // Level 1b → reorder cap = min(396 / 99, 16) = 4.
        let (reorder, buffering) = output_dpb_sizing(&sps);
        assert_eq!(reorder, 4);
        assert_eq!(buffering, 4);

        // Same SPS, but clear constraint_set3_flag → Level 1.1 →
        // reorder cap = min(900 / 99, 16) = 9.
        sps.constraint_set_flags = 0;
        let (reorder, buffering) = output_dpb_sizing(&sps);
        assert_eq!(reorder, 9);
        assert_eq!(buffering, 9);
    }

    // -- §7.4.1.2.4 first-VCL-of-primary-coded-picture detection -------
    //
    // These tests seed an `in_progress` PictureInProgress by hand and
    // then probe `is_first_vcl_of_new_picture` with different trailing
    // slice headers. The assembly code itself is exercised end-to-end
    // by `tests/integration_multislice_assembly.rs`; these tests cover
    // the boundary-condition matrix without needing a real bitstream.

    use crate::slice_header::{RefPicListModification, SliceHeader as Hdr, SliceType as ST};

    /// Minimal SPS for the seed helpers — exact field values do not
    /// matter since the seeded in-progress picture is never actually
    /// reconstructed; these tests only exercise
    /// `is_first_vcl_of_new_picture` which consults the slice header.
    fn test_sps() -> crate::sps::Sps {
        crate::sps::Sps {
            profile_idc: 66,
            constraint_set_flags: 0,
            level_idc: 10,
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: false,
            seq_scaling_lists: None,
            log2_max_frame_num_minus4: 0,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 1,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs_minus1: 0,
            pic_height_in_map_units_minus1: 0,
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: false,
            frame_cropping: None,
            vui_parameters_present_flag: false,
            vui: None,
        }
    }

    fn test_pps() -> crate::pps::Pps {
        crate::pps::Pps {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            entropy_coding_mode_flag: false,
            bottom_field_pic_order_in_frame_present_flag: false,
            num_slice_groups_minus1: 0,
            slice_group_map: None,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            pic_init_qp_minus26: 0,
            pic_init_qs_minus26: 0,
            chroma_qp_index_offset: 0,
            deblocking_filter_control_present_flag: false,
            constrained_intra_pred_flag: false,
            redundant_pic_cnt_present_flag: false,
            extension: None,
        }
    }

    /// Build a minimal SliceHeader for boundary-detection testing. All
    /// fields default to "non-IDR P frame at frame_num=0, POC lsb=0" —
    /// tests tweak the specific fields they want to compare.
    fn hdr_base() -> Hdr {
        Hdr {
            first_mb_in_slice: 0,
            slice_type_raw: 0,
            slice_type: ST::P,
            all_slices_same_type: false,
            pic_parameter_set_id: 0,
            colour_plane_id: 0,
            frame_num: 0,
            field_pic_flag: false,
            bottom_field_flag: false,
            idr_pic_id: 0,
            pic_order_cnt_lsb: 0,
            delta_pic_order_cnt_bottom: 0,
            delta_pic_order_cnt: [0, 0],
            redundant_pic_cnt: 0,
            direct_spatial_mv_pred_flag: false,
            num_ref_idx_active_override_flag: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            ref_pic_list_modification: RefPicListModification::default(),
            pred_weight_table: None,
            dec_ref_pic_marking: None,
            cabac_init_idc: 0,
            slice_qp_delta: 0,
            sp_for_switch_flag: false,
            slice_qs_delta: 0,
            disable_deblocking_filter_idc: 0,
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            slice_group_change_cycle: 0,
        }
    }

    /// Seed a dummy PictureInProgress on a decoder so
    /// `is_first_vcl_of_new_picture` has something to compare against.
    fn seed_in_progress(dec: &mut H264CodecDecoder, nut: u8, nri: u8, header: Hdr) {
        let pic = Picture::new(16, 16, 1, 8, 8);
        let grid = MbGrid::new(1, 1);
        let sps = test_sps();
        let pps = test_pps();
        let prepared = dec
            .picture_frontend
            .prepare_parsed_picture(nut, nri, header.clone(), sps.clone(), pps.clone(), 1)
            .expect("prepare test picture");
        let poc = prepared.poc;
        let structure = prepared.structure;
        dec.in_progress = Some(PictureInProgress {
            prepared,
            pic,
            grid,
            first_nal_unit_type: nut,
            first_nal_ref_idc: nri,
            first_header: header,
            is_reference: nri != 0,
            is_idr: nut == 5,
            poc,
            structure,
            pts: None,
            time_base: TimeBase::new(1, 1),
            deblock_enabled: false,
            deblock_alpha_off: 0,
            deblock_beta_off: 0,
            mb_field_flags: Vec::new(),
            sps,
            pps,
            any_slice_succeeded: true,
        });
    }

    /// §7.4.1.2.4 — no picture in progress ⇒ any slice starts a new
    /// primary coded picture.
    #[test]
    fn first_vcl_when_no_picture_in_progress() {
        let dec = H264CodecDecoder::new(CodecId::new("h264"));
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &hdr_base()));
    }

    /// §7.4.1.2.4 — identical header + nal_unit_type + nal_ref_idc ⇒
    /// SAME primary coded picture (continuation slice).
    #[test]
    fn same_picture_when_all_conditions_match() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        // Same everything except first_mb_in_slice (which is NOT in
        // the §7.4.1.2.4 list of differing conditions).
        let mut h = hdr_base();
        h.first_mb_in_slice = 384;
        assert!(!dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — `frame_num` differs ⇒ new picture.
    #[test]
    fn different_frame_num_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        let mut h = hdr_base();
        h.frame_num = 1;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — `pic_parameter_set_id` differs ⇒ new picture.
    #[test]
    fn different_pps_id_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        let mut h = hdr_base();
        h.pic_parameter_set_id = 3;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — `field_pic_flag` differs ⇒ new picture.
    #[test]
    fn different_field_pic_flag_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        let mut h = hdr_base();
        h.field_pic_flag = true;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — the two fields of a complementary pair share
    /// `frame_num` + `field_pic_flag` but differ in `bottom_field_flag`,
    /// so the bottom field opens a new primary coded picture (forcing
    /// the top field to finalize first). With matching `pic_order_cnt_lsb`
    /// only the `bottom_field_flag` condition distinguishes them.
    #[test]
    fn different_bottom_field_flag_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let mut top = hdr_base();
        top.field_pic_flag = true;
        top.bottom_field_flag = false;
        seed_in_progress(&mut dec, 1, 2, top);
        let mut bottom = hdr_base();
        bottom.field_pic_flag = true;
        bottom.bottom_field_flag = true;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &bottom));
    }

    /// §7.4.1.2.4 — `bottom_field_flag` is ignored for frame pictures
    /// (`field_pic_flag == 0`): a stale `bottom_field_flag` difference on
    /// two frame slices must NOT be read as a picture boundary.
    #[test]
    fn bottom_field_flag_ignored_for_frame_pictures() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let mut a = hdr_base();
        a.field_pic_flag = false;
        a.bottom_field_flag = false;
        seed_in_progress(&mut dec, 1, 2, a);
        let mut b = hdr_base();
        b.field_pic_flag = false;
        b.bottom_field_flag = true; // ignored when field_pic_flag == 0
        assert!(!dec.is_first_vcl_of_new_picture(1, 2, &b));
    }

    /// §7.4.1.2.4 — nal_ref_idc zero-ness differs (prev ref, new
    /// non-ref) ⇒ new picture.
    #[test]
    fn different_nal_ref_idc_zero_ness_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        // Old nal_ref_idc = 2 (non-zero), new = 0.
        assert!(dec.is_first_vcl_of_new_picture(1, 0, &hdr_base()));
    }

    /// §7.4.1.2.4 — both nal_ref_idc non-zero but different value
    /// (e.g. 1 vs 2) is NOT a new picture — only the *zero-ness*
    /// matters.
    #[test]
    fn same_nal_ref_idc_nonzero_is_same_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        assert!(!dec.is_first_vcl_of_new_picture(1, 1, &hdr_base()));
    }

    /// §7.4.1.2.4 — `pic_order_cnt_lsb` differs ⇒ new picture.
    #[test]
    fn different_poc_lsb_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        let mut h = hdr_base();
        h.pic_order_cnt_lsb = 4;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — `delta_pic_order_cnt_bottom` differs ⇒ new picture.
    #[test]
    fn different_delta_poc_bottom_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        let mut h = hdr_base();
        h.delta_pic_order_cnt_bottom = 1;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — `delta_pic_order_cnt[0]` differs ⇒ new picture.
    #[test]
    fn different_delta_poc_0_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        let mut h = hdr_base();
        h.delta_pic_order_cnt[0] = 2;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — `delta_pic_order_cnt[1]` differs ⇒ new picture.
    #[test]
    fn different_delta_poc_1_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        let mut h = hdr_base();
        h.delta_pic_order_cnt[1] = 3;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — IdrPicFlag differs (one IDR, other not) ⇒ new.
    #[test]
    fn different_idr_flag_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        // IDR is nal_unit_type == 5. The new slice type 5 differs from
        // prev type 1.
        let mut h = hdr_base();
        h.slice_type_raw = 2;
        h.slice_type = ST::I;
        assert!(dec.is_first_vcl_of_new_picture(5, 2, &h));
    }

    /// §7.4.1.2.4 — both IDR but `idr_pic_id` differs ⇒ new.
    #[test]
    fn different_idr_pic_id_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let mut prev = hdr_base();
        prev.idr_pic_id = 0;
        prev.slice_type_raw = 2;
        prev.slice_type = ST::I;
        seed_in_progress(&mut dec, 5, 3, prev);
        let mut h = hdr_base();
        h.idr_pic_id = 1;
        h.slice_type_raw = 2;
        h.slice_type = ST::I;
        assert!(dec.is_first_vcl_of_new_picture(5, 3, &h));
    }

    /// §7.4.1.2.4 — both IDR with SAME idr_pic_id and all other fields
    /// match ⇒ SAME picture.
    #[test]
    fn same_idr_pic_id_is_same_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let mut prev = hdr_base();
        prev.idr_pic_id = 7;
        prev.slice_type_raw = 2;
        prev.slice_type = ST::I;
        seed_in_progress(&mut dec, 5, 3, prev.clone());
        // Continuation slice in a multi-slice IDR picture.
        let mut h = prev;
        h.first_mb_in_slice = 384;
        assert!(!dec.is_first_vcl_of_new_picture(5, 3, &h));
    }

    // ====== ISO/IEC 14496-15 §5.2.4.1.1 — avcC parser tests =========

    /// Minimal Baseline avcC: configurationVersion=1, profile_idc=66,
    /// profile_compat=0, level_idc=30, lengthSizeMinusOne=3 (=> 4-byte
    /// prefix), 0 SPS, 0 PPS. No High-profile extension. Smallest
    /// legal record that `consume_extradata` should accept.
    fn baseline_avcc_zero_sps_zero_pps(length_size_minus_one: u8) -> Vec<u8> {
        vec![
            0x01,                               // configurationVersion
            66,                                 // AVCProfileIndication = Baseline
            0x00,                               // profile_compatibility
            30,                                 // AVCLevelIndication = 3.0
            0xfc | (length_size_minus_one & 3), // reserved (6 bits = 111111) | lengthSizeMinusOne
            0xe0, // reserved (3 bits = 111) | numOfSequenceParameterSets = 0
            0x00, // numOfPictureParameterSets = 0
        ]
    }

    /// §5.2.4.1.1 — `consume_extradata` accepts the minimal 7-byte
    /// header and stores `length_size = 4` for `lengthSizeMinusOne = 3`.
    #[test]
    fn avcc_minimal_baseline_record_accepted() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let extra = baseline_avcc_zero_sps_zero_pps(3);
        dec.consume_extradata(&extra).expect("minimal avcC ok");
        assert_eq!(dec.length_size, Some(4));
        assert_eq!(dec.avcc_profile_idc(), Some(66));
        assert_eq!(dec.avcc_level_idc(), Some(30));
        // Baseline doesn't have the High-profile extension.
        assert_eq!(dec.avcc_chroma_format(), None);
        assert_eq!(dec.avcc_bit_depth_luma(), None);
        assert_eq!(dec.avcc_bit_depth_chroma(), None);
    }

    /// §5.2.4.1.1 — `lengthSizeMinusOne = 0` (1-byte prefix) is
    /// legal and yields `length_size = 1`.
    #[test]
    fn avcc_length_size_minus_one_0_means_1_byte_prefix() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        dec.consume_extradata(&baseline_avcc_zero_sps_zero_pps(0))
            .expect("lengthSize = 1 ok");
        assert_eq!(dec.length_size, Some(1));
    }

    /// §5.2.4.1.1 — `lengthSizeMinusOne = 1` (2-byte prefix) is
    /// legal and yields `length_size = 2`.
    #[test]
    fn avcc_length_size_minus_one_1_means_2_byte_prefix() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        dec.consume_extradata(&baseline_avcc_zero_sps_zero_pps(1))
            .expect("lengthSize = 2 ok");
        assert_eq!(dec.length_size, Some(2));
    }

    /// §5.2.4.1.1 — `lengthSizeMinusOne = 2` is forbidden by the spec
    /// (3-byte length prefix is not a legal AVCC framing). Verify the
    /// parser rejects up front rather than silently building an
    /// illegal splitter.
    #[test]
    fn avcc_length_size_minus_one_2_is_rejected() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let err = dec
            .consume_extradata(&baseline_avcc_zero_sps_zero_pps(2))
            .expect_err("lengthSize == 3 must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("lengthSizeMinusOne"),
            "error message should name the forbidden field: {msg}"
        );
        // Even though we rejected, `length_size` must NOT be populated
        // with the illegal value — the decoder stays in Annex B mode.
        assert_eq!(dec.length_size, None);
    }

    /// §5.2.4.1.1 — configurationVersion ≠ 1 is rejected.
    #[test]
    fn avcc_wrong_version_is_rejected() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let mut extra = baseline_avcc_zero_sps_zero_pps(3);
        extra[0] = 2;
        let err = dec
            .consume_extradata(&extra)
            .expect_err("version 2 must be rejected");
        assert!(format!("{err}").contains("configurationVersion"));
    }

    /// §5.2.4.1.1 — 6-byte header is short of the 7-byte minimum.
    #[test]
    fn avcc_short_header_is_rejected() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let err = dec
            .consume_extradata(&[0x01, 0x42, 0x00, 0x1e, 0xff, 0xe0])
            .expect_err("6-byte avcC must be rejected");
        assert!(format!("{err}").contains("shorter than avcC header"));
    }

    /// §5.2.4.1.1 — High-profile (profile_idc=100) extension: the
    /// chroma_format / bit_depth_*_minus8 / numOfSequenceParameterSetExt
    /// trailer is parsed and surfaced through the accessor methods.
    #[test]
    fn avcc_high_profile_extension_parsed() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let extra = vec![
            0x01, // configurationVersion
            100,  // AVCProfileIndication = High
            0x00, // profile_compatibility
            30,   // AVCLevelIndication
            0xff, // reserved | lengthSizeMinusOne = 3
            0xe0, // reserved | numOfSequenceParameterSets = 0
            0x00, // numOfPictureParameterSets = 0
            // §5.2.4.1.1 extension begins here:
            0xfc | 0x01, // reserved | chroma_format = 1 (4:2:0)
            0xf8 | 0x02, // reserved | bit_depth_luma_minus8 = 2 (10-bit)
            0xf8 | 0x02, // reserved | bit_depth_chroma_minus8 = 2 (10-bit)
            0x00,        // numOfSequenceParameterSetExt = 0
        ];
        dec.consume_extradata(&extra).expect("High avcC ext ok");
        assert_eq!(dec.avcc_profile_idc(), Some(100));
        assert_eq!(dec.avcc_chroma_format(), Some(1));
        assert_eq!(dec.avcc_bit_depth_luma(), Some(10));
        assert_eq!(dec.avcc_bit_depth_chroma(), Some(10));
    }

    /// §5.2.4.1.1 — the High-profile extension trailer is sometimes
    /// elided by real-world muxers even on profile_idc=100. Accept
    /// the truncated record (length_size still picked up) rather
    /// than hard-fail; the accessors remain `None`.
    #[test]
    fn avcc_high_profile_missing_extension_tolerated() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let mut extra = baseline_avcc_zero_sps_zero_pps(3);
        extra[1] = 100; // promote to High
        dec.consume_extradata(&extra)
            .expect("missing High ext tolerated");
        assert_eq!(dec.avcc_profile_idc(), Some(100));
        assert_eq!(dec.length_size, Some(4));
        // No High extension bytes → no surfaced extension fields.
        assert_eq!(dec.avcc_chroma_format(), None);
        assert_eq!(dec.avcc_bit_depth_luma(), None);
        assert_eq!(dec.avcc_bit_depth_chroma(), None);
    }

    /// §5.2.4.1.1 — profile_idc=244 (High 4:4:4 Predictive) extends
    /// the §5.2.4.1.1 enumeration. Accept the chroma_format / bit depth
    /// trailer for it too.
    #[test]
    fn avcc_high_444_profile_extension_parsed() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let extra = vec![
            0x01,
            244,
            0x00,
            30,          // header
            0xff,        // lengthSizeMinusOne = 3
            0xe0,        // 0 SPS
            0x00,        // 0 PPS
            0xfc | 0x03, // chroma_format = 3 (4:4:4)
            0xf8 | 0x04, // bit_depth_luma_minus8 = 4 (12-bit)
            0xf8 | 0x04, // bit_depth_chroma_minus8 = 4
            0x00,        // 0 SPS-Ext
        ];
        dec.consume_extradata(&extra).expect("4:4:4 avcC ok");
        assert_eq!(dec.avcc_chroma_format(), Some(3));
        assert_eq!(dec.avcc_bit_depth_luma(), Some(12));
        assert_eq!(dec.avcc_bit_depth_chroma(), Some(12));
    }

    /// §5.2.4.1.1 + §7.4.2.1.1 — `bit_depth_*_minus8` is capped at 6
    /// (i.e. 14-bit pixel samples) by the spec; reject values 7 even
    /// though the 3-bit field can carry it.
    #[test]
    fn avcc_bit_depth_minus8_overflow_is_rejected() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let extra = vec![
            0x01,
            100,
            0x00,
            30,
            0xff,
            0xe0,
            0x00,
            0xfc | 0x01, // chroma_format = 1
            0xf8 | 0x07, // bit_depth_luma_minus8 = 7  ← invalid
            0xf8,        // bit_depth_chroma_minus8 = 0
            0x00,        // 0 SPS-Ext
        ];
        let err = dec
            .consume_extradata(&extra)
            .expect_err("bit_depth = 7 must be rejected");
        assert!(format!("{err}").contains("bit_depth_luma_minus8"));
    }

    /// §5.2.4.1.1 — SPS body length exceeding the record bound is
    /// rejected (no panic, surfaced as `Error::Invalid`).
    #[test]
    fn avcc_truncated_sps_body_is_rejected() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let extra = vec![
            0x01, 66, 0x00, 30, 0xff, 0xe1, // 1 SPS announced
            0x00, 0x10, // SPS length = 16 …
            0xde, // … but only 1 byte present.
        ];
        let err = dec.consume_extradata(&extra).expect_err("truncated SPS");
        assert!(format!("{err}").contains("avcC truncated at SPS body"));
    }

    /// §5.2.4.1.1 — the PPS count byte is mandatory; a record that
    /// runs out of bytes after the SPS list is rejected.
    #[test]
    fn avcc_truncated_at_pps_count_is_rejected() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let extra = vec![0x01, 66, 0x00, 30, 0xff, 0xe0];
        let err = dec
            .consume_extradata(&extra)
            .expect_err("missing PPS count");
        assert!(format!("{err}").contains("avcC"));
    }

    // ---- §C.4.4 PAFF field pairing + interleave ---------------------

    /// Build a small half-height field [`Picture`] whose every luma
    /// sample equals `fill` and chroma samples equal `cfill`, with the
    /// given POC + frame_num stamped on.
    fn field_pic(w: u32, field_h: u32, fill: i32, cfill: i32, poc: i32, frame_num: u32) -> Picture {
        let mut p = Picture::new(w, field_h, 1, 8, 8);
        p.fill_luma(fill);
        p.fill_cb(cfill);
        p.fill_cr(cfill);
        p.pic_order_cnt = poc;
        p.frame_num = frame_num;
        p
    }

    #[test]
    fn interleave_fields_places_top_on_even_bottom_on_odd_rows() {
        // 16-wide, 2-MB-tall field → 32 field rows each, 64 frame rows.
        let w = 16u32;
        let field_h = 4u32; // small enough to enumerate
        let top = field_pic(w, field_h, 10, 110, 4, 7);
        let bottom = field_pic(w, field_h, 20, 120, 6, 7);
        let frame = interleave_fields(&top, &bottom);

        assert_eq!(frame.width_in_samples, w);
        assert_eq!(frame.height_in_samples, field_h * 2);
        // Even luma rows come from the top field (10), odd from the
        // bottom field (20).
        let wl = w as usize;
        for r in 0..(field_h * 2) as usize {
            let expect = if r % 2 == 0 { 10 } else { 20 };
            for c in 0..wl {
                assert_eq!(frame.luma_sample(r * wl + c), expect, "luma row {r}");
            }
        }
        // Chroma: 4:2:0 → half-width, half field height; interleave on
        // the chroma plane height too.
        let cw = frame.chroma_width() as usize;
        let cfh = top.chroma_height() as usize;
        for r in 0..(cfh * 2) {
            let expect = if r % 2 == 0 { 110 } else { 120 };
            for c in 0..cw {
                assert_eq!(frame.cb_sample(r * cw + c), expect, "cb row {r}");
                assert_eq!(frame.cr_sample(r * cw + c), expect, "cr row {r}");
            }
        }
        // §8.2.1 eq. 8-1 — frame POC = min(top, bottom) field POC.
        assert_eq!(frame.pic_order_cnt, 4);
        assert_eq!(frame.frame_num, 7);
    }

    #[test]
    fn complementary_field_pair_outputs_single_full_height_frame() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        // Top field then bottom field of the same frame_num.
        let top = field_pic(16, 4, 30, 128, 8, 3);
        dec.handle_field_output(top, false, 3, 8, Some(99))
            .expect("queue top field");
        // The first field alone produces no output (held pending).
        assert!(dec.ready.is_empty());
        assert!(dec.pending_field.is_some());

        let bottom = field_pic(16, 4, 40, 128, 10, 3);
        dec.handle_field_output(bottom, true, 3, 10, None)
            .expect("complete field pair");
        // Pair completed → pending cleared, one frame queued (possibly
        // still inside the output DPB until bumped). Force a drain.
        assert!(dec.pending_field.is_none());
        dec.eof = true;
        let lease = dec
            .receive_frame_lease()
            .expect("paired frame must drain as a lease");
        let frame = lease
            .as_arena_video()
            .expect("paired PAFF output must remain arena-backed");
        // Full-height (8 rows) 16-wide luma; pts inherited from the
        // first (top) field.
        assert_eq!(frame.header().presentation_timestamp, Some(99));
        assert_eq!(frame.plane_stride(0), Some(16));
        let luma = frame.plane(0).expect("luma plane");
        assert_eq!(luma.len(), 16 * 8);
        // Even rows = top field (30), odd rows = bottom (40).
        for r in 0..8 {
            let expect = if r % 2 == 0 { 30u8 } else { 40u8 };
            for c in 0..16 {
                assert_eq!(luma[r * 16 + c], expect);
            }
        }
    }

    #[test]
    fn non_complementary_second_field_flushes_orphan() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        // Two consecutive TOP fields (same parity) → not a pair. The
        // first must be emitted on its own, the second held pending.
        let top1 = field_pic(16, 4, 30, 128, 8, 3);
        dec.handle_field_output(top1, false, 3, 8, None)
            .expect("queue first top field");
        let top2 = field_pic(16, 4, 50, 128, 12, 4);
        dec.handle_field_output(top2, false, 4, 12, None)
            .expect("flush orphaned field");
        // First top field orphaned → one half-height frame queued; the
        // second top field is now pending.
        assert!(dec.pending_field.is_some());
        dec.eof = true;
        let lease = dec
            .receive_frame_lease()
            .expect("orphan field drains as a lease");
        let frame = lease
            .as_arena_video()
            .expect("unpaired PAFF output must remain arena-backed");
        // Half-height (4 rows) — an unpaired field is emitted as-is.
        let luma = frame.plane(0).expect("luma plane");
        assert_eq!(luma.len(), 16 * 4);
        assert_eq!(luma[0], 30);
    }

    fn gray_arena_4x2(fill: u8, pts: Option<i64>) -> FrameLease {
        let pool = ArenaPool::new(1, 8 + H264_PICTURE_ARENA_PADDING);
        let arena = pool.lease().expect("gray arena");
        let mut builder = VideoFrameBuilder::<u8>::new(arena, &[8], &[4]).expect("gray builder");
        builder.plane_mut(0).expect("gray plane").fill(fill);
        let frame = builder
            .freeze(FrameHeader::new(4, 2, PixelFormat::Gray8, pts))
            .expect("gray frame");
        FrameLease::from_arena_video(frame)
    }

    #[test]
    fn separate_colour_plane_merge_outputs_arena_video() {
        let y = gray_arena_4x2(10, Some(77));
        let cb = gray_arena_4x2(20, None);
        let cr = gray_arena_4x2(30, None);
        let mut pool = ArenaPool::new(H264_PICTURE_POOL_MAX_ARENAS, 1);
        let lease = merge_separate_colour_planes(&mut pool, &y, &cb, &cr, None, &HashSet::new())
            .expect("merge separate colour planes");
        let frame = lease
            .as_arena_video()
            .expect("SCP output must remain arena-backed");
        assert_eq!(frame.header().pixel_format, PixelFormat::Yuv444P);
        assert_eq!(frame.header().presentation_timestamp, Some(77));
        assert_eq!(frame.plane_count(), 3);
        assert!(frame.plane(0).unwrap().iter().all(|&value| value == 10));
        assert!(frame.plane(1).unwrap().iter().all(|&value| value == 20));
        assert!(frame.plane(2).unwrap().iter().all(|&value| value == 30));
    }

    #[test]
    fn picture_pool_pressure_blocks_until_retained_arena_returns() {
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let required =
            Picture::required_bytes(16, 16, 1, 8, 8).saturating_add(H264_PICTURE_ARENA_PADDING);
        let pool = ArenaPool::new(1, required);
        dec.picture_pool = Arc::clone(&pool);
        dec.set_cancellation_token(CancellationToken::new());

        // Model a downstream consumer retaining the only pooled picture.
        let mut first = Picture::new_in(&pool, 16, 16, 1, 8, 8).expect("first pooled picture");
        let retained = FrameLease::from_arena_video(first.freeze(None).expect("freeze first"));
        drop(first);

        let (tx, rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let result = dec.allocate_picture(16, 16, 1, 8, 8).is_ok();
            tx.send(result).expect("report allocation result");
        });

        assert!(
            rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "decoder allocation must wait while every arena is retained"
        );

        drop(retained);
        assert!(
            rx.recv_timeout(Duration::from_secs(1))
                .expect("allocation should resume after arena release"),
            "decoder allocation should succeed after retained arena returns"
        );
        worker.join().expect("allocation worker");
    }

    #[test]
    fn picture_pool_wait_returns_cancelled_when_pipeline_cancels() {
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let required =
            Picture::required_bytes(16, 16, 1, 8, 8).saturating_add(H264_PICTURE_ARENA_PADDING);
        let pool = ArenaPool::new(1, required);
        dec.picture_pool = Arc::clone(&pool);
        let cancellation = CancellationToken::new();
        dec.set_cancellation_token(cancellation.clone());

        let mut first = Picture::new_in(&pool, 16, 16, 1, 8, 8).expect("first pooled picture");
        let retained = FrameLease::from_arena_video(first.freeze(None).expect("freeze first"));
        drop(first);

        let (tx, rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let result = dec.allocate_picture(16, 16, 1, 8, 8).map(|_| ());
            tx.send(result).expect("report allocation result");
        });

        assert!(
            rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "decoder must be waiting before cancellation"
        );
        cancellation.cancel();
        let result = rx
            .recv_timeout(Duration::from_secs(1))
            .expect("cancellation must wake decoder promptly");
        assert!(matches!(result, Err(Error::Cancelled(_))));

        // Cancellation does not steal/recycle a still-retained arena.
        assert_eq!(pool.checked_out_count(), 1);
        drop(retained);
        worker.join().expect("allocation worker");
    }

    #[test]
    fn picture_pool_detects_decoder_self_deadlock() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let required =
            Picture::required_bytes(16, 16, 1, 8, 8).saturating_add(H264_PICTURE_ARENA_PADDING);
        let pool = ArenaPool::new(1, required);
        dec.picture_pool = Arc::clone(&pool);
        dec.set_cancellation_token(CancellationToken::new());

        let mut first = dec
            .allocate_picture(16, 16, 1, 8, 8)
            .expect("first pooled picture");
        let retained = first.freeze(None).expect("freeze first");
        drop(first);
        dec.ready.push_back(FrameLease::from_arena_video(retained));

        let err = dec
            .allocate_picture(16, 16, 1, 8, 8)
            .expect_err("decoder-owned sole slot must not be waited on");
        assert!(
            matches!(&err, Error::ResourceExhausted(message) if message.contains("self-deadlock")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn assembly_pool_detects_decoder_self_deadlock() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let pool = ArenaPool::new(1, 64);
        dec.assembly_pool = Arc::clone(&pool);
        dec.set_cancellation_token(CancellationToken::new());

        let arena = pool.lease().expect("assembly arena");
        let mut builder =
            VideoFrameBuilder::<u8>::new(arena, &[8], &[4]).expect("assembly builder");
        builder.plane_mut(0).expect("plane").fill(7);
        let frame = builder
            .freeze(FrameHeader::new(4, 2, PixelFormat::Gray8, None))
            .expect("assembly frame");
        dec.ready.push_back(FrameLease::from_arena_video(frame));

        let err = match dec.lease_with_backpressure(&dec.assembly_pool, "test assembly") {
            Ok(_) => panic!("decoder-owned assembly slot must not be waited on"),
            Err(error) => error,
        };
        assert!(
            matches!(&err, Error::ResourceExhausted(message) if message.contains("self-deadlock")),
            "unexpected error: {err}"
        );
    }

    /// Round 430 (2026-07-25 scheduled-fuzz OOM triage) — §8.2.5.2
    /// frame_num gap fill must stay memory-bounded. A hostile stream
    /// can declare `gaps_in_frame_num_value_allowed_flag = 1` with
    /// MaxFrameNum = 2^16 and jump `frame_num` by tens of thousands;
    /// the gap loop used to allocate a full placeholder picture per
    /// missing frame_num (and the store never released ANY picture),
    /// which is an unbounded allocation driven by a few input bytes.
    /// Post-fix: only the gap entries that survive the §8.2.5.3
    /// sliding window carry sample buffers, and the store holds
    /// exactly the DPB's pictures.
    #[test]
    fn frame_num_gap_fill_is_memory_bounded() {
        use crate::slice_header::DecRefPicMarking;
        use crate::sps::Sps;

        let sps = Sps {
            profile_idc: 66,
            constraint_set_flags: 0,
            level_idc: 30,
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: false,
            seq_scaling_lists: None,
            log2_max_frame_num_minus4: 12,
            pic_order_cnt_type: 2,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 3,
            gaps_in_frame_num_value_allowed_flag: true,
            pic_width_in_mbs_minus1: 3,
            pic_height_in_map_units_minus1: 3,
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: true,
            frame_cropping: None,
            vui_parameters_present_flag: false,
            vui: None,
        };
        let pps = test_pps();
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));

        // Seed PrevRefFrameNum with an IDR reference at frame_num 0.
        let mut seed_header = hdr_base();
        seed_header.slice_type_raw = 2;
        seed_header.slice_type = ST::I;
        seed_header.dec_ref_pic_marking = Some(DecRefPicMarking {
            no_output_of_prior_pics_flag: false,
            long_term_reference_flag: false,
            adaptive_marking: None,
        });
        let seed = dec
            .picture_frontend
            .prepare_parsed_picture(5, 3, seed_header, sps.clone(), pps.clone(), 1)
            .expect("prepare seed IDR");
        let seed_commit = dec.picture_frontend.commit(seed);
        let seed_entry = seed_commit.current_dpb_entry.expect("seed reference");
        dec.ref_store
            .insert(seed_entry.dpb_key, gray_picture(64, 64, 1, 8, 8));

        let mut far_header = hdr_base();
        far_header.frame_num = 40_000;
        let far = dec
            .picture_frontend
            .prepare_parsed_picture(1, 2, far_header, sps.clone(), pps.clone(), 1)
            .expect("large gap preparation");

        // Tens of thousands of logical gap steps produce only the surviving
        // sliding-window metadata, not one allocation per missing frame.
        assert_eq!(far.references.len(), 3);
        assert_eq!(far.synthetic_references.len(), 3);
        let frame_nums: Vec<u32> = far.references.iter().map(|e| e.frame_num).collect();
        assert_eq!(frame_nums, vec![39_997, 39_998, 39_999]);

        // The software backend materialises samples only for those surviving
        // synthetic references. Transactionality retains the old seed until
        // reconstruction succeeds, so the transient bound is DPB + 1 here.
        let gray = gray_picture(64, 64, 1, 8, 8);
        for e in &far.synthetic_references {
            dec.ref_store.insert(e.dpb_key, gray.deep_copy());
        }
        assert!(dec.ref_picture_count() <= 4);
        for e in &far.synthetic_references {
            assert!(dec.ref_store.get_by_key(e.dpb_key).unwrap().non_existing);
        }

        let far_commit = dec.picture_frontend.commit(far);
        let current = far_commit.current_dpb_entry.expect("far reference");
        dec.ref_store.insert(current.dpb_key, gray.deep_copy());
        let live: Vec<u32> = dec
            .picture_frontend
            .references()
            .iter()
            .map(|e| e.dpb_key)
            .collect();
        dec.ref_store.retain_keys(&live);
        assert_eq!(dec.picture_frontend.references().len(), 3);
        assert_eq!(dec.ref_picture_count(), 3);

        // A second gap proves the committed shared state advanced to frame
        // 40000 instead of restarting from the original seed.
        let mut second_header = hdr_base();
        second_header.frame_num = 40_010;
        let second = dec
            .picture_frontend
            .prepare_parsed_picture(1, 2, second_header, sps, pps, 1)
            .expect("second gap preparation");
        assert_eq!(second.synthetic_references.len(), 3);
        let second_nums: Vec<u32> = second.references.iter().map(|e| e.frame_num).collect();
        assert_eq!(second_nums, vec![40_007, 40_008, 40_009]);
    }

    #[test]
    fn video_signal_maps_full_range_bt709_metadata() {
        use crate::vui::{ColourDescription, VideoSignalType};

        let info = video_color_from_signal(&VideoSignalType {
            video_format: 5,
            video_full_range_flag: true,
            colour_description: Some(ColourDescription {
                colour_primaries: 1,
                transfer_characteristics: 1,
                matrix_coefficients: 1,
            }),
        });
        assert_eq!(info.range, Some(VideoColorRange::Full));
        assert_eq!(info.matrix, Some(VideoMatrixCoefficients::Bt709));
        assert_eq!(info.colour_primaries, Some(1));
        assert_eq!(info.transfer_characteristics, Some(1));
    }

    #[test]
    fn video_signal_maps_limited_bt601_metadata() {
        use crate::vui::{ColourDescription, VideoSignalType};

        let info = video_color_from_signal(&VideoSignalType {
            video_format: 5,
            video_full_range_flag: false,
            colour_description: Some(ColourDescription {
                colour_primaries: 6,
                transfer_characteristics: 6,
                matrix_coefficients: 6,
            }),
        });
        assert_eq!(info.range, Some(VideoColorRange::Limited));
        assert_eq!(info.matrix, Some(VideoMatrixCoefficients::Smpte170M));
        assert_eq!(info.colour_primaries, Some(6));
        assert_eq!(info.transfer_characteristics, Some(6));
    }
}
