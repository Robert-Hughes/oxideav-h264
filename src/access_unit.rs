//! Optional streaming Annex-B access-unit reconstruction.
//!
//! Container packet boundaries are not codec access-unit boundaries. In
//! particular MPEG-TS demuxers naturally emit PES payloads, and a PES may
//! begin with continuation bytes from the preceding H.264 picture before the
//! next access-unit delimiter (AUD) appears. Decoders which already own a
//! streaming parser (for example NVIDIA `cuvidParser`) should consume the raw
//! packets directly and do not need this helper.
//!
//! [`AnnexBAccessUnitAssembler`] is deliberately opt-in. It preserves the
//! timing metadata of the packet in which an AUD begins, appends bytes before
//! that AUD to the preceding access unit, and can split multiple AUD-delimited
//! units from one input packet without inventing timestamps for later units.

use oxideav_core::{Error, Packet, Result};

/// Stateful, opt-in assembler for AUD-delimited Annex-B H.264 streams.
#[derive(Debug, Default)]
pub struct AnnexBAccessUnitAssembler {
    pending: Option<Packet>,
}

impl AnnexBAccessUnitAssembler {
    /// Feed one arbitrary container packet/PES payload.
    ///
    /// Returns every access unit made complete by boundaries in `packet`.
    /// The final access unit stays pending until a later boundary or [`flush`](Self::flush).
    ///
    /// H.264 does not require AUD NALs. If no AUD has ever anchored the
    /// stream, an Annex-B-aligned packet is passed through unchanged to
    /// preserve the useful packet-aligned fallback used by MP4/elementary
    /// sources. Once an AUD has established streaming state, packets without
    /// a new AUD extend the pending access unit.
    pub fn push(&mut self, packet: &Packet) -> Result<Vec<Packet>> {
        if packet.data.is_empty() {
            return Ok(Vec::new());
        }

        let auds = aud_offsets(&packet.data);
        if auds.is_empty() {
            if let Some(pending) = self.pending.as_mut() {
                pending.data.extend_from_slice(&packet.data);
                return Ok(Vec::new());
            }
            if contains_annex_b_start_code(&packet.data) {
                // Preserve the parser's long-standing ability to resynchronise
                // past leading garbage/prefix bytes before the first NAL.
                return Ok(vec![packet.clone()]);
            }
            return Err(Error::unsupported(
                "H.264 Annex-B assembler received unanchored continuation bytes",
            ));
        }

        let mut completed = Vec::new();
        let first_aud = auds[0];

        if let Some(mut previous) = self.pending.take() {
            // The bytes before the first AUD belong to the previous AU. The
            // previous AU keeps the timing from the packet in which it began.
            previous.data.extend_from_slice(&packet.data[..first_aud]);
            completed.push(previous);
        }
        // With no pending AU, prefix bytes are an unanchored continuation
        // (common after seek/open in the middle of a PES) and are discarded so
        // we resynchronise at the first explicit AUD.

        for (idx, &start) in auds.iter().enumerate() {
            let end = auds.get(idx + 1).copied().unwrap_or(packet.data.len());
            let mut au = packet.clone();
            au.data = packet.data[start..end].to_vec();

            if idx > 0 {
                // ISO/IEC 13818-1 associates a PES timestamp with the first
                // access unit beginning in that PES. If one PES happens to
                // contain additional complete AUs, do not manufacture timing.
                au.pts = None;
                au.dts = None;
                au.duration = None;
                au.flags.keyframe = false;
            }

            if idx + 1 < auds.len() {
                completed.push(au);
            } else {
                self.pending = Some(au);
            }
        }

        Ok(completed)
    }

    /// Emit the final pending access unit, if any.
    pub fn flush(&mut self) -> Option<Packet> {
        self.pending.take()
    }

    /// Drop all buffered byte-stream state (e.g. after a seek).
    pub fn reset(&mut self) {
        self.pending = None;
    }

    /// Whether an AUD-anchored access unit is currently buffered.
    #[must_use]
    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }
}

/// True when `data` begins with a 3- or 4-byte Annex-B start code.
#[must_use]
pub fn starts_with_annex_b(data: &[u8]) -> bool {
    data.starts_with(&[0, 0, 1]) || data.starts_with(&[0, 0, 0, 1])
}

fn contains_annex_b_start_code(data: &[u8]) -> bool {
    data.windows(3).any(|w| w == [0, 0, 1])
}

/// Byte offsets of Annex-B Access Unit Delimiter NAL start-code prefixes.
/// Emulation-prevention guarantees `00 00 01` cannot occur unescaped inside
/// a NAL payload, so scanning raw Annex-B bytes is safe.
fn aud_offsets(data: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 3 < data.len() {
        let (prefix_len, nal_at) = if i + 4 <= data.len() && data[i..i + 4] == [0, 0, 0, 1] {
            (4usize, i + 4)
        } else if data[i..i + 3] == [0, 0, 1] {
            (3usize, i + 3)
        } else {
            i += 1;
            continue;
        };
        if nal_at < data.len() && (data[nal_at] & 0x1f) == 9 {
            out.push(i);
        }
        i += prefix_len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::TimeBase;

    fn packet(pts: i64, data: &[u8]) -> Packet {
        Packet::new(0, TimeBase::new(1, 90_000), data.to_vec()).with_pts(pts)
    }

    const AUD: &[u8] = &[0, 0, 0, 1, 0x09, 0xf0];

    #[test]
    fn split_pes_continuation_completes_previous_au_and_preserves_its_pts() {
        let mut a = AnnexBAccessUnitAssembler::default();
        let mut first = AUD.to_vec();
        first.extend_from_slice(&[0, 0, 1, 0x65, 0xaa, 0xbb]);
        assert!(a.push(&packet(100, &first)).unwrap().is_empty());

        let mut second = vec![0xcc, 0xdd];
        second.extend_from_slice(AUD);
        second.extend_from_slice(&[0, 0, 1, 0x41, 0x11]);
        let out = a.push(&packet(200, &second)).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].pts, Some(100));
        assert!(out[0].data.ends_with(&[0xaa, 0xbb, 0xcc, 0xdd]));
        assert_eq!(a.flush().unwrap().pts, Some(200));
    }

    #[test]
    fn aligned_aud_packet_completes_previous_without_stealing_next_pts() {
        let mut a = AnnexBAccessUnitAssembler::default();
        let mut one = AUD.to_vec();
        one.extend_from_slice(&[0, 0, 1, 0x65, 1]);
        let mut two = AUD.to_vec();
        two.extend_from_slice(&[0, 0, 1, 0x41, 2]);
        assert!(a.push(&packet(10, &one)).unwrap().is_empty());
        let out = a.push(&packet(20, &two)).unwrap();
        assert_eq!(out[0].pts, Some(10));
        assert_eq!(a.flush().unwrap().pts, Some(20));
    }

    #[test]
    fn multiple_auds_in_one_packet_are_split_without_inventing_later_pts() {
        let mut data = AUD.to_vec();
        data.extend_from_slice(&[0, 0, 1, 0x65, 1]);
        data.extend_from_slice(AUD);
        data.extend_from_slice(&[0, 0, 1, 0x41, 2]);
        data.extend_from_slice(AUD);
        data.extend_from_slice(&[0, 0, 1, 0x41, 3]);

        let mut a = AnnexBAccessUnitAssembler::default();
        let out = a.push(&packet(77, &data)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].pts, Some(77));
        assert_eq!(out[1].pts, None);
        let last = a.flush().unwrap();
        assert_eq!(last.pts, None);
    }

    #[test]
    fn packet_aligned_annex_b_without_aud_remains_passthrough() {
        let data = [0, 0, 1, 0x65, 0xaa];
        let mut a = AnnexBAccessUnitAssembler::default();
        let out = a.push(&packet(9, &data)).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].data, data);
        assert!(!a.has_pending());
    }

    #[test]
    fn leading_junk_before_annex_b_start_code_remains_parser_resynchronisable() {
        let data = [0xaa, 0xbb, 0xcc, 0, 0, 1, 0x65, 0x11];
        let mut a = AnnexBAccessUnitAssembler::default();
        let out = a.push(&packet(12, &data)).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].data, data);
        assert_eq!(out[0].pts, Some(12));
    }

    #[test]
    fn unanchored_continuation_is_rejected() {
        let mut a = AnnexBAccessUnitAssembler::default();
        assert!(a.push(&packet(1, &[0xaa, 0xbb])).is_err());
    }
}
