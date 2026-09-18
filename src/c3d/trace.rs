// SPDX-License-Identifier: GPL-3.0-or-later

//! The C3D conformance/capture trace container: a file of tagged,
//! length-framed sections recording one command-stream submission session
//! well enough that a native runner can replay it -- with no Amiga, no
//! guest library and no ROM -- and compare the pixels it produces against
//! golden images. `docs/internals/c3d.md`'s "Conformance" section is
//! the contract this module implements; that section describes the
//! container only loosely ("finalised with the runner in protocol 1.0"),
//! so the exact layout below is this module's finalisation of it, chosen
//! conservatively. Fold it back into the spec on review.
//!
//! [`TraceWriter`] frames sections onto any [`Write`](std::io::Write) as
//! they are captured; [`TraceReader`] iterates them back out of a byte
//! slice a section at a time, borrowing each payload rather than copying
//! it, so replaying a multi-megabyte trace of texture uploads costs no
//! more memory than the file itself already occupies. Every malformed
//! input -- a truncated section, a length claiming more bytes than the
//! file has, a stray tag before the header, a header that is not first --
//! is reported as a specific [`TraceError`] variant; nothing in this
//! module panics on untrusted input.
//!
//! ## Container layout
//!
//! A trace is a sequence of sections back to back, no padding, no
//! alignment, and no trailer: the reader stops cleanly when the byte
//! slice runs out. Each section is:
//!
//! ```text
//! tag[4] length[4] payload[length]
//! ```
//!
//! `tag` is four bytes (not necessarily ASCII -- an unrecognised tag is
//! skipped, not rejected, for forward compatibility); `length` is the
//! payload's byte count as a **big-endian `u32`**, matching every other
//! width on the C3D wire. A section whose `length` would run past the end
//! of the file is [`TraceError::TruncatedPayload`]; there is no other size
//! limit; on a 32-bit host, a payload longer than that platform's `usize`
//! is [`TraceError::PayloadTooLarge`] rather than an arithmetic overflow.
//!
//! | Tag | Fixed prefix (all fields `u32` big-endian) | Trailing bytes | Meaning |
//! |---|---|---|---|
//! | `C3DT` | `container_version, protocol_version, capability_mask` | none (fixed 12 bytes) | Trace header. Must be the **first** section in the file and must appear **exactly once**. `protocol_version` is the C3D `VERSION` register value the trace was captured against; `capability_mask` is the `CAPS0` bits the trace exercises -- a runner refuses to replay against a device that lacks any of them. |
//! | `APER` | `offset` | aperture image bytes | An initial image of part of the data aperture, to be written at `offset` before replay begins. May appear zero or more times. |
//! | `RING` | `context, ring_tail` | ring bytes | One submission: the bytes the guest appended to `context`'s ring, and the `RING_TAIL` value it then wrote (the doorbell). A runner writes the bytes at the ring's current tail and then sets `RING_TAIL` to `ring_tail`, in that order, exactly reproducing the two-step guest sequence. May appear any number of times; replayed in file order. |
//! | `BLOB` | `space, address` | referenced bytes | The bytes a ref (`docs/internals/c3d.md`, "References") named at capture time, to be materialised at guest (or aperture) `address` before the `RING` section that references them is replayed. May appear any number of times. |
//! | `GOLD` | `context, fence_id, surface_id` | PNG bytes | A golden image: the expected contents of `surface_id` once `context`'s `FENCE_COMPLETED` reaches `fence_id`. The PNG bytes are opaque to this module -- carried, never decoded. May appear any number of times. |
//!
//! Any other tag is read as [`Section::Unknown`], carrying its raw
//! payload, and the reader simply continues past it: a future minor
//! revision of the trace format can add a section kind without breaking
//! an older runner, the same forward-compatibility rule the protocol
//! itself uses for unknown opcodes.
//!
//! ### Decisions this module finalises
//!
//! The spec sketches each section as a tuple of named fields without
//! saying which are fixed-width, their order on the wire, or how a reader
//! tells a short payload from a truncated file. This module fixes:
//!
//! - **Every fixed field is a big-endian `u32`.** The C3D wire format is
//!   32-bit big-endian throughout (`docs/internals/c3d.md`'s "Conventions");
//!   the trace container follows the same rule rather than inventing a
//!   second endianness or width for its own bookkeeping fields. `length`
//!   is also a big-endian `u32` (not a `u64`) for the same reason, capping
//!   a single section at 4 GiB, which comfortably covers a golden PNG or a
//!   captured texture upload; nothing in the trace format needs a section
//!   larger than that, and a format that did would need block framing like
//!   `.clstate`'s streamed chunks, not a bigger length field.
//! - **`RING`'s fixed prefix is `context, ring_tail`, the variable ring
//!   bytes last.** The spec's prose lists the submission as "context, the
//!   bytes appended to the ring, the `RING_TAIL` written" -- bytes before
//!   the tail -- but every other section here is a fixed-size prefix
//!   followed by one run of variable-length bytes, which is what lets a
//!   reader locate every fixed field with fixed offsets and slice the
//!   trailing bytes without scanning. `RING` follows the same shape; only
//!   the prose order differs from the wire order, not the information
//!   carried.
//! - **`C3DT` must be first and unique.** A runner needs the protocol
//!   version and capability mask before it can make sense of anything
//!   else (in particular, before it can decide whether it can even attempt
//!   the trace), and a second header would leave later readers to guess
//!   which one governs. A file without a leading `C3DT` is refused wholesale
//!   ([`TraceError::MissingHeader`]) rather than partially accepted.
//! - **Every other section may repeat, in any order, any number of times**
//!   (including zero). Semantic ordering -- a `BLOB` before the `RING`
//!   that references it, a `GOLD` after the `RING`/`FENCE` it checks -- is
//!   a replay-correctness concern for the runner, not a framing rule this
//!   container enforces; a capture tool that emits sections in capture
//!   order satisfies it naturally.
//! - **No padding or alignment between sections.** Every field inside a
//!   payload is already a fixed-width `u32` or an explicitly-lengthed byte
//!   run, so nothing in this container needs alignment for correctness,
//!   and skipping alignment keeps a trace of many small `RING` sections
//!   (the common case -- one small section per `glBegin`/`glEnd`) from
//!   bloating with padding.
//! - **Unknown tags are skipped, not rejected**, exactly as the protocol
//!   itself treats an unknown opcode
//!   (Opcode map (`docs/internals/c3d.md`)) --
//!   forward compatibility matters in both places for the same reason.
//! - **What makes a file malformed**, each a distinct [`TraceError`]
//!   variant: the file is empty; the first section is not `C3DT`; `C3DT`
//!   appears more than once; a section header (the 8-byte tag+length) is
//!   cut short at end of file; a section's payload runs past the end of
//!   the file; `C3DT`'s payload is not exactly 12 bytes; a `RING`/`APER`/
//!   `BLOB`/`GOLD` payload is shorter than its fixed prefix; `C3DT`'s
//!   `container_version` is not one this build reads; a payload length
//!   that does not fit the host's `usize` (32-bit hosts only, since
//!   `length` is a `u32`). There is no checksum and no "file too large"
//!   limit beyond what the fields above already imply -- a trace is a
//!   captured artefact read from local disk by a developer running the
//!   conformance suite, not an input this process receives from an
//!   untrusted network peer, so the bar is "never panics or over-allocates
//!   on a corrupt file", not "resists an adversarial one".

use std::error::Error;
use std::fmt;
use std::io::{self, Write};

/// A section's four-byte tag. Not necessarily printable ASCII: an
/// implementation that adds a new section kind chooses whatever tag it
/// likes, and an older reader must still skip it cleanly.
pub type Tag = [u8; 4];

/// Trace header, `C3DT`. Must be first and unique.
pub const TAG_HEADER: Tag = *b"C3DT";
/// Initial aperture image, `APER`.
pub const TAG_APERTURE: Tag = *b"APER";
/// One ring submission, `RING`.
pub const TAG_RING: Tag = *b"RING";
/// Guest memory a ref named at capture time, `BLOB`.
pub const TAG_BLOB: Tag = *b"BLOB";
/// A golden image at a fence, `GOLD`.
pub const TAG_GOLD: Tag = *b"GOLD";

/// The container version this build writes and reads. See the module doc
/// comment's layout table; a version this build does not recognise is
/// [`TraceError::UnsupportedContainerVersion`], not silently guessed at.
pub const CONTAINER_VERSION: u32 = 1;

/// Bytes in a section's frame (`tag[4] length[4]`) ahead of its payload.
const SECTION_HEADER_LEN: usize = 8;
/// Bytes in a decoded `C3DT` payload: three `u32` fields, no trailing data.
const HEADER_PAYLOAD_LEN: usize = 12;

/// Everything that makes a trace file malformed. Every variant is
/// reachable from untrusted input alone; none of them is raised by a
/// panic or an unwrap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceError {
    /// The file is zero bytes: there is no header at all.
    Empty,
    /// The first section's tag was not `C3DT`.
    MissingHeader { tag: Tag },
    /// A second `C3DT` section appeared at this byte offset.
    DuplicateHeader { at: usize },
    /// Fewer than [`SECTION_HEADER_LEN`] bytes remained at this offset for
    /// a section's tag and length.
    TruncatedSectionHeader { at: usize, have: usize },
    /// A section's declared `length` reaches past the end of the file.
    TruncatedPayload {
        tag: Tag,
        at: usize,
        need: u32,
        have: usize,
    },
    /// A section's declared `length` does not fit in this host's `usize`
    /// (32-bit hosts only).
    PayloadTooLarge { tag: Tag, at: usize, need: u32 },
    /// `C3DT`'s payload was not exactly [`HEADER_PAYLOAD_LEN`] bytes.
    BadHeaderLength { actual: u32 },
    /// A `RING`/`APER`/`BLOB`/`GOLD` payload was shorter than the fixed
    /// prefix its tag requires.
    ShortPayload {
        tag: Tag,
        at: usize,
        need: usize,
        have: u32,
    },
    /// `C3DT`'s `container_version` is not one this build reads.
    UnsupportedContainerVersion(u32),
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "empty trace file (no C3DT header)"),
            Self::MissingHeader { tag } => write!(
                f,
                "trace does not start with a C3DT header (found {})",
                tag_name(*tag)
            ),
            Self::DuplicateHeader { at } => {
                write!(f, "duplicate C3DT header at offset {at}")
            }
            Self::TruncatedSectionHeader { at, have } => write!(
                f,
                "truncated section header at offset {at}: {have} byte(s) remain, need {SECTION_HEADER_LEN}"
            ),
            Self::TruncatedPayload {
                tag,
                at,
                need,
                have,
            } => write!(
                f,
                "{} section at offset {at} claims {need} payload byte(s) but only {have} remain",
                tag_name(*tag)
            ),
            Self::PayloadTooLarge { tag, at, need } => write!(
                f,
                "{} section at offset {at} claims {need} payload byte(s), too large for this platform",
                tag_name(*tag)
            ),
            Self::BadHeaderLength { actual } => write!(
                f,
                "C3DT header payload is {actual} byte(s), expected {HEADER_PAYLOAD_LEN}"
            ),
            Self::ShortPayload {
                tag,
                at,
                need,
                have,
            } => write!(
                f,
                "{} section at offset {at} has a {have}-byte payload, shorter than its {need}-byte fixed prefix",
                tag_name(*tag)
            ),
            Self::UnsupportedContainerVersion(v) => write!(
                f,
                "trace container version {v} is not supported (this build reads version {CONTAINER_VERSION})"
            ),
        }
    }
}

impl Error for TraceError {}

/// A tag as text for messages: the four bytes if printable ASCII,
/// otherwise an escaped/hex form. Mirrors `savestate::chunk::tag_name`'s
/// role for this container's own tags.
fn tag_name(tag: Tag) -> String {
    if tag.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
        format!("{}", tag.escape_ascii()).trim_end().to_string()
    } else {
        format!("{tag:02x?}")
    }
}

fn read_u32(bytes: &[u8]) -> u32 {
    // Callers only ever pass a slice already checked to be at least 4
    // bytes long, so this cannot fail; `expect` documents that invariant
    // rather than hiding a panic path.
    u32::from_be_bytes(bytes[..4].try_into().expect("checked length"))
}

/// The trace header's decoded fields (`C3DT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderSection {
    pub container_version: u32,
    pub protocol_version: u32,
    pub capability_mask: u32,
}

/// An initial aperture image (`APER`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApertureSection<'a> {
    pub offset: u32,
    pub data: &'a [u8],
}

/// One ring submission (`RING`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingSection<'a> {
    pub context: u32,
    pub ring_tail: u32,
    pub bytes: &'a [u8],
}

/// Guest memory a ref named at capture time (`BLOB`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobSection<'a> {
    /// Which address space `address` names: `0` the data aperture, `1`
    /// guest memory. A reference in the command stream carries the same
    /// distinction, and a trace captured from a `CAP_GUESTMEM` device has
    /// blobs in both, so the container has to carry it too -- without it
    /// a runner must guess, and a guess that differs from the capturing
    /// implementation's silently replays the trace against the wrong
    /// memory.
    pub space: u32,
    pub address: u32,
    pub data: &'a [u8],
}

/// A golden image for the draw surface at a fence (`GOLD`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldSection<'a> {
    pub context: u32,
    pub fence_id: u32,
    pub surface_id: u32,
    /// PNG bytes, opaque to this module: carried, never decoded.
    pub png: &'a [u8],
}

/// One decoded section, borrowing its payload from the slice
/// [`TraceReader`] was built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section<'a> {
    Header(HeaderSection),
    Aperture(ApertureSection<'a>),
    Ring(RingSection<'a>),
    Blob(BlobSection<'a>),
    Gold(GoldSection<'a>),
    /// A tag this module does not recognise, skipped over per the
    /// forward-compatibility rule in the module doc comment.
    Unknown {
        tag: Tag,
        payload: &'a [u8],
    },
}

impl<'a> Section<'a> {
    /// This section's tag, whichever variant it decoded as.
    pub fn tag(&self) -> Tag {
        match self {
            Self::Header(_) => TAG_HEADER,
            Self::Aperture(_) => TAG_APERTURE,
            Self::Ring(_) => TAG_RING,
            Self::Blob(_) => TAG_BLOB,
            Self::Gold(_) => TAG_GOLD,
            Self::Unknown { tag, .. } => *tag,
        }
    }
}

/// Iterates the sections of a trace held in memory as `&[u8]`, decoding
/// each payload without copying it. Stops cleanly at the end of the
/// slice; any framing problem short of that is reported once as a
/// [`TraceError`] and ends iteration (the iterator is fused after an
/// error: further calls return `None`, not a repeat of the error or an
/// attempt to resync).
pub struct TraceReader<'a> {
    data: &'a [u8],
    pos: usize,
    seen_header: bool,
    failed: bool,
}

impl<'a> TraceReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            seen_header: false,
            failed: false,
        }
    }

    /// Decode the fixed-width prefix common to `APER`/`RING`/`BLOB`/`GOLD`:
    /// `count` big-endian `u32` fields, then the trailing bytes. `tag`/`at`
    /// are only for the error message.
    fn split_prefix(
        tag: Tag,
        at: usize,
        payload: &'a [u8],
        count: usize,
    ) -> Result<(&'a [u8], &'a [u8]), TraceError> {
        let need = count * 4;
        if payload.len() < need {
            return Err(TraceError::ShortPayload {
                tag,
                at,
                need,
                have: payload.len() as u32,
            });
        }
        Ok(payload.split_at(need))
    }
}

impl<'a> Iterator for TraceReader<'a> {
    type Item = Result<Section<'a>, TraceError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        if self.pos == 0 && self.data.is_empty() {
            self.failed = true;
            return Some(Err(TraceError::Empty));
        }
        if self.pos == self.data.len() {
            // Clean end of file: every section so far was complete.
            return None;
        }

        let at = self.pos;
        let remaining = self.data.len() - at;
        if remaining < SECTION_HEADER_LEN {
            self.failed = true;
            return Some(Err(TraceError::TruncatedSectionHeader {
                at,
                have: remaining,
            }));
        }

        let tag: Tag = self.data[at..at + 4]
            .try_into()
            .expect("checked length above");
        let length = read_u32(&self.data[at + 4..at + 8]);
        let payload_start = at + SECTION_HEADER_LEN;
        let available = (self.data.len() - payload_start) as u64;
        if u64::from(length) > available {
            self.failed = true;
            return Some(Err(TraceError::TruncatedPayload {
                tag,
                at,
                need: length,
                have: available as usize,
            }));
        }
        let Ok(length_usize) = usize::try_from(length) else {
            self.failed = true;
            return Some(Err(TraceError::PayloadTooLarge {
                tag,
                at,
                need: length,
            }));
        };
        let payload_end = payload_start + length_usize;
        let payload = &self.data[payload_start..payload_end];

        if !self.seen_header && tag != TAG_HEADER {
            self.failed = true;
            return Some(Err(TraceError::MissingHeader { tag }));
        }
        if self.seen_header && tag == TAG_HEADER {
            self.failed = true;
            return Some(Err(TraceError::DuplicateHeader { at }));
        }

        let section = match tag {
            TAG_HEADER => {
                if payload.len() != HEADER_PAYLOAD_LEN {
                    self.failed = true;
                    return Some(Err(TraceError::BadHeaderLength {
                        actual: payload.len() as u32,
                    }));
                }
                let container_version = read_u32(&payload[0..4]);
                let protocol_version = read_u32(&payload[4..8]);
                let capability_mask = read_u32(&payload[8..12]);
                if container_version != CONTAINER_VERSION {
                    self.failed = true;
                    return Some(Err(TraceError::UnsupportedContainerVersion(
                        container_version,
                    )));
                }
                self.seen_header = true;
                Section::Header(HeaderSection {
                    container_version,
                    protocol_version,
                    capability_mask,
                })
            }
            TAG_APERTURE => {
                let (prefix, data) = match Self::split_prefix(tag, at, payload, 1) {
                    Ok(v) => v,
                    Err(e) => {
                        self.failed = true;
                        return Some(Err(e));
                    }
                };
                Section::Aperture(ApertureSection {
                    offset: read_u32(prefix),
                    data,
                })
            }
            TAG_RING => {
                let (prefix, bytes) = match Self::split_prefix(tag, at, payload, 2) {
                    Ok(v) => v,
                    Err(e) => {
                        self.failed = true;
                        return Some(Err(e));
                    }
                };
                Section::Ring(RingSection {
                    context: read_u32(&prefix[0..4]),
                    ring_tail: read_u32(&prefix[4..8]),
                    bytes,
                })
            }
            TAG_BLOB => {
                let (prefix, data) = match Self::split_prefix(tag, at, payload, 2) {
                    Ok(v) => v,
                    Err(e) => {
                        self.failed = true;
                        return Some(Err(e));
                    }
                };
                Section::Blob(BlobSection {
                    space: read_u32(prefix),
                    address: read_u32(&prefix[4..]),
                    data,
                })
            }
            TAG_GOLD => {
                let (prefix, png) = match Self::split_prefix(tag, at, payload, 3) {
                    Ok(v) => v,
                    Err(e) => {
                        self.failed = true;
                        return Some(Err(e));
                    }
                };
                Section::Gold(GoldSection {
                    context: read_u32(&prefix[0..4]),
                    fence_id: read_u32(&prefix[4..8]),
                    surface_id: read_u32(&prefix[8..12]),
                    png,
                })
            }
            _ => Section::Unknown { tag, payload },
        };

        self.pos = payload_end;
        Some(Ok(section))
    }
}

/// Frames trace sections onto any [`Write`]. Call [`Self::header`] exactly
/// once, first; the other methods return an [`io::Error`] if called before
/// it or if a length does not fit a `u32`, rather than writing a file a
/// [`TraceReader`] could not parse back.
pub struct TraceWriter<W: Write> {
    inner: W,
    wrote_header: bool,
}

impl<W: Write> TraceWriter<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            wrote_header: false,
        }
    }

    fn write_section(&mut self, tag: Tag, prefix: &[&[u8]], trailing: &[u8]) -> io::Result<()> {
        let prefix_len: usize = prefix.iter().map(|p| p.len()).sum();
        let total = prefix_len + trailing.len();
        let length = u32::try_from(total).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{} section payload is {total} bytes, too large for a u32 length",
                    tag_name(tag)
                ),
            )
        })?;
        self.inner.write_all(&tag)?;
        self.inner.write_all(&length.to_be_bytes())?;
        for part in prefix {
            self.inner.write_all(part)?;
        }
        self.inner.write_all(trailing)
    }

    fn require_header(&self) -> io::Result<()> {
        if self.wrote_header {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "the C3DT header must be written before any other trace section",
            ))
        }
    }

    /// Write the `C3DT` header. Must be the first call; a second call is
    /// refused.
    pub fn header(&mut self, protocol_version: u32, capability_mask: u32) -> io::Result<()> {
        if self.wrote_header {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "the C3DT header has already been written",
            ));
        }
        self.write_section(
            TAG_HEADER,
            &[
                &CONTAINER_VERSION.to_be_bytes(),
                &protocol_version.to_be_bytes(),
                &capability_mask.to_be_bytes(),
            ],
            &[],
        )?;
        self.wrote_header = true;
        Ok(())
    }

    /// Append an `APER` section.
    pub fn aperture(&mut self, offset: u32, data: &[u8]) -> io::Result<()> {
        self.require_header()?;
        self.write_section(TAG_APERTURE, &[&offset.to_be_bytes()], data)
    }

    /// Append a `RING` section: the bytes appended to `context`'s ring,
    /// and the `RING_TAIL` value the doorbell write then set.
    pub fn ring(&mut self, context: u32, ring_tail: u32, bytes: &[u8]) -> io::Result<()> {
        self.require_header()?;
        self.write_section(
            TAG_RING,
            &[&context.to_be_bytes(), &ring_tail.to_be_bytes()],
            bytes,
        )
    }

    /// Append a `BLOB` section: the bytes a ref named at capture time.
    pub fn blob(&mut self, space: u32, address: u32, data: &[u8]) -> io::Result<()> {
        self.require_header()?;
        self.write_section(
            TAG_BLOB,
            &[&space.to_be_bytes(), &address.to_be_bytes()],
            data,
        )
    }

    /// Append a `GOLD` section: a golden PNG for `surface_id` once
    /// `context`'s `FENCE_COMPLETED` reaches `fence_id`. `png` is carried
    /// opaquely; this module does not validate or decode it.
    pub fn gold(
        &mut self,
        context: u32,
        fence_id: u32,
        surface_id: u32,
        png: &[u8],
    ) -> io::Result<()> {
        self.require_header()?;
        self.write_section(
            TAG_GOLD,
            &[
                &context.to_be_bytes(),
                &fence_id.to_be_bytes(),
                &surface_id.to_be_bytes(),
            ],
            png,
        )
    }

    /// Hand the underlying writer back.
    pub fn into_inner(self) -> W {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_trace() -> Vec<u8> {
        let mut buf = Vec::new();
        let mut w = TraceWriter::new(&mut buf);
        w.header(0x0001_0004, 0b0101_1111).unwrap();
        w.aperture(0x1000, &[1, 2, 3, 4]).unwrap();
        w.ring(0, 0x40, &[0, 0, 0, 1, 0, 0, 0, 0]).unwrap();
        w.blob(0, 0x2000, b"texel data").unwrap();
        w.gold(0, 7, 1, b"\x89PNG\r\n\x1a\nfakepngbytes").unwrap();
        buf
    }

    #[test]
    fn round_trips_a_full_trace_through_writer_and_reader() {
        let buf = sample_trace();
        let sections: Vec<Section> = TraceReader::new(&buf).map(|r| r.unwrap()).collect();
        assert_eq!(sections.len(), 5);
        match sections[0] {
            Section::Header(h) => {
                assert_eq!(h.container_version, CONTAINER_VERSION);
                assert_eq!(h.protocol_version, 0x0001_0004);
                assert_eq!(h.capability_mask, 0b0101_1111);
            }
            _ => panic!("expected a header section"),
        }
        match sections[1] {
            Section::Aperture(a) => {
                assert_eq!(a.offset, 0x1000);
                assert_eq!(a.data, &[1, 2, 3, 4]);
            }
            _ => panic!("expected an aperture section"),
        }
        match sections[2] {
            Section::Ring(r) => {
                assert_eq!(r.context, 0);
                assert_eq!(r.ring_tail, 0x40);
                assert_eq!(r.bytes, &[0, 0, 0, 1, 0, 0, 0, 0]);
            }
            _ => panic!("expected a ring section"),
        }
        match sections[3] {
            Section::Blob(b) => {
                assert_eq!(b.address, 0x2000);
                assert_eq!(b.data, b"texel data");
            }
            _ => panic!("expected a blob section"),
        }
        match sections[4] {
            Section::Gold(g) => {
                assert_eq!(g.context, 0);
                assert_eq!(g.fence_id, 7);
                assert_eq!(g.surface_id, 1);
                assert_eq!(g.png, b"\x89PNG\r\n\x1a\nfakepngbytes");
            }
            _ => panic!("expected a gold section"),
        }
    }

    #[test]
    fn an_empty_file_is_reported_as_empty_not_as_zero_sections() {
        let err = TraceReader::new(&[]).next().unwrap().unwrap_err();
        assert_eq!(err, TraceError::Empty);
    }

    #[test]
    fn a_file_not_starting_with_the_header_is_rejected() {
        let mut buf = Vec::new();
        let mut w = TraceWriter::new(&mut buf);
        w.wrote_header = true; // bypass the writer's own ordering guard
        w.aperture(0, &[1, 2, 3, 4]).unwrap();
        let err = TraceReader::new(&buf).next().unwrap().unwrap_err();
        assert_eq!(err, TraceError::MissingHeader { tag: TAG_APERTURE });
    }

    #[test]
    fn a_second_header_section_is_rejected() {
        let mut buf = sample_trace();
        let mut extra = Vec::new();
        TraceWriter::new(&mut extra).header(1, 0).unwrap();
        buf.extend_from_slice(&extra);
        let mut reader = TraceReader::new(&buf);
        let mut last = None;
        for item in &mut reader {
            last = Some(item);
        }
        assert!(matches!(
            last,
            Some(Err(TraceError::DuplicateHeader { .. }))
        ));
    }

    #[test]
    fn a_truncated_section_header_is_reported_with_its_offset() {
        let buf = sample_trace();
        // Just past the header section (tag+length+12-byte payload), plus
        // three more bytes: enough for a second section's tag but not its
        // full 8-byte tag+length frame.
        let header_end = SECTION_HEADER_LEN + HEADER_PAYLOAD_LEN;
        let cut = &buf[..header_end + 3];
        let mut reader = TraceReader::new(cut);
        assert!(reader.next().unwrap().is_ok()); // the header section itself is intact
        let err = reader.next().unwrap().unwrap_err();
        assert!(matches!(err, TraceError::TruncatedSectionHeader { .. }));
    }

    #[test]
    fn a_payload_length_reaching_past_the_file_is_a_truncated_payload_error() {
        let mut buf = Vec::new();
        TraceWriter::new(&mut buf).header(1, 0).unwrap();
        buf.extend_from_slice(b"APER");
        buf.extend_from_slice(&100u32.to_be_bytes()); // claims 100 bytes, none present
        let mut reader = TraceReader::new(&buf);
        assert!(reader.next().unwrap().is_ok());
        let err = reader.next().unwrap().unwrap_err();
        assert!(matches!(
            err,
            TraceError::TruncatedPayload { need: 100, .. }
        ));
    }

    #[test]
    fn a_header_payload_of_the_wrong_length_is_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"C3DT");
        buf.extend_from_slice(&4u32.to_be_bytes());
        buf.extend_from_slice(&[0, 0, 0, 1]);
        let err = TraceReader::new(&buf).next().unwrap().unwrap_err();
        assert_eq!(err, TraceError::BadHeaderLength { actual: 4 });
    }

    #[test]
    fn a_ring_payload_shorter_than_its_prefix_is_rejected() {
        let mut buf = Vec::new();
        TraceWriter::new(&mut buf).header(1, 0).unwrap();
        buf.extend_from_slice(b"RING");
        buf.extend_from_slice(&3u32.to_be_bytes());
        buf.extend_from_slice(&[0, 0, 0]);
        let mut reader = TraceReader::new(&buf);
        assert!(reader.next().unwrap().is_ok());
        let err = reader.next().unwrap().unwrap_err();
        assert!(matches!(
            err,
            TraceError::ShortPayload {
                tag: TAG_RING,
                need: 8,
                have: 3,
                ..
            }
        ));
    }

    #[test]
    fn an_unsupported_container_version_is_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"C3DT");
        buf.extend_from_slice(&12u32.to_be_bytes());
        buf.extend_from_slice(&99u32.to_be_bytes()); // container_version
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&0u32.to_be_bytes());
        let err = TraceReader::new(&buf).next().unwrap().unwrap_err();
        assert_eq!(err, TraceError::UnsupportedContainerVersion(99));
    }

    #[test]
    fn an_unknown_section_tag_is_skipped_and_iteration_continues() {
        let mut buf = Vec::new();
        let mut w = TraceWriter::new(&mut buf);
        w.header(1, 0).unwrap();
        w.ring(0, 4, &[]).unwrap();
        // Splice an unrecognised section between the two real ones.
        let ring_section = buf.split_off(SECTION_HEADER_LEN + HEADER_PAYLOAD_LEN);
        buf.extend_from_slice(b"FUTR");
        buf.extend_from_slice(&4u32.to_be_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&ring_section);

        let sections: Vec<Section> = TraceReader::new(&buf).map(|r| r.unwrap()).collect();
        assert_eq!(sections.len(), 3);
        assert_eq!(
            sections[1],
            Section::Unknown {
                tag: *b"FUTR",
                payload: b"data"
            }
        );
        assert!(matches!(sections[2], Section::Ring(_)));
    }

    #[test]
    fn writer_refuses_a_section_before_the_header() {
        let mut buf = Vec::new();
        let mut w = TraceWriter::new(&mut buf);
        let err = w.aperture(0, &[1, 2, 3, 4]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn writer_refuses_a_second_header() {
        let mut buf = Vec::new();
        let mut w = TraceWriter::new(&mut buf);
        w.header(1, 0).unwrap();
        let err = w.header(1, 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    /// Feeds a corpus of mutated byte strings through the reader --
    /// truncation at every offset, and a single flipped byte at every
    /// offset -- of an otherwise-valid multi-section trace, and asserts
    /// the reader always terminates with either a full parse or a
    /// `TraceError`, never a panic. A panic here fails the test on its own
    /// (Rust aborts the test on an unwind), so the assertions below just
    /// confirm every path is actually exercised and produces a `Result`.
    #[test]
    fn the_reader_never_panics_on_truncated_or_corrupted_trace_bytes() {
        let base = sample_trace();

        for cut in 0..=base.len() {
            let slice = &base[..cut];
            // Draining to exhaustion without panicking is the assertion:
            // every item is a well-formed `Result`, whichever variant.
            for item in TraceReader::new(slice) {
                let _ = item;
            }
        }

        for flip in 0..base.len() {
            let mut mutated = base.clone();
            mutated[flip] ^= 0xFF;
            let reader = TraceReader::new(&mutated);
            for item in reader {
                // Reading to exhaustion without panicking is the assertion;
                // both Ok and Err are acceptable outcomes of a corrupted
                // byte.
                let _ = item;
            }
        }

        for length_field_at in [8usize /* header length */] {
            if length_field_at + 4 <= base.len() {
                let mut mutated = base.clone();
                mutated[length_field_at..length_field_at + 4]
                    .copy_from_slice(&u32::MAX.to_be_bytes());
                for item in TraceReader::new(&mutated) {
                    let _ = item;
                }
            }
        }
    }

    #[test]
    fn every_length_field_corrupted_to_a_huge_value_never_panics() {
        let base = sample_trace();
        let mut offset = 0usize;
        while offset + SECTION_HEADER_LEN <= base.len() {
            let length_at = offset + 4;
            let mut mutated = base.clone();
            mutated[length_at..length_at + 4].copy_from_slice(&0xFFFF_FFFFu32.to_be_bytes());
            for item in TraceReader::new(&mutated) {
                let _ = item;
            }
            // Advance past this section using the *original* trace's
            // framing so we visit each real section's length field once.
            let len = u32::from_be_bytes(base[length_at..length_at + 4].try_into().unwrap());
            offset += SECTION_HEADER_LEN + len as usize;
        }
    }
}
