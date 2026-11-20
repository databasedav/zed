use std::{
    io::{ErrorKind, Read},
    num::NonZero,
    path::{Path, PathBuf},
    process::{Child, Stdio as StdStdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, Receiver, TryRecvError},
    },
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, anyhow, bail};
use rodio::{ChannelCount, MixerDeviceSink, Player, Sample, SampleRate, Source};
use serde::Deserialize;
use util::{ResultExt as _, command::Stdio as AsyncStdio};

const MAX_DECODE_DIMENSION: u32 = 1920;
const MAX_TARGET_FPS: f64 = 60.0;
const MIN_TARGET_FPS: f64 = 1.0;
const DEFAULT_FPS: f64 = 30.0;
const VIDEO_CHANNEL_CAPACITY: usize = 8;
const AUDIO_CHANNEL_CAPACITY: usize = 16;
const AUDIO_SAMPLE_RATE_HZ: u32 = 48_000;
const AUDIO_CHANNEL_COUNT: u16 = 2;
const AUDIO_CHUNK_SAMPLES: usize = 4_800; // 50ms of stereo 48kHz audio.
const AUDIO_PREROLL_FALLBACK_TIMEOUT: Duration = Duration::from_millis(250);

pub const PLAYBACK_SPEEDS: &[f64] = &[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0];

#[derive(Clone, Debug, PartialEq)]
pub struct VideoMetadata {
    pub display_width: u32,
    pub display_height: u32,
    pub decode_width: u32,
    pub decode_height: u32,
    pub fps: f64,
    pub target_fps: f64,
    pub duration: Option<Duration>,
    pub video_stream_index: Option<usize>,
    pub audio_stream_index: Option<usize>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub file_size: u64,
}

impl VideoMetadata {
    pub fn has_video(&self) -> bool {
        self.video_stream_index.is_some() && self.decode_width > 0 && self.decode_height > 0
    }

    pub fn has_audio(&self) -> bool {
        self.audio_stream_index.is_some()
    }

    pub fn frame_duration(&self) -> Duration {
        let fps = if self.target_fps > 0.0 {
            self.target_fps
        } else {
            DEFAULT_FPS
        };
        Duration::from_secs_f64(1.0 / fps)
    }
}

#[derive(Clone, Debug)]
pub struct DecodedFrame {
    pub pts: Duration,
    pub width: u32,
    pub height: u32,
    pub bgra_bytes: Vec<u8>,
}

pub enum FramePollResult {
    Frame(DecodedFrame),
    Waiting,
    Ended,
    Error(String),
}

#[derive(Deserialize)]
struct FfprobeOutput {
    #[serde(default)]
    streams: Vec<FfprobeStream>,
    format: Option<FfprobeFormat>,
}

#[derive(Deserialize)]
struct FfprobeFormat {
    duration: Option<String>,
    size: Option<String>,
}

#[derive(Deserialize)]
struct FfprobeStream {
    index: usize,
    codec_type: Option<String>,
    codec_name: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    sample_aspect_ratio: Option<String>,
    avg_frame_rate: Option<String>,
    r_frame_rate: Option<String>,
    duration: Option<String>,
    #[serde(default)]
    disposition: Option<FfprobeDisposition>,
    #[serde(default)]
    tags: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    side_data_list: Vec<FfprobeSideData>,
}

#[derive(Deserialize)]
struct FfprobeDisposition {
    #[serde(default)]
    attached_pic: i32,
}

#[derive(Deserialize)]
struct FfprobeSideData {
    rotation: Option<serde_json::Value>,
}

pub async fn probe_video(path: &Path) -> Result<VideoMetadata> {
    let output = util::command::new_command("ffprobe")
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
        ])
        .arg(path)
        .stdin(AsyncStdio::null())
        .output()
        .await
        .map_err(|error| {
            if error.kind() == ErrorKind::NotFound {
                anyhow!(
                    "Could not find `ffprobe` on PATH. Install FFmpeg to play videos inline, or open in your default app."
                )
            } else {
                anyhow!("Failed to run `ffprobe`: {error}")
            }
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            bail!("`ffprobe` failed with status {}", output.status);
        } else {
            bail!("`ffprobe` failed: {stderr}");
        }
    }

    let file_size_from_fs = std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    parse_ffprobe_json(&output.stdout, file_size_from_fs)
}

pub(crate) fn parse_ffprobe_json(
    json_bytes: &[u8],
    fallback_file_size: u64,
) -> Result<VideoMetadata> {
    let probe: FfprobeOutput =
        serde_json::from_slice(json_bytes).context("Failed to parse `ffprobe` JSON output")?;

    let video_stream = probe
        .streams
        .iter()
        .find(|stream| {
            stream.codec_type.as_deref() == Some("video")
                && stream
                    .disposition
                    .as_ref()
                    .is_none_or(|disposition| disposition.attached_pic == 0)
        })
        .or_else(|| {
            probe
                .streams
                .iter()
                .find(|stream| stream.codec_type.as_deref() == Some("video"))
        });

    let audio_stream = probe
        .streams
        .iter()
        .find(|stream| stream.codec_type.as_deref() == Some("audio"));

    if video_stream.is_none() && audio_stream.is_none() {
        bail!("No playable video or audio streams found in file");
    }

    let (display_width, display_height, decode_width, decode_height, fps, target_fps) =
        if let Some(stream) = video_stream {
            let coded_width = stream.width.unwrap_or(0);
            let coded_height = stream.height.unwrap_or(0);
            if coded_width == 0 || coded_height == 0 {
                bail!("Video stream reported zero dimensions");
            }
            let (display_width, display_height) = compute_display_dimensions(
                coded_width,
                coded_height,
                stream.sample_aspect_ratio.as_deref(),
                extract_rotation_degrees(stream),
            );
            let (decode_width, decode_height) =
                compute_decode_dimensions(display_width, display_height, MAX_DECODE_DIMENSION);
            let fps = stream
                .avg_frame_rate
                .as_deref()
                .and_then(parse_frame_rate)
                .or_else(|| stream.r_frame_rate.as_deref().and_then(parse_frame_rate))
                .unwrap_or(DEFAULT_FPS);
            let target_fps = fps.clamp(MIN_TARGET_FPS, MAX_TARGET_FPS);
            (
                display_width,
                display_height,
                decode_width,
                decode_height,
                fps,
                target_fps,
            )
        } else {
            (0, 0, 0, 0, DEFAULT_FPS, DEFAULT_FPS)
        };

    let duration = probe
        .format
        .as_ref()
        .and_then(|format| format.duration.as_deref())
        .and_then(parse_duration_seconds)
        .or_else(|| {
            video_stream
                .and_then(|stream| stream.duration.as_deref())
                .and_then(parse_duration_seconds)
        })
        .or_else(|| {
            audio_stream
                .and_then(|stream| stream.duration.as_deref())
                .and_then(parse_duration_seconds)
        })
        .or_else(|| video_stream.and_then(extract_tag_duration))
        .or_else(|| audio_stream.and_then(extract_tag_duration));

    let file_size = probe
        .format
        .as_ref()
        .and_then(|format| format.size.as_deref())
        .and_then(|size| size.parse::<u64>().ok())
        .unwrap_or(fallback_file_size);

    Ok(VideoMetadata {
        display_width,
        display_height,
        decode_width,
        decode_height,
        fps,
        target_fps,
        duration,
        video_stream_index: video_stream.map(|stream| stream.index),
        audio_stream_index: audio_stream.map(|stream| stream.index),
        video_codec: video_stream.and_then(|stream| stream.codec_name.clone()),
        audio_codec: audio_stream.and_then(|stream| stream.codec_name.clone()),
        file_size,
    })
}

fn extract_rotation_degrees(stream: &FfprobeStream) -> i32 {
    for side_data in &stream.side_data_list {
        if let Some(rotation) = &side_data.rotation {
            if let Some(degrees) = rotation.as_i64() {
                return degrees as i32;
            }
            if let Some(degrees) = rotation.as_f64() {
                return degrees.round() as i32;
            }
            if let Some(text) = rotation.as_str()
                && let Ok(degrees) = text.trim().parse::<f64>()
            {
                return degrees.round() as i32;
            }
        }
    }

    if let Some(tags) = &stream.tags {
        for (key, value) in tags {
            if key.eq_ignore_ascii_case("rotate") {
                if let Some(text) = value.as_str()
                    && let Ok(degrees) = text.trim().parse::<f64>()
                {
                    return degrees.round() as i32;
                }
                if let Some(degrees) = value.as_i64() {
                    return degrees as i32;
                }
            }
        }
    }

    0
}

fn extract_tag_duration(stream: &FfprobeStream) -> Option<Duration> {
    let tags = stream.tags.as_ref()?;
    for (key, value) in tags {
        if key.eq_ignore_ascii_case("duration")
            && let Some(text) = value.as_str()
            && let Some(duration) = parse_hms_duration(text)
        {
            return Some(duration);
        }
    }
    None
}

pub(crate) fn parse_hms_duration(text: &str) -> Option<Duration> {
    let trimmed = text.trim();
    let mut parts = trimmed.split(':');
    let hours: f64 = parts.next()?.parse().ok()?;
    let minutes: f64 = parts.next()?.parse().ok()?;
    let seconds: f64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    let total_seconds = hours * 3600.0 + minutes * 60.0 + seconds;
    if total_seconds.is_finite() && total_seconds >= 0.0 {
        Some(Duration::from_secs_f64(total_seconds))
    } else {
        None
    }
}

pub(crate) fn parse_duration_seconds(text: &str) -> Option<Duration> {
    let seconds: f64 = text.trim().parse().ok()?;
    if seconds.is_finite() && seconds >= 0.0 {
        Some(Duration::from_secs_f64(seconds))
    } else {
        None
    }
}

pub(crate) fn parse_frame_rate(rate: &str) -> Option<f64> {
    let trimmed = rate.trim();
    if let Some((numerator_str, denominator_str)) = trimmed.split_once('/') {
        let numerator: f64 = numerator_str.parse().ok()?;
        let denominator: f64 = denominator_str.parse().ok()?;
        if denominator > 0.0 && numerator > 0.0 {
            let value = numerator / denominator;
            if value.is_finite() && value > 0.0 {
                return Some(value);
            }
        }
        None
    } else {
        let value: f64 = trimmed.parse().ok()?;
        if value.is_finite() && value > 0.0 {
            Some(value)
        } else {
            None
        }
    }
}

pub(crate) fn compute_display_dimensions(
    coded_width: u32,
    coded_height: u32,
    sample_aspect_ratio: Option<&str>,
    rotation_degrees: i32,
) -> (u32, u32) {
    let mut display_width = coded_width;
    let mut display_height = coded_height;

    if let Some(sample_aspect_ratio) = sample_aspect_ratio
        && let Some((numerator_str, denominator_str)) = sample_aspect_ratio.trim().split_once(':')
        && let (Ok(numerator), Ok(denominator)) =
            (numerator_str.parse::<u64>(), denominator_str.parse::<u64>())
        && numerator > 0
        && denominator > 0
        && numerator != denominator
    {
        display_width = ((coded_width as f64) * (numerator as f64) / (denominator as f64))
            .round()
            .max(1.0) as u32;
    }

    if rotation_degrees.rem_euclid(180) == 90 {
        std::mem::swap(&mut display_width, &mut display_height);
    }

    (display_width.max(1), display_height.max(1))
}

pub(crate) fn compute_decode_dimensions(
    display_width: u32,
    display_height: u32,
    max_dimension: u32,
) -> (u32, u32) {
    let longest_edge = display_width.max(display_height);
    if longest_edge <= max_dimension {
        return (display_width.max(1), display_height.max(1));
    }

    let scale = max_dimension as f64 / longest_edge as f64;
    let scaled_width = ((display_width as f64 * scale).round() as u32).max(2);
    let scaled_height = ((display_height as f64 * scale).round() as u32).max(2);
    // Keep even dimensions when downscaling for consistent scaler behavior.
    let even_width = (scaled_width / 2).max(1) * 2;
    let even_height = (scaled_height / 2).max(1) * 2;
    (even_width, even_height)
}

pub(crate) fn build_atempo_filter(rate: f64) -> Option<String> {
    if (rate - 1.0).abs() < 1e-3 {
        return None;
    }
    let mut remaining = rate.clamp(0.25, 4.0);
    let mut stages = Vec::new();
    while remaining < 0.5 {
        stages.push("atempo=0.5".to_string());
        remaining /= 0.5;
    }
    while remaining > 2.0 {
        stages.push("atempo=2.0".to_string());
        remaining /= 2.0;
    }
    stages.push(format!("atempo={remaining:.4}"));
    Some(stages.join(","))
}

pub async fn decode_single_frame(
    path: &Path,
    metadata: &VideoMetadata,
    position: Duration,
) -> Result<Option<DecodedFrame>> {
    let Some(video_stream_index) = metadata.video_stream_index else {
        return Ok(None);
    };
    if metadata.decode_width == 0 || metadata.decode_height == 0 {
        return Ok(None);
    }

    if let Some(frame) =
        run_single_frame_decode(path, metadata, video_stream_index, position).await?
    {
        return Ok(Some(frame));
    }

    // Seeking to the very end of a stream can land past the last keyframe/frame;
    // step slightly earlier so paused seeks at EOF still render a visible frame.
    if position > Duration::ZERO {
        let fallback_position = position.saturating_sub(Duration::from_millis(500));
        return run_single_frame_decode(path, metadata, video_stream_index, fallback_position)
            .await;
    }

    Ok(None)
}

async fn run_single_frame_decode(
    path: &Path,
    metadata: &VideoMetadata,
    video_stream_index: usize,
    position: Duration,
) -> Result<Option<DecodedFrame>> {
    let mut command = util::command::new_command("ffmpeg");
    command.args(["-nostdin", "-v", "error"]);
    if position > Duration::ZERO {
        command.args(["-ss", &format!("{:.6}", position.as_secs_f64())]);
    }
    command.arg("-i").arg(path);
    command.args([
        "-map",
        &format!("0:{video_stream_index}"),
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        "1",
        "-vf",
        &format!("scale={}:{}", metadata.decode_width, metadata.decode_height),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgra",
        "pipe:1",
    ]);

    let output = command
        .stdin(AsyncStdio::null())
        .output()
        .await
        .map_err(|error| {
            if error.kind() == ErrorKind::NotFound {
                anyhow!(
                    "Could not find `ffmpeg` on PATH. Install FFmpeg to play videos inline, or open in your default app."
                )
            } else {
                anyhow!("Failed to run `ffmpeg`: {error}")
            }
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            bail!("`ffmpeg` failed with status {}", output.status);
        } else {
            bail!("`ffmpeg` failed: {stderr}");
        }
    }

    let expected_bytes = (metadata.decode_width as usize) * (metadata.decode_height as usize) * 4;
    if output.stdout.len() < expected_bytes {
        return Ok(None);
    }

    let mut bgra_bytes = output.stdout;
    bgra_bytes.truncate(expected_bytes);
    Ok(Some(DecodedFrame {
        pts: position,
        width: metadata.decode_width,
        height: metadata.decode_height,
        bgra_bytes,
    }))
}

struct AudioPlaybackHandle {
    _output_sink: MixerDeviceSink,
    player: Player,
    samples_consumed: Arc<AtomicU64>,
    audio_started: Arc<AtomicBool>,
    audio_eof: Arc<AtomicBool>,
}

struct PcmChannelSource {
    receiver: Receiver<Vec<f32>>,
    current_chunk: Vec<f32>,
    chunk_offset: usize,
    channel_index: u16,
    stop_flag: Arc<AtomicBool>,
    samples_consumed: Arc<AtomicU64>,
    audio_started: Arc<AtomicBool>,
    audio_eof: Arc<AtomicBool>,
    finished: bool,
}

impl PcmChannelSource {
    fn new(
        receiver: Receiver<Vec<f32>>,
        stop_flag: Arc<AtomicBool>,
        samples_consumed: Arc<AtomicU64>,
        audio_started: Arc<AtomicBool>,
        audio_eof: Arc<AtomicBool>,
    ) -> Self {
        Self {
            receiver,
            current_chunk: Vec::new(),
            chunk_offset: 0,
            channel_index: 0,
            stop_flag,
            samples_consumed,
            audio_started,
            audio_eof,
            finished: false,
        }
    }

    fn advance_channel(&mut self) {
        self.channel_index = (self.channel_index + 1) % AUDIO_CHANNEL_COUNT;
    }
}

impl Iterator for PcmChannelSource {
    type Item = Sample;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        if self.stop_flag.load(Ordering::Relaxed) {
            if self.channel_index != 0 {
                self.advance_channel();
                return Some(0.0);
            }
            self.finished = true;
            return None;
        }

        if self.chunk_offset < self.current_chunk.len() {
            let sample = self.current_chunk[self.chunk_offset];
            self.chunk_offset += 1;
            self.samples_consumed.fetch_add(1, Ordering::Relaxed);
            self.advance_channel();
            return Some(sample);
        }

        match self.receiver.try_recv() {
            Ok(chunk) if !chunk.is_empty() => {
                self.audio_started.store(true, Ordering::Release);
                self.current_chunk = chunk;
                self.chunk_offset = 1;
                self.samples_consumed.fetch_add(1, Ordering::Relaxed);
                self.advance_channel();
                Some(self.current_chunk[0])
            }
            Ok(_) | Err(TryRecvError::Empty) => {
                // Non-blocking silence while FFmpeg buffers so the audio device callback never stalls.
                self.advance_channel();
                Some(0.0)
            }
            Err(TryRecvError::Disconnected) => {
                self.audio_eof.store(true, Ordering::Release);
                if self.channel_index != 0 {
                    self.advance_channel();
                    return Some(0.0);
                }
                self.finished = true;
                None
            }
        }
    }
}

impl Source for PcmChannelSource {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> ChannelCount {
        NonZero::new(AUDIO_CHANNEL_COUNT).unwrap_or(rodio::nz!(2))
    }

    fn sample_rate(&self) -> SampleRate {
        NonZero::new(AUDIO_SAMPLE_RATE_HZ).unwrap_or(rodio::nz!(48_000))
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

enum VideoDecoderMessage {
    Frame(DecodedFrame),
    Error(String),
}

pub struct PlaybackSession {
    stop_flag: Arc<AtomicBool>,
    video_child: Arc<Mutex<Option<Child>>>,
    audio_child: Arc<Mutex<Option<Child>>>,
    video_rx: Option<Receiver<VideoDecoderMessage>>,
    pending_frame: Option<DecodedFrame>,
    video_eof: bool,
    has_video: bool,
    audio_handle: Option<AudioPlaybackHandle>,
    start_position: Duration,
    playback_rate: f64,
    paused: bool,
    paused_position: Duration,
    started_at: Instant,
    wall_anchor_position: Duration,
    wall_anchor_instant: Instant,
    duration: Option<Duration>,
}

impl PlaybackSession {
    pub fn start(
        path: &Path,
        metadata: &VideoMetadata,
        start_position: Duration,
        playback_rate: f64,
        volume: f32,
        muted: bool,
        output_device: Option<cpal::DeviceId>,
    ) -> Result<Self> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let video_child = Arc::new(Mutex::new(None));
        let audio_child = Arc::new(Mutex::new(None));

        let video_rx = if let Some(video_stream_index) = metadata.video_stream_index {
            Some(spawn_video_decoder(
                path,
                metadata,
                video_stream_index,
                start_position,
                stop_flag.clone(),
                video_child.clone(),
            )?)
        } else {
            None
        };

        let audio_handle = if let Some(audio_stream_index) = metadata.audio_stream_index {
            match spawn_audio_decoder(
                path,
                audio_stream_index,
                start_position,
                playback_rate,
                volume,
                muted,
                output_device,
                stop_flag.clone(),
                audio_child.clone(),
            ) {
                Ok(handle) => Some(handle),
                Err(error) => {
                    log::warn!("Video viewer falling back to silent playback: {error:#}");
                    None
                }
            }
        } else {
            None
        };

        let now = Instant::now();
        Ok(Self {
            stop_flag,
            video_child,
            audio_child,
            video_rx,
            pending_frame: None,
            video_eof: !metadata.has_video(),
            has_video: metadata.has_video(),
            audio_handle,
            start_position,
            playback_rate,
            paused: false,
            paused_position: start_position,
            started_at: now,
            wall_anchor_position: start_position,
            wall_anchor_instant: now,
            duration: metadata.duration,
        })
    }

    pub fn current_position(&mut self) -> Duration {
        if self.paused {
            return self.paused_position;
        }

        if let Some(audio) = &self.audio_handle {
            let started = audio.audio_started.load(Ordering::Acquire);
            let finished = audio.audio_eof.load(Ordering::Acquire);
            if started && !finished {
                let samples = audio.samples_consumed.load(Ordering::Relaxed);
                let audio_wall_seconds =
                    samples as f64 / (AUDIO_SAMPLE_RATE_HZ as f64 * AUDIO_CHANNEL_COUNT as f64);
                let media_position = self.start_position
                    + Duration::from_secs_f64(audio_wall_seconds * self.playback_rate);
                self.wall_anchor_position = media_position;
                self.wall_anchor_instant = Instant::now();
                return self.clamp_to_duration(media_position);
            }
            if !started && self.started_at.elapsed() < AUDIO_PREROLL_FALLBACK_TIMEOUT {
                self.wall_anchor_position = self.start_position;
                self.wall_anchor_instant = Instant::now();
                return self.start_position;
            }
        }

        let elapsed = self.wall_anchor_instant.elapsed().as_secs_f64() * self.playback_rate;
        let media_position = self.wall_anchor_position + Duration::from_secs_f64(elapsed);
        self.clamp_to_duration(media_position)
    }

    fn clamp_to_duration(&self, position: Duration) -> Duration {
        if let Some(duration) = self.duration {
            position.min(duration)
        } else {
            position
        }
    }

    pub fn take_due_frame(&mut self) -> FramePollResult {
        let clock_position = self.current_position();
        let audio_ended = self
            .audio_handle
            .as_ref()
            .is_none_or(|audio| audio.audio_eof.load(Ordering::Acquire));

        if !self.has_video {
            if audio_ended
                || self
                    .duration
                    .is_some_and(|duration| duration > Duration::ZERO && clock_position >= duration)
            {
                return FramePollResult::Ended;
            }
            return FramePollResult::Waiting;
        }

        let mut latest_due_frame = None;

        if let Some(pending) = self.pending_frame.take() {
            if pending.pts <= clock_position {
                latest_due_frame = Some(pending);
            } else {
                self.pending_frame = Some(pending);
                return FramePollResult::Waiting;
            }
        }

        if let Some(receiver) = &self.video_rx {
            loop {
                match receiver.try_recv() {
                    Ok(VideoDecoderMessage::Frame(frame)) => {
                        if frame.pts <= clock_position {
                            latest_due_frame = Some(frame);
                        } else {
                            self.pending_frame = Some(frame);
                            break;
                        }
                    }
                    Ok(VideoDecoderMessage::Error(message)) => {
                        self.video_eof = true;
                        return FramePollResult::Error(message);
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.video_eof = true;
                        break;
                    }
                }
            }
        }

        if let Some(frame) = latest_due_frame {
            return FramePollResult::Frame(frame);
        }

        if self.video_eof
            && self.pending_frame.is_none()
            && (audio_ended
                || self.duration.is_some_and(|duration| {
                    duration > Duration::ZERO && clock_position >= duration
                }))
        {
            return FramePollResult::Ended;
        }

        FramePollResult::Waiting
    }

    pub fn pause(&mut self) {
        if self.paused {
            return;
        }
        let position = self.current_position();
        self.paused = true;
        self.paused_position = position;
        if let Some(audio) = &self.audio_handle {
            audio.player.pause();
        }
    }

    pub fn resume(&mut self) {
        if !self.paused {
            return;
        }
        self.paused = false;
        self.wall_anchor_position = self.paused_position;
        self.wall_anchor_instant = Instant::now();
        if let Some(audio) = &self.audio_handle {
            audio.player.play();
        }
    }

    pub fn set_volume(&self, volume: f32, muted: bool) {
        if let Some(audio) = &self.audio_handle {
            audio.player.set_volume(if muted { 0.0 } else { volume });
        }
    }

    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
        kill_child(&self.video_child);
        kill_child(&self.audio_child);
        self.video_rx.take();
        self.pending_frame.take();
        if let Some(audio) = self.audio_handle.take() {
            audio.player.stop();
        }
    }
}

impl Drop for PlaybackSession {
    fn drop(&mut self) {
        self.stop();
    }
}

fn kill_child(child_slot: &Arc<Mutex<Option<Child>>>) {
    if let Ok(mut guard) = child_slot.lock()
        && let Some(child) = guard.as_mut()
    {
        if let Err(error) = child.kill()
            && error.kind() != ErrorKind::InvalidInput
        {
            log::debug!("Failed to kill ffmpeg child process: {error}");
        }
    }
}

fn reap_child(child_slot: &Arc<Mutex<Option<Child>>>) {
    if let Ok(mut guard) = child_slot.lock()
        && let Some(mut child) = guard.take()
    {
        if let Err(error) = child.wait() {
            log::debug!("Failed to reap ffmpeg child process: {error}");
        }
    }
}

fn spawn_video_decoder(
    path: &Path,
    metadata: &VideoMetadata,
    video_stream_index: usize,
    start_position: Duration,
    stop_flag: Arc<AtomicBool>,
    child_slot: Arc<Mutex<Option<Child>>>,
) -> Result<Receiver<VideoDecoderMessage>> {
    let width = metadata.decode_width;
    let height = metadata.decode_height;
    let target_fps = metadata.target_fps;
    let frame_byte_len = (width as usize) * (height as usize) * 4;
    let path_for_thread = PathBuf::from(path);

    let (sender, receiver) = mpsc::sync_channel::<VideoDecoderMessage>(VIDEO_CHANNEL_CAPACITY);
    std::thread::Builder::new()
        .name("video-viewer-decoder".to_string())
        .spawn(move || {
            #[allow(
                clippy::disallowed_methods,
                reason = "blocking ffmpeg pipe I/O on dedicated media decoder OS thread"
            )]
            let spawn_result = {
                let mut command = util::command::new_std_command("ffmpeg");
                command.args(["-nostdin", "-v", "error"]);
                if start_position > Duration::ZERO {
                    command.args(["-ss", &format!("{:.6}", start_position.as_secs_f64())]);
                }
                command.arg("-i").arg(&path_for_thread);
                command.args([
                    "-map",
                    &format!("0:{video_stream_index}"),
                    "-an",
                    "-sn",
                    "-dn",
                    "-vf",
                    &format!("fps={target_fps:.6},scale={width}:{height}"),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgra",
                    "pipe:1",
                ]);
                command
                    .stdin(StdStdio::null())
                    .stdout(StdStdio::piped())
                    .stderr(StdStdio::null())
                    .spawn()
            };

            let mut child = match spawn_result {
                Ok(child) => child,
                Err(error) => {
                    let message = if error.kind() == ErrorKind::NotFound {
                        "Could not find `ffmpeg` on PATH. Install FFmpeg to play videos inline, or open in your default app.".to_string()
                    } else {
                        format!("Failed to spawn video decoder `ffmpeg`: {error}")
                    };
                    if sender.send(VideoDecoderMessage::Error(message)).is_err() {
                        log::debug!("Video viewer stopped before receiving decoder spawn error");
                    }
                    return;
                }
            };

            let Some(mut stdout) = child.stdout.take() else {
                if sender
                    .send(VideoDecoderMessage::Error(
                        "Video decoder `ffmpeg` stdout pipe unavailable".to_string(),
                    ))
                    .is_err()
                {
                    log::debug!("Video viewer stopped before receiving stdout error");
                }
                if let Ok(mut guard) = child_slot.lock() {
                    *guard = Some(child);
                }
                kill_child(&child_slot);
                reap_child(&child_slot);
                return;
            };

            if let Ok(mut guard) = child_slot.lock() {
                *guard = Some(child);
            }

            if stop_flag.load(Ordering::Acquire) {
                drop(stdout);
                kill_child(&child_slot);
                reap_child(&child_slot);
                return;
            }

            let mut frame_index: u64 = 0;
            while !stop_flag.load(Ordering::Acquire) {
                let mut buffer = vec![0u8; frame_byte_len];
                match stdout.read_exact(&mut buffer) {
                    Ok(()) => {
                        let pts = start_position
                            + Duration::from_secs_f64(frame_index as f64 / target_fps);
                        frame_index += 1;
                        let frame = DecodedFrame {
                            pts,
                            width,
                            height,
                            bgra_bytes: buffer,
                        };
                        if sender.send(VideoDecoderMessage::Frame(frame)).is_err() {
                            break;
                        }
                    }
                    Err(error) if error.kind() == ErrorKind::UnexpectedEof => {
                        break;
                    }
                    Err(error) => {
                        if !stop_flag.load(Ordering::Acquire) {
                            log::debug!(
                                "Video decoder read ended for {}: {error}",
                                path_for_thread.display()
                            );
                        }
                        break;
                    }
                }
            }
            drop(stdout);
            kill_child(&child_slot);
            reap_child(&child_slot);
        })
        .context("Failed to spawn video decoder thread")?;

    Ok(receiver)
}

#[allow(clippy::too_many_arguments)]
fn spawn_audio_decoder(
    path: &Path,
    audio_stream_index: usize,
    start_position: Duration,
    playback_rate: f64,
    volume: f32,
    muted: bool,
    output_device: Option<cpal::DeviceId>,
    stop_flag: Arc<AtomicBool>,
    child_slot: Arc<Mutex<Option<Child>>>,
) -> Result<AudioPlaybackHandle> {
    let mut output_sink =
        audio::open_test_output(output_device).context("Could not open audio output device")?;
    output_sink.log_on_drop(false);

    let (sender, receiver) = mpsc::sync_channel::<Vec<f32>>(AUDIO_CHANNEL_CAPACITY);
    let stop_for_thread = stop_flag.clone();
    let path_for_thread = PathBuf::from(path);
    std::thread::Builder::new()
        .name("video-viewer-audio".to_string())
        .spawn(move || {
            #[allow(
                clippy::disallowed_methods,
                reason = "blocking ffmpeg pipe I/O on dedicated media decoder OS thread"
            )]
            let spawn_result = {
                let mut command = util::command::new_std_command("ffmpeg");
                command.args(["-nostdin", "-v", "error"]);
                if start_position > Duration::ZERO {
                    command.args(["-ss", &format!("{:.6}", start_position.as_secs_f64())]);
                }
                command.arg("-i").arg(&path_for_thread);
                command.args([
                    "-map",
                    &format!("0:{audio_stream_index}"),
                    "-vn",
                    "-sn",
                    "-dn",
                ]);
                if let Some(atempo) = build_atempo_filter(playback_rate) {
                    command.args(["-af", &atempo]);
                }
                command.args([
                    "-f",
                    "f32le",
                    "-ac",
                    &AUDIO_CHANNEL_COUNT.to_string(),
                    "-ar",
                    &AUDIO_SAMPLE_RATE_HZ.to_string(),
                    "pipe:1",
                ]);
                command
                    .stdin(StdStdio::null())
                    .stdout(StdStdio::piped())
                    .stderr(StdStdio::null())
                    .spawn()
            };

            let mut child = match spawn_result {
                Ok(child) => child,
                Err(error) => {
                    log::warn!("Video viewer failed to spawn audio decoder `ffmpeg`: {error}");
                    return;
                }
            };

            let Some(mut stdout) = child.stdout.take() else {
                if let Ok(mut guard) = child_slot.lock() {
                    *guard = Some(child);
                }
                kill_child(&child_slot);
                reap_child(&child_slot);
                return;
            };

            if let Ok(mut guard) = child_slot.lock() {
                *guard = Some(child);
            }

            if stop_for_thread.load(Ordering::Acquire) {
                drop(stdout);
                kill_child(&child_slot);
                reap_child(&child_slot);
                return;
            }

            let chunk_bytes_len = AUDIO_CHUNK_SAMPLES * std::mem::size_of::<f32>();
            let mut raw_bytes = vec![0u8; chunk_bytes_len];
            let mut leftover = Vec::new();
            while !stop_for_thread.load(Ordering::Acquire) {
                match stdout.read(&mut raw_bytes) {
                    Ok(0) => break,
                    Ok(bytes_read) => {
                        leftover.extend_from_slice(&raw_bytes[..bytes_read]);
                        let frame_byte_stride =
                            std::mem::size_of::<f32>() * AUDIO_CHANNEL_COUNT as usize;
                        let usable_bytes = (leftover.len() / frame_byte_stride) * frame_byte_stride;
                        if usable_bytes == 0 {
                            continue;
                        }
                        let mut samples = Vec::with_capacity(usable_bytes / 4);
                        for sample_bytes in leftover[..usable_bytes].chunks_exact(4) {
                            if let Ok(array) = <[u8; 4]>::try_from(sample_bytes) {
                                samples.push(f32::from_le_bytes(array));
                            }
                        }
                        leftover.drain(..usable_bytes);
                        if !samples.is_empty() && sender.send(samples).is_err() {
                            break;
                        }
                    }
                    Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                    Err(error) => {
                        if !stop_for_thread.load(Ordering::Acquire) {
                            log::debug!("Audio decoder read ended: {error}");
                        }
                        break;
                    }
                }
            }
            drop(stdout);
            kill_child(&child_slot);
            reap_child(&child_slot);
        })
        .context("Failed to spawn audio decoder thread")?;

    let samples_consumed = Arc::new(AtomicU64::new(0));
    let audio_started = Arc::new(AtomicBool::new(false));
    let audio_eof = Arc::new(AtomicBool::new(false));
    let source = PcmChannelSource::new(
        receiver,
        stop_flag,
        samples_consumed.clone(),
        audio_started.clone(),
        audio_eof.clone(),
    );

    let player = Player::connect_new(output_sink.mixer());
    player.set_volume(if muted { 0.0 } else { volume });
    player.append(source);
    player.play();

    Ok(AudioPlaybackHandle {
        _output_sink: output_sink,
        player,
        samples_consumed,
        audio_started,
        audio_eof,
    })
}

pub fn format_timestamp(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

pub fn format_speed(speed: f64) -> String {
    if (speed - speed.round()).abs() < 1e-3 {
        format!("{}×", speed.round() as i32)
    } else if ((speed * 10.0) - (speed * 10.0).round()).abs() < 1e-3 {
        format!("{speed:.1}×")
    } else {
        format!("{speed:.2}×")
    }
}

pub fn next_playback_speed(current: f64) -> f64 {
    for &speed in PLAYBACK_SPEEDS {
        if speed > current + 1e-3 {
            return speed;
        }
    }
    PLAYBACK_SPEEDS[0]
}

pub fn step_playback_speed(current: f64, faster: bool) -> f64 {
    let mut closest_index = 0;
    let mut best_distance = f64::INFINITY;
    for (index, &speed) in PLAYBACK_SPEEDS.iter().enumerate() {
        let distance = (speed - current).abs();
        if distance < best_distance {
            best_distance = distance;
            closest_index = index;
        }
    }
    if faster {
        PLAYBACK_SPEEDS[(closest_index + 1).min(PLAYBACK_SPEEDS.len() - 1)]
    } else {
        PLAYBACK_SPEEDS[closest_index.saturating_sub(1)]
    }
}

pub fn drop_render_image(window: &mut gpui::Window, image: Arc<gpui::RenderImage>) {
    window.drop_image(image).log_err();
}
