mod playback;
mod video_info;

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Context as _;
use audio::AudioSettings;
use editor::{EditorSettings, RevealInFileManager, items::entry_git_aware_label_color};
use file_icons::FileIcons;
use gpui::{
    AnyElement, App, Bounds, Context, Corners, DispatchPhase, Element, ElementId, Entity,
    EventEmitter, FocusHandle, Focusable, Font, GlobalElementId, Hitbox, HitboxBehavior,
    InspectorElementId, InteractiveElement, IntoElement, LayoutId, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, ParentElement, Pixels, Render, RenderImage, Style, Styled, Task,
    WeakEntity, Window, actions, div, point, px, relative, size,
};
use image::{Frame, RgbaImage};
use persistence::VideoViewerDb;
use project::{Project, ProjectEntryId, ProjectPath};
use settings::Settings;
use smallvec::SmallVec;
use theme_settings::ThemeSettings;
use ui::{Button, ButtonStyle, IconButton, Tooltip, prelude::*};
use util::{ResultExt as _, paths::PathExt};
use workspace::{
    ItemId, ItemSettings, Pane, ToolbarItemLocation, Workspace, WorkspaceId, delete_unloaded_items,
    item::{HighlightedText, Item, ProjectItem, SerializableItem, TabContentParams},
};

use crate::playback::{
    DecodedFrame, FramePollResult, PlaybackSession, VideoMetadata, decode_single_frame,
    drop_render_image, format_speed, format_timestamp, next_playback_speed, probe_video,
    step_playback_speed,
};
pub use crate::video_info::VideoInfo;

actions!(
    video_viewer,
    [
        /// Toggle video playback between playing and paused.
        TogglePlayback,
        /// Seek backward by 5 seconds.
        SeekBackward,
        /// Seek forward by 5 seconds.
        SeekForward,
        /// Seek backward by 10 seconds.
        SeekBackwardLong,
        /// Seek forward by 10 seconds.
        SeekForwardLong,
        /// Step backward by one frame.
        StepBackward,
        /// Step forward by one frame.
        StepForward,
        /// Seek to the beginning of the video.
        SeekToStart,
        /// Toggle audio mute.
        ToggleMute,
        /// Increase playback volume.
        VolumeUp,
        /// Decrease playback volume.
        VolumeDown,
        /// Toggle looping playback.
        ToggleLoop,
        /// Increase playback speed.
        IncreaseSpeed,
        /// Decrease playback speed.
        DecreaseSpeed,
        /// Cycle through playback speeds.
        CycleSpeed,
        /// Open the video file in the system default application.
        OpenInDefaultApp
    ]
);

const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "m4v", "mov", "webm", "mkv", "avi", "ogv", "mpeg", "mpg", "ts", "mts",
];

const SHORT_SEEK_SECONDS: u64 = 5;
const LONG_SEEK_SECONDS: u64 = 10;
const VOLUME_STEP: f32 = 0.1;
const SCRUB_PREVIEW_THROTTLE: Duration = Duration::from_millis(100);

pub fn is_video_extension(extension: &str) -> bool {
    VIDEO_EXTENSIONS
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(extension))
}

pub fn is_video_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(is_video_extension)
}

fn is_video_file(project: &Entity<Project>, path: &ProjectPath, cx: &App) -> bool {
    if !project.read(cx).is_local() {
        return false;
    }

    if path.path.extension().is_some_and(is_video_extension) {
        return true;
    }

    // Single-file worktrees have an empty relative path, so check the absolute path too.
    project
        .read(cx)
        .absolute_path(path, cx)
        .is_some_and(|abs_path| is_video_path(&abs_path))
}

pub struct VideoItem {
    project_path: ProjectPath,
    abs_path: PathBuf,
    entry_id: Option<ProjectEntryId>,
    file_name: SharedString,
}

impl VideoItem {
    pub fn abs_path(&self) -> &Path {
        &self.abs_path
    }

    pub fn project_path(&self) -> &ProjectPath {
        &self.project_path
    }
}

impl project::ProjectItem for VideoItem {
    fn try_open(
        project: &Entity<Project>,
        path: &ProjectPath,
        cx: &mut App,
    ) -> Option<Task<anyhow::Result<Entity<Self>>>> {
        if !is_video_file(project, path, cx) {
            return None;
        }

        let abs_path = project.read(cx).absolute_path(path, cx)?;
        let entry_id = project
            .read(cx)
            .entry_for_path(path, cx)
            .map(|entry| entry.id);
        let file_name: SharedString = abs_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.path.as_unix_str().to_string())
            .into();
        let project_path = path.clone();

        Some(Task::ready(Ok(cx.new(|_| Self {
            project_path,
            abs_path,
            entry_id,
            file_name,
        }))))
    }

    fn entry_id(&self, _cx: &App) -> Option<ProjectEntryId> {
        self.entry_id
    }

    fn project_path(&self, _cx: &App) -> Option<ProjectPath> {
        Some(self.project_path.clone())
    }

    fn is_dirty(&self) -> bool {
        false
    }
}

pub enum VideoViewEvent {
    TitleChanged,
}

impl EventEmitter<VideoViewEvent> for VideoView {}

pub struct VideoView {
    video_item: Entity<VideoItem>,
    project: Entity<Project>,
    focus_handle: FocusHandle,
    metadata: Option<VideoMetadata>,
    load_error: Option<SharedString>,
    is_loading: bool,
    autoplay: bool,
    is_playing: bool,
    is_ended: bool,
    is_looping: bool,
    is_muted: bool,
    volume: f32,
    playback_rate: f64,
    current_position: Duration,
    current_frame: Option<Arc<RenderImage>>,
    session: Option<PlaybackSession>,
    load_task: Option<Task<()>>,
    seek_preview_task: Option<Task<()>>,
    seek_preview_generation: u64,
    is_scrubbing: bool,
    was_playing_before_scrub: bool,
    last_scrub_preview_at: Option<Instant>,
    scrub_bar_bounds: Option<Bounds<Pixels>>,
    is_adjusting_volume: bool,
    volume_bar_bounds: Option<Bounds<Pixels>>,
}

impl Drop for VideoView {
    fn drop(&mut self) {
        self.stop_playback();
    }
}

impl VideoView {
    pub fn new(
        video_item: Entity<VideoItem>,
        project: Entity<Project>,
        autoplay: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let git_store = project.read(cx).git_store().clone();
        cx.subscribe(&git_store, |_, _, event, cx| {
            if matches!(event, project::git_store::GitStoreEvent::DiffBaseChanged(_)) {
                cx.emit(VideoViewEvent::TitleChanged);
            }
        })
        .detach();

        cx.on_release_in(window, |this, _window, cx| {
            this.stop_playback();
            if let Some(frame) = this.current_frame.take() {
                cx.drop_image(frame, None);
            }
        })
        .detach();

        let mut view = Self {
            video_item,
            project,
            focus_handle: cx.focus_handle(),
            metadata: None,
            load_error: None,
            is_loading: true,
            autoplay,
            is_playing: false,
            is_ended: false,
            is_looping: false,
            is_muted: false,
            volume: 1.0,
            playback_rate: 1.0,
            current_position: Duration::ZERO,
            current_frame: None,
            session: None,
            load_task: None,
            seek_preview_task: None,
            seek_preview_generation: 0,
            is_scrubbing: false,
            was_playing_before_scrub: false,
            last_scrub_preview_at: None,
            scrub_bar_bounds: None,
            is_adjusting_volume: false,
            volume_bar_bounds: None,
        };
        view.load_video(window, cx);
        view
    }

    pub fn metadata(&self) -> Option<&VideoMetadata> {
        self.metadata.as_ref()
    }

    fn abs_path(&self, cx: &App) -> PathBuf {
        self.video_item.read(cx).abs_path().to_path_buf()
    }

    fn load_video(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.stop_playback();
        self.is_loading = true;
        self.load_error = None;
        self.metadata = None;
        self.current_position = Duration::ZERO;
        self.is_ended = false;
        cx.notify();

        let abs_path = self.abs_path(cx);
        let autoplay = self.autoplay;
        self.load_task = Some(cx.spawn_in(window, async move |this, cx| {
            let probe_result = cx
                .background_spawn({
                    let abs_path = abs_path.clone();
                    async move { probe_video(&abs_path).await }
                })
                .await;

            match probe_result {
                Ok(metadata) => {
                    let initial_frame = if !autoplay && metadata.has_video() {
                        cx.background_spawn({
                            let abs_path = abs_path.clone();
                            let metadata = metadata.clone();
                            async move {
                                decode_single_frame(&abs_path, &metadata, Duration::ZERO).await
                            }
                        })
                        .await
                        .ok()
                        .flatten()
                    } else {
                        None
                    };

                    this.update_in(cx, |this, window, cx| {
                        this.is_loading = false;
                        this.metadata = Some(metadata);
                        if let Some(frame) = initial_frame {
                            this.set_current_frame(frame, window);
                        }
                        if autoplay {
                            this.start_playback_from(Duration::ZERO, window, cx);
                        }
                        cx.notify();
                    })
                    .log_err();
                }
                Err(error) => {
                    this.update(cx, |this, cx| {
                        this.is_loading = false;
                        this.load_error = Some(error.to_string().into());
                        cx.notify();
                    })
                    .log_err();
                }
            }
        }));
    }

    fn stop_playback(&mut self) {
        if let Some(mut session) = self.session.take() {
            session.stop();
        }
        self.is_playing = false;
    }

    fn schedule_next_frame(&self, window: &Window, cx: &Context<Self>) {
        let entity_id = cx.entity_id();
        window.on_next_frame(move |_, cx| cx.notify(entity_id));
    }

    fn clamp_position(&self, position: Duration) -> Duration {
        if let Some(duration) = self
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.duration)
        {
            position.min(duration)
        } else {
            position
        }
    }

    fn start_playback_from(
        &mut self,
        position: Duration,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(metadata) = self.metadata.clone() else {
            return;
        };

        self.stop_playback();
        self.seek_preview_task.take();
        let clamped_position = self.clamp_position(position);
        let abs_path = self.abs_path(cx);
        let output_device = AudioSettings::get_global(cx).output_audio_device.clone();

        match PlaybackSession::start(
            &abs_path,
            &metadata,
            clamped_position,
            self.playback_rate,
            self.volume,
            self.is_muted,
            output_device,
        ) {
            Ok(session) => {
                self.session = Some(session);
                self.current_position = clamped_position;
                self.is_playing = true;
                self.is_ended = false;
                self.load_error = None;
                self.schedule_next_frame(window, cx);
                cx.notify();
            }
            Err(error) => {
                self.load_error = Some(error.to_string().into());
                self.is_playing = false;
                cx.notify();
            }
        }
    }

    fn set_current_frame(&mut self, decoded: DecodedFrame, window: &mut Window) {
        let Some(rgba_image) =
            RgbaImage::from_raw(decoded.width, decoded.height, decoded.bgra_bytes)
        else {
            return;
        };
        let render_image = Arc::new(RenderImage::new(SmallVec::from_const([Frame::new(
            rgba_image,
        )])));
        if let Some(previous) = self.current_frame.replace(render_image) {
            drop_render_image(window, previous);
        }
    }

    fn request_single_frame_preview(
        &mut self,
        position: Duration,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(metadata) = self.metadata.clone() else {
            return;
        };
        if !metadata.has_video() {
            return;
        }

        self.seek_preview_generation = self.seek_preview_generation.wrapping_add(1);
        let generation = self.seek_preview_generation;
        let abs_path = self.abs_path(cx);

        self.seek_preview_task = Some(cx.spawn_in(window, async move |this, cx| {
            let decoded = cx
                .background_spawn(async move {
                    decode_single_frame(&abs_path, &metadata, position).await
                })
                .await;

            if let Ok(Some(frame)) = decoded {
                this.update_in(cx, |this, window, cx| {
                    if this.seek_preview_generation == generation && !this.is_playing {
                        this.set_current_frame(frame, window);
                        cx.notify();
                    }
                })
                .log_err();
            }
        }));
    }

    fn advance_playback_frame(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.is_playing {
            return;
        }

        let Some(session) = self.session.as_mut() else {
            self.is_playing = false;
            return;
        };

        self.current_position = session.current_position();
        match session.take_due_frame() {
            FramePollResult::Frame(frame) => {
                self.set_current_frame(frame, window);
                self.schedule_next_frame(window, cx);
            }
            FramePollResult::Waiting => {
                self.schedule_next_frame(window, cx);
            }
            FramePollResult::Ended => {
                if self.is_looping {
                    self.start_playback_from(Duration::ZERO, window, cx);
                } else {
                    if let Some(duration) = self
                        .metadata
                        .as_ref()
                        .and_then(|metadata| metadata.duration)
                    {
                        self.current_position = duration;
                    }
                    self.stop_playback();
                    self.is_ended = true;
                }
            }
            FramePollResult::Error(message) => {
                self.load_error = Some(message.into());
                self.stop_playback();
            }
        }
    }

    fn toggle_playback(&mut self, _: &TogglePlayback, window: &mut Window, cx: &mut Context<Self>) {
        if self.metadata.is_none() {
            return;
        }

        if self.is_playing {
            if let Some(session) = self.session.as_mut() {
                session.pause();
                self.current_position = session.current_position();
            }
            self.is_playing = false;
            cx.notify();
            return;
        }

        if self.is_ended {
            self.start_playback_from(Duration::ZERO, window, cx);
            return;
        }

        if let Some(session) = self.session.as_mut() {
            session.resume();
            self.is_playing = true;
            self.schedule_next_frame(window, cx);
            cx.notify();
        } else {
            self.start_playback_from(self.current_position, window, cx);
        }
    }

    fn seek_to(&mut self, target: Duration, window: &mut Window, cx: &mut Context<Self>) {
        if self.metadata.is_none() {
            return;
        }
        let clamped = self.clamp_position(target);
        self.current_position = clamped;
        self.is_ended = false;

        if self.is_playing {
            self.start_playback_from(clamped, window, cx);
        } else {
            self.stop_playback();
            self.request_single_frame_preview(clamped, window, cx);
            cx.notify();
        }
    }

    fn seek_relative(&mut self, delta_seconds: i64, window: &mut Window, cx: &mut Context<Self>) {
        let base = if let Some(session) = self.session.as_mut() {
            session.current_position()
        } else {
            self.current_position
        };
        let target = if delta_seconds >= 0 {
            base.saturating_add(Duration::from_secs(delta_seconds as u64))
        } else {
            base.saturating_sub(Duration::from_secs(delta_seconds.unsigned_abs()))
        };
        self.seek_to(target, window, cx);
    }

    fn step_frame(&mut self, forward: bool, window: &mut Window, cx: &mut Context<Self>) {
        let Some(metadata) = self.metadata.as_ref() else {
            return;
        };
        let frame_step = metadata.frame_duration();
        let base = if let Some(session) = self.session.as_mut() {
            session.current_position()
        } else {
            self.current_position
        };
        self.stop_playback();
        let target = if forward {
            base.saturating_add(frame_step)
        } else {
            base.saturating_sub(frame_step)
        };
        let clamped = self.clamp_position(target);
        self.current_position = clamped;
        self.is_ended = false;
        self.request_single_frame_preview(clamped, window, cx);
        cx.notify();
    }

    fn seek_backward(&mut self, _: &SeekBackward, window: &mut Window, cx: &mut Context<Self>) {
        self.seek_relative(-(SHORT_SEEK_SECONDS as i64), window, cx);
    }

    fn seek_forward(&mut self, _: &SeekForward, window: &mut Window, cx: &mut Context<Self>) {
        self.seek_relative(SHORT_SEEK_SECONDS as i64, window, cx);
    }

    fn seek_backward_long(
        &mut self,
        _: &SeekBackwardLong,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.seek_relative(-(LONG_SEEK_SECONDS as i64), window, cx);
    }

    fn seek_forward_long(
        &mut self,
        _: &SeekForwardLong,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.seek_relative(LONG_SEEK_SECONDS as i64, window, cx);
    }

    fn step_backward(&mut self, _: &StepBackward, window: &mut Window, cx: &mut Context<Self>) {
        self.step_frame(false, window, cx);
    }

    fn step_forward(&mut self, _: &StepForward, window: &mut Window, cx: &mut Context<Self>) {
        self.step_frame(true, window, cx);
    }

    fn seek_to_start(&mut self, _: &SeekToStart, window: &mut Window, cx: &mut Context<Self>) {
        self.seek_to(Duration::ZERO, window, cx);
    }

    fn toggle_mute(&mut self, _: &ToggleMute, _window: &mut Window, cx: &mut Context<Self>) {
        self.is_muted = !self.is_muted;
        if let Some(session) = &self.session {
            session.set_volume(self.volume, self.is_muted);
        }
        cx.notify();
    }

    fn set_volume(&mut self, volume: f32, cx: &mut Context<Self>) {
        self.volume = volume.clamp(0.0, 1.0);
        if self.volume > 0.0 && self.is_muted {
            self.is_muted = false;
        }
        if let Some(session) = &self.session {
            session.set_volume(self.volume, self.is_muted);
        }
        cx.notify();
    }

    fn volume_up(&mut self, _: &VolumeUp, _window: &mut Window, cx: &mut Context<Self>) {
        self.set_volume(self.volume + VOLUME_STEP, cx);
    }

    fn volume_down(&mut self, _: &VolumeDown, _window: &mut Window, cx: &mut Context<Self>) {
        self.set_volume(self.volume - VOLUME_STEP, cx);
    }

    fn toggle_loop(&mut self, _: &ToggleLoop, _window: &mut Window, cx: &mut Context<Self>) {
        self.is_looping = !self.is_looping;
        cx.notify();
    }

    fn set_playback_rate(&mut self, rate: f64, window: &mut Window, cx: &mut Context<Self>) {
        if (self.playback_rate - rate).abs() < 1e-3 {
            return;
        }
        let current = if let Some(session) = self.session.as_mut() {
            session.current_position()
        } else {
            self.current_position
        };
        self.playback_rate = rate;
        if self.is_playing {
            self.start_playback_from(current, window, cx);
        } else {
            self.stop_playback();
            self.current_position = current;
            cx.notify();
        }
    }

    fn increase_speed(&mut self, _: &IncreaseSpeed, window: &mut Window, cx: &mut Context<Self>) {
        let next = step_playback_speed(self.playback_rate, true);
        self.set_playback_rate(next, window, cx);
    }

    fn decrease_speed(&mut self, _: &DecreaseSpeed, window: &mut Window, cx: &mut Context<Self>) {
        let prev = step_playback_speed(self.playback_rate, false);
        self.set_playback_rate(prev, window, cx);
    }

    fn cycle_speed(&mut self, _: &CycleSpeed, window: &mut Window, cx: &mut Context<Self>) {
        let next = next_playback_speed(self.playback_rate);
        self.set_playback_rate(next, window, cx);
    }

    fn open_in_default_app(
        &mut self,
        _: &OpenInDefaultApp,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let abs_path = self.abs_path(cx);
        cx.open_with_system(&abs_path);
    }

    fn reveal_in_file_manager(
        &mut self,
        _: &RevealInFileManager,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let abs_path = self.abs_path(cx);
        self.project
            .update(cx, |project, cx| project.reveal_path(&abs_path, cx));
    }

    fn position_for_scrub_x(&self, x: Pixels) -> Option<Duration> {
        let bounds = self.scrub_bar_bounds?;
        let duration = self.metadata.as_ref()?.duration?;
        if duration.is_zero() || bounds.size.width <= Pixels::ZERO {
            return Some(Duration::ZERO);
        }
        let ratio = ((x - bounds.origin.x) / bounds.size.width).clamp(0.0, 1.0) as f64;
        Some(Duration::from_secs_f64(duration.as_secs_f64() * ratio))
    }

    fn begin_scrub(&mut self, x: Pixels, window: &mut Window, cx: &mut Context<Self>) {
        let Some(target) = self.position_for_scrub_x(x) else {
            return;
        };
        self.was_playing_before_scrub = self.is_playing;
        self.stop_playback();
        self.is_scrubbing = true;
        self.is_ended = false;
        self.current_position = target;
        self.last_scrub_preview_at = Some(Instant::now());
        self.request_single_frame_preview(target, window, cx);
        cx.notify();
    }

    fn update_scrub(&mut self, x: Pixels, window: &mut Window, cx: &mut Context<Self>) {
        if !self.is_scrubbing {
            return;
        }
        let Some(target) = self.position_for_scrub_x(x) else {
            return;
        };
        self.current_position = target;
        let should_decode_preview = self
            .last_scrub_preview_at
            .is_none_or(|last| last.elapsed() >= SCRUB_PREVIEW_THROTTLE);
        if should_decode_preview {
            self.last_scrub_preview_at = Some(Instant::now());
            self.request_single_frame_preview(target, window, cx);
        }
        cx.notify();
    }

    fn finish_scrub(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.is_scrubbing {
            return;
        }
        self.is_scrubbing = false;
        let target = self.current_position;
        if self.was_playing_before_scrub {
            self.was_playing_before_scrub = false;
            self.start_playback_from(target, window, cx);
        } else {
            self.request_single_frame_preview(target, window, cx);
            cx.notify();
        }
    }

    fn volume_for_bar_x(&self, x: Pixels) -> Option<f32> {
        let bounds = self.volume_bar_bounds?;
        if bounds.size.width <= Pixels::ZERO {
            return None;
        }
        Some(((x - bounds.origin.x) / bounds.size.width).clamp(0.0, 1.0))
    }

    fn begin_volume_drag(&mut self, x: Pixels, cx: &mut Context<Self>) {
        self.is_adjusting_volume = true;
        if let Some(volume) = self.volume_for_bar_x(x) {
            self.set_volume(volume, cx);
        }
    }

    fn update_volume_drag(&mut self, x: Pixels, cx: &mut Context<Self>) {
        if !self.is_adjusting_volume {
            return;
        }
        if let Some(volume) = self.volume_for_bar_x(x) {
            self.set_volume(volume, cx);
        }
    }

    fn finish_volume_drag(&mut self, cx: &mut Context<Self>) {
        if self.is_adjusting_volume {
            self.is_adjusting_volume = false;
            cx.notify();
        }
    }

    fn progress_ratio(&self) -> f32 {
        let Some(duration) = self.metadata.as_ref().and_then(|meta| meta.duration) else {
            return 0.0;
        };
        if duration.is_zero() {
            return 0.0;
        }
        (self.current_position.as_secs_f64() / duration.as_secs_f64()).clamp(0.0, 1.0) as f32
    }

    fn render_viewport(&self, cx: &mut Context<Self>) -> AnyElement {
        if let Some(error) = self.load_error.as_ref() {
            return v_flex()
                .flex_1()
                .size_full()
                .items_center()
                .justify_center()
                .gap_3()
                .px_6()
                .child(
                    Label::new("Unable to play video")
                        .size(LabelSize::Large)
                        .color(Color::Error),
                )
                .child(
                    Label::new(error.clone())
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
                .child(
                    h_flex()
                        .gap_2()
                        .child(
                            Button::new("retry-video", "Retry")
                                .style(ButtonStyle::Filled)
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.load_video(window, cx);
                                })),
                        )
                        .child(
                            Button::new("open-video-external", "Open in Default App")
                                .style(ButtonStyle::Subtle)
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.open_in_default_app(&OpenInDefaultApp, window, cx);
                                })),
                        ),
                )
                .into_any_element();
        }

        if self.is_loading {
            return v_flex()
                .flex_1()
                .size_full()
                .items_center()
                .justify_center()
                .gap_2()
                .child(
                    Label::new("Loading video...")
                        .size(LabelSize::Default)
                        .color(Color::Muted),
                )
                .into_any_element();
        }

        if self
            .metadata
            .as_ref()
            .is_some_and(|metadata| !metadata.has_video())
        {
            let file_name = self.video_item.read(cx).file_name.clone();
            return v_flex()
                .flex_1()
                .size_full()
                .items_center()
                .justify_center()
                .gap_2()
                .cursor_pointer()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|this, _, window, cx| {
                        this.focus_handle.focus(window, cx);
                        this.toggle_playback(&TogglePlayback, window, cx);
                    }),
                )
                .child(
                    Icon::new(IconName::AudioOn)
                        .size(IconSize::XLarge)
                        .color(Color::Muted),
                )
                .child(Label::new(file_name).size(LabelSize::Default))
                .child(
                    Label::new("Audio-only media file")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
                .into_any_element();
        }

        div()
            .id("video-viewport")
            .flex_1()
            .w_full()
            .min_h_0()
            .overflow_hidden()
            .bg(gpui::black())
            .cursor_pointer()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, window, cx| {
                    this.focus_handle.focus(window, cx);
                    this.toggle_playback(&TogglePlayback, window, cx);
                }),
            )
            .child(VideoFrameElement::new(cx.entity()))
            .into_any_element()
    }

    fn render_controls(&self, cx: &mut Context<Self>) -> AnyElement {
        let has_metadata = self.metadata.is_some();
        let has_audio = self.metadata.as_ref().is_some_and(|meta| meta.has_audio());
        let has_duration = self
            .metadata
            .as_ref()
            .and_then(|meta| meta.duration)
            .is_some_and(|duration| !duration.is_zero());

        let elapsed_text = format_timestamp(self.current_position);
        let time_label =
            if let Some(duration) = self.metadata.as_ref().and_then(|meta| meta.duration) {
                format!("{elapsed_text} / {}", format_timestamp(duration))
            } else {
                elapsed_text
            };

        let speed_label = format_speed(self.playback_rate);
        let effective_volume = if self.is_muted { 0.0 } else { self.volume };

        v_flex()
            .w_full()
            .flex_none()
            .bg(cx.theme().colors().toolbar_background)
            .border_t_1()
            .border_color(cx.theme().colors().border_variant)
            .px_3()
            .pt_1p5()
            .pb_2()
            .gap_1()
            .when(has_duration, |this| {
                this.child(ScrubBarElement::new(cx.entity()))
            })
            .child(
                h_flex()
                    .w_full()
                    .justify_between()
                    .items_center()
                    .gap_2()
                    .child(
                        h_flex()
                            .items_center()
                            .gap_1()
                            .child(
                                IconButton::new(
                                    "video-toggle-playback",
                                    if self.is_playing {
                                        IconName::DebugPause
                                    } else {
                                        IconName::PlayFilled
                                    },
                                )
                                .disabled(!has_metadata)
                                .icon_size(IconSize::Small)
                                .tooltip({
                                    let is_playing = self.is_playing;
                                    move |_window, cx| {
                                        Tooltip::for_action(
                                            if is_playing { "Pause" } else { "Play" },
                                            &TogglePlayback,
                                            cx,
                                        )
                                    }
                                })
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.toggle_playback(&TogglePlayback, window, cx);
                                    },
                                )),
                            )
                            .child(
                                IconButton::new("video-seek-backward", IconName::RotateCcw)
                                    .disabled(!has_metadata)
                                    .icon_size(IconSize::Small)
                                    .tooltip(|_window, cx| {
                                        Tooltip::for_action("Seek Backward 5s", &SeekBackward, cx)
                                    })
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.seek_backward(&SeekBackward, window, cx);
                                    })),
                            )
                            .child(
                                IconButton::new("video-seek-forward", IconName::RotateCw)
                                    .disabled(!has_metadata)
                                    .icon_size(IconSize::Small)
                                    .tooltip(|_window, cx| {
                                        Tooltip::for_action("Seek Forward 5s", &SeekForward, cx)
                                    })
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.seek_forward(&SeekForward, window, cx);
                                    })),
                            )
                            .child(
                                div().pl_1().child(
                                    Label::new(time_label)
                                        .size(LabelSize::Small)
                                        .color(Color::Muted),
                                ),
                            ),
                    )
                    .child(
                        h_flex()
                            .items_center()
                            .gap_1()
                            .child(
                                Button::new("video-speed", speed_label)
                                    .disabled(!has_metadata)
                                    .style(ButtonStyle::Subtle)
                                    .label_size(LabelSize::Small)
                                    .tooltip(|_window, cx| {
                                        Tooltip::for_action("Cycle Playback Speed", &CycleSpeed, cx)
                                    })
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.cycle_speed(&CycleSpeed, window, cx);
                                    })),
                            )
                            .child(
                                IconButton::new("video-loop", IconName::Rerun)
                                    .disabled(!has_metadata)
                                    .icon_size(IconSize::Small)
                                    .toggle_state(self.is_looping)
                                    .tooltip({
                                        let is_looping = self.is_looping;
                                        move |_window, cx| {
                                            Tooltip::for_action(
                                                if is_looping {
                                                    "Disable Loop"
                                                } else {
                                                    "Enable Loop"
                                                },
                                                &ToggleLoop,
                                                cx,
                                            )
                                        }
                                    })
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.toggle_loop(&ToggleLoop, window, cx);
                                    })),
                            )
                            .when(has_audio, |this| {
                                this.child(
                                    IconButton::new(
                                        "video-mute",
                                        if effective_volume <= 0.0 {
                                            IconName::AudioOff
                                        } else {
                                            IconName::AudioOn
                                        },
                                    )
                                    .icon_size(IconSize::Small)
                                    .tooltip({
                                        let is_muted = self.is_muted;
                                        move |_window, cx| {
                                            Tooltip::for_action(
                                                if is_muted { "Unmute" } else { "Mute" },
                                                &ToggleMute,
                                                cx,
                                            )
                                        }
                                    })
                                    .on_click(cx.listener(
                                        |this, _, window, cx| {
                                            this.toggle_mute(&ToggleMute, window, cx);
                                        },
                                    )),
                                )
                                .child(VolumeBarElement::new(cx.entity()))
                            })
                            .child(
                                IconButton::new("video-open-external", IconName::ArrowUpRight)
                                    .icon_size(IconSize::Small)
                                    .tooltip(|_window, cx| {
                                        Tooltip::for_action(
                                            "Open in Default App",
                                            &OpenInDefaultApp,
                                            cx,
                                        )
                                    })
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.open_in_default_app(&OpenInDefaultApp, window, cx);
                                    })),
                            ),
                    ),
            )
            .into_any_element()
    }
}

struct VideoFrameElement {
    video_view: Entity<VideoView>,
}

impl VideoFrameElement {
    fn new(video_view: Entity<VideoView>) -> Self {
        Self { video_view }
    }
}

impl Element for VideoFrameElement {
    type RequestLayoutState = ();
    type PrepaintState = Option<(Arc<RenderImage>, Bounds<Pixels>)>;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        (
            window.request_layout(
                Style {
                    size: size(relative(1.0).into(), relative(1.0).into()),
                    ..Default::default()
                },
                [],
                cx,
            ),
            (),
        )
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let frame = self.video_view.read(cx).current_frame.clone()?;
        let image_size = frame.size(0);
        let frame_width = image_size.width.0 as f32;
        let frame_height = image_size.height.0 as f32;
        if frame_width <= 0.0 || frame_height <= 0.0 {
            return None;
        }

        let container_width: f32 = bounds.size.width.into();
        let container_height: f32 = bounds.size.height.into();
        if container_width <= 0.0 || container_height <= 0.0 {
            return None;
        }

        let scale = (container_width / frame_width).min(container_height / frame_height);
        let draw_width = px(frame_width * scale);
        let draw_height = px(frame_height * scale);
        let origin = point(
            bounds.origin.x + (bounds.size.width - draw_width) / 2.0,
            bounds.origin.y + (bounds.size.height - draw_height) / 2.0,
        );
        let image_bounds = Bounds::new(origin, size(draw_width, draw_height));

        Some((frame, image_bounds))
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        _cx: &mut App,
    ) {
        if let Some((frame, image_bounds)) = prepaint.take() {
            window
                .paint_image(bounds, image_bounds, Corners::default(), frame, 0, false)
                .log_err();
        }
    }
}

struct ScrubBarElement {
    video_view: Entity<VideoView>,
}

impl ScrubBarElement {
    fn new(video_view: Entity<VideoView>) -> Self {
        Self { video_view }
    }
}

impl Element for ScrubBarElement {
    type RequestLayoutState = ();
    type PrepaintState = Option<(AnyElement, Hitbox, bool)>;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        (
            window.request_layout(
                Style {
                    size: size(relative(1.0).into(), px(16.0).into()),
                    ..Default::default()
                },
                [],
                cx,
            ),
            (),
        )
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let (ratio, is_scrubbing) = self.video_view.update(cx, |this, _cx| {
            this.scrub_bar_bounds = Some(bounds);
            (this.progress_ratio(), this.is_scrubbing)
        });

        let track_bg = cx.theme().colors().element_background;
        let fill_color = cx.theme().colors().text_accent;
        let thumb_size = px(10.0);
        let fill_width = bounds.size.width * ratio;
        let thumb_left = (fill_width - thumb_size / 2.0)
            .max(px(0.0))
            .min((bounds.size.width - thumb_size).max(px(0.0)));

        let mut content = div()
            .size_full()
            .flex()
            .items_center()
            .cursor_pointer()
            .child(
                div()
                    .relative()
                    .w_full()
                    .h(px(4.0))
                    .rounded_full()
                    .bg(track_bg)
                    .child(div().h_full().w(fill_width).rounded_full().bg(fill_color))
                    .child(
                        div()
                            .absolute()
                            .top(px(-3.0))
                            .left(thumb_left)
                            .size(thumb_size)
                            .rounded_full()
                            .bg(fill_color),
                    ),
            )
            .into_any_element();

        content.prepaint_as_root(bounds.origin, bounds.size.into(), window, cx);
        let hitbox = window.insert_hitbox(bounds, HitboxBehavior::Normal);
        Some((content, hitbox, is_scrubbing))
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let Some((mut element, hitbox, is_scrubbing)) = prepaint.take() else {
            return;
        };

        let video_view_down = self.video_view.downgrade();
        window.on_mouse_event(move |event: &MouseDownEvent, phase, window, cx| {
            if phase == DispatchPhase::Bubble
                && event.button == MouseButton::Left
                && hitbox.is_hovered(window)
                && let Some(entity) = video_view_down.upgrade()
            {
                entity.update(cx, |this, cx| {
                    this.focus_handle.focus(window, cx);
                    this.begin_scrub(event.position.x, window, cx);
                    cx.stop_propagation();
                });
            }
        });

        if is_scrubbing {
            let video_view_move = self.video_view.downgrade();
            window.on_mouse_event(move |event: &MouseMoveEvent, phase, window, cx| {
                if phase == DispatchPhase::Bubble
                    && let Some(entity) = video_view_move.upgrade()
                {
                    entity.update(cx, |this, cx| {
                        if event.dragging() {
                            this.update_scrub(event.position.x, window, cx);
                        } else {
                            this.finish_scrub(window, cx);
                        }
                    });
                }
            });

            let video_view_up = self.video_view.downgrade();
            window.on_mouse_event(move |_event: &MouseUpEvent, phase, window, cx| {
                if phase == DispatchPhase::Bubble
                    && let Some(entity) = video_view_up.upgrade()
                {
                    entity.update(cx, |this, cx| {
                        this.finish_scrub(window, cx);
                    });
                }
            });
        }

        element.paint(window, cx);
    }
}

struct VolumeBarElement {
    video_view: Entity<VideoView>,
}

impl VolumeBarElement {
    fn new(video_view: Entity<VideoView>) -> Self {
        Self { video_view }
    }
}

impl Element for VolumeBarElement {
    type RequestLayoutState = ();
    type PrepaintState = Option<(AnyElement, Hitbox, bool)>;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        (
            window.request_layout(
                Style {
                    size: size(px(64.0).into(), px(16.0).into()),
                    ..Default::default()
                },
                [],
                cx,
            ),
            (),
        )
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let (effective_volume, is_adjusting) = self.video_view.update(cx, |this, _cx| {
            this.volume_bar_bounds = Some(bounds);
            let volume = if this.is_muted { 0.0 } else { this.volume };
            (volume, this.is_adjusting_volume)
        });

        let track_bg = cx.theme().colors().element_background;
        let fill_color = cx.theme().colors().text_accent;
        let fill_width = bounds.size.width * effective_volume;

        let mut content = div()
            .size_full()
            .flex()
            .items_center()
            .cursor_pointer()
            .child(
                div()
                    .w_full()
                    .h(px(4.0))
                    .rounded_full()
                    .bg(track_bg)
                    .child(div().h_full().w(fill_width).rounded_full().bg(fill_color)),
            )
            .into_any_element();

        content.prepaint_as_root(bounds.origin, bounds.size.into(), window, cx);
        let hitbox = window.insert_hitbox(bounds, HitboxBehavior::Normal);
        Some((content, hitbox, is_adjusting))
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let Some((mut element, hitbox, is_adjusting)) = prepaint.take() else {
            return;
        };

        let video_view_down = self.video_view.downgrade();
        window.on_mouse_event(move |event: &MouseDownEvent, phase, window, cx| {
            if phase == DispatchPhase::Bubble
                && event.button == MouseButton::Left
                && hitbox.is_hovered(window)
                && let Some(entity) = video_view_down.upgrade()
            {
                entity.update(cx, |this, cx| {
                    this.focus_handle.focus(window, cx);
                    this.begin_volume_drag(event.position.x, cx);
                    cx.stop_propagation();
                });
            }
        });

        if is_adjusting {
            let video_view_move = self.video_view.downgrade();
            window.on_mouse_event(move |event: &MouseMoveEvent, phase, _window, cx| {
                if phase == DispatchPhase::Bubble
                    && let Some(entity) = video_view_move.upgrade()
                {
                    entity.update(cx, |this, cx| {
                        if event.dragging() {
                            this.update_volume_drag(event.position.x, cx);
                        } else {
                            this.finish_volume_drag(cx);
                        }
                    });
                }
            });

            let video_view_up = self.video_view.downgrade();
            window.on_mouse_event(move |_event: &MouseUpEvent, phase, _window, cx| {
                if phase == DispatchPhase::Bubble
                    && let Some(entity) = video_view_up.upgrade()
                {
                    entity.update(cx, |this, cx| {
                        this.finish_volume_drag(cx);
                    });
                }
            });
        }

        element.paint(window, cx);
    }
}

impl Focusable for VideoView {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for VideoView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.advance_playback_frame(window, cx);

        v_flex()
            .track_focus(&self.focus_handle(cx))
            .key_context("VideoViewer")
            .on_action(cx.listener(Self::toggle_playback))
            .on_action(cx.listener(Self::seek_backward))
            .on_action(cx.listener(Self::seek_forward))
            .on_action(cx.listener(Self::seek_backward_long))
            .on_action(cx.listener(Self::seek_forward_long))
            .on_action(cx.listener(Self::step_backward))
            .on_action(cx.listener(Self::step_forward))
            .on_action(cx.listener(Self::seek_to_start))
            .on_action(cx.listener(Self::toggle_mute))
            .on_action(cx.listener(Self::volume_up))
            .on_action(cx.listener(Self::volume_down))
            .on_action(cx.listener(Self::toggle_loop))
            .on_action(cx.listener(Self::increase_speed))
            .on_action(cx.listener(Self::decrease_speed))
            .on_action(cx.listener(Self::cycle_speed))
            .on_action(cx.listener(Self::open_in_default_app))
            .on_action(cx.listener(Self::reveal_in_file_manager))
            .size_full()
            .bg(cx.theme().colors().editor_background)
            .child(self.render_viewport(cx))
            .child(self.render_controls(cx))
    }
}

impl Item for VideoView {
    type Event = VideoViewEvent;

    fn to_item_events(event: &Self::Event, f: &mut dyn FnMut(workspace::item::ItemEvent)) {
        match event {
            VideoViewEvent::TitleChanged => {
                f(workspace::item::ItemEvent::UpdateTab);
                f(workspace::item::ItemEvent::UpdateBreadcrumbs);
            }
        }
    }

    fn for_each_project_item(
        &self,
        cx: &App,
        f: &mut dyn FnMut(gpui::EntityId, &dyn project::ProjectItem),
    ) {
        f(self.video_item.entity_id(), self.video_item.read(cx));
    }

    fn tab_tooltip_text(&self, cx: &App) -> Option<SharedString> {
        let file_path = self
            .video_item
            .read(cx)
            .abs_path()
            .compact()
            .to_string_lossy()
            .into_owned();
        Some(file_path.into())
    }

    fn tab_content(&self, params: TabContentParams, _window: &Window, cx: &App) -> AnyElement {
        let project_path = self.video_item.read(cx).project_path().clone();

        let label_color = if ItemSettings::get_global(cx).git_status {
            let git_status = self
                .project
                .read(cx)
                .git_store()
                .read(cx)
                .display_status_for_project_path(&project_path, cx)
                .map(|status| status.summary())
                .unwrap_or_default();

            self.project
                .read(cx)
                .entry_for_path(&project_path, cx)
                .map(|entry| {
                    entry_git_aware_label_color(git_status, entry.is_ignored, params.selected)
                })
                .unwrap_or_else(|| params.text_color())
        } else {
            params.text_color()
        };

        Label::new(self.tab_content_text(params.detail.unwrap_or_default(), cx))
            .single_line()
            .color(label_color)
            .when(params.preview, |this| this.italic())
            .into_any_element()
    }

    fn tab_content_text(&self, _detail: usize, cx: &App) -> SharedString {
        self.video_item.read(cx).file_name.clone()
    }

    fn tab_icon(&self, _window: &Window, cx: &App) -> Option<Icon> {
        let path = self.video_item.read(cx).abs_path();
        ItemSettings::get_global(cx)
            .file_icons
            .then(|| FileIcons::get_icon(path, cx))
            .flatten()
            .map(Icon::from_path)
            .or_else(|| Some(Icon::new(IconName::PlayOutlined)))
    }

    fn breadcrumb_location(&self, cx: &App) -> ToolbarItemLocation {
        if EditorSettings::get_global(cx).toolbar.breadcrumbs {
            ToolbarItemLocation::PrimaryLeft
        } else {
            ToolbarItemLocation::Hidden
        }
    }

    fn breadcrumbs(&self, cx: &App) -> Option<(Vec<HighlightedText>, Option<Font>)> {
        let text = breadcrumbs_text_for_video(self.project.read(cx), self.video_item.read(cx), cx);
        let font = ThemeSettings::get_global(cx).buffer_font.clone();
        Some((
            vec![HighlightedText {
                text: text.into(),
                highlights: vec![],
            }],
            Some(font),
        ))
    }

    fn can_split(&self) -> bool {
        true
    }

    fn clone_on_split(
        &self,
        _workspace_id: Option<WorkspaceId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Option<Entity<Self>>>
    where
        Self: Sized,
    {
        let video_item = self.video_item.clone();
        let project = self.project.clone();
        Task::ready(Some(
            cx.new(|cx| VideoView::new(video_item, project, false, window, cx)),
        ))
    }

    fn has_deleted_file(&self, cx: &App) -> bool {
        !self.video_item.read(cx).abs_path().exists()
    }

    fn buffer_kind(&self, _cx: &App) -> workspace::item::ItemBufferKind {
        workspace::item::ItemBufferKind::Singleton
    }
}

fn breadcrumbs_text_for_video(project: &Project, video: &VideoItem, cx: &App) -> String {
    let project_path = video.project_path();
    let mut path = project_path.path.to_rel_path_buf();
    if project.visible_worktrees(cx).count() > 1
        && let Some(worktree) = project.worktree_for_id(project_path.worktree_id, cx)
    {
        path = worktree.read(cx).root_name().join(&path);
    }
    if path.is_empty() {
        return video.file_name.to_string();
    }
    path.display(project.path_style(cx)).to_string()
}

impl ProjectItem for VideoView {
    type Item = VideoItem;

    fn for_project_item(
        project: Entity<Project>,
        _pane: Option<&Pane>,
        item: Entity<Self::Item>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        Self::new(item, project, true, window, cx)
    }
}

impl SerializableItem for VideoView {
    fn serialized_item_kind() -> &'static str {
        "VideoView"
    }

    fn deserialize(
        project: Entity<Project>,
        _workspace: WeakEntity<Workspace>,
        workspace_id: WorkspaceId,
        item_id: ItemId,
        window: &mut Window,
        cx: &mut App,
    ) -> Task<anyhow::Result<Entity<Self>>> {
        let db = VideoViewerDb::global(cx);
        window.spawn(cx, async move |cx| {
            let video_path = db
                .get_video_path(item_id, workspace_id)?
                .context("No video path found")?;

            let (worktree, relative_path) = project
                .update(cx, |project, cx| {
                    project.find_or_create_worktree(video_path.clone(), false, cx)
                })
                .await
                .context("Path not found")?;
            let worktree_id = worktree.update(cx, |worktree, _cx| worktree.id());

            let project_path = ProjectPath {
                worktree_id,
                path: relative_path,
            };

            let video_item = cx
                .update(|_, cx| {
                    <VideoItem as project::ProjectItem>::try_open(&project, &project_path, cx)
                })?
                .context("Video path is not a supported local video file")?
                .await?;

            cx.update(|window, cx| {
                // Restored tabs start paused on frame 0 so restarting Zed does not autoplay audio.
                Ok(cx.new(|cx| VideoView::new(video_item, project, false, window, cx)))
            })?
        })
    }

    fn cleanup(
        workspace_id: WorkspaceId,
        alive_items: Vec<ItemId>,
        _window: &mut Window,
        cx: &mut App,
    ) -> Task<anyhow::Result<()>> {
        let db = VideoViewerDb::global(cx);
        delete_unloaded_items(alive_items, workspace_id, "video_viewers", &db, cx)
    }

    fn serialize(
        &mut self,
        workspace: &mut Workspace,
        item_id: ItemId,
        _closing: bool,
        cx: &mut Context<Self>,
    ) -> Option<Task<anyhow::Result<()>>> {
        let workspace_id = workspace.database_id()?;
        let video_path = self.abs_path(cx);
        let db = VideoViewerDb::global(cx);
        Some(cx.background_spawn(async move {
            log::debug!("Saving video at path {video_path:?}");
            db.save_video_path(item_id, workspace_id, video_path).await
        }))
    }

    fn should_serialize(&self, _event: &Self::Event) -> bool {
        false
    }
}

pub fn init(cx: &mut App) {
    workspace::register_project_item::<VideoView>(cx);
    workspace::register_serializable_item::<VideoView>(cx);
}

mod persistence {
    use std::path::PathBuf;

    use db::{
        query,
        sqlez::{domain::Domain, thread_safe_connection::ThreadSafeConnection},
        sqlez_macros::sql,
    };
    use workspace::{ItemId, WorkspaceDb, WorkspaceId};

    pub struct VideoViewerDb(ThreadSafeConnection);

    impl Domain for VideoViewerDb {
        const NAME: &str = stringify!(VideoViewerDb);

        const MIGRATIONS: &[&str] = &[sql!(
            CREATE TABLE video_viewers (
                workspace_id INTEGER,
                item_id INTEGER UNIQUE,

                video_path BLOB,

                PRIMARY KEY(workspace_id, item_id),
                FOREIGN KEY(workspace_id) REFERENCES workspaces(workspace_id)
                ON DELETE CASCADE
            ) STRICT;
        )];
    }

    db::static_connection!(VideoViewerDb, [WorkspaceDb]);

    impl VideoViewerDb {
        query! {
            pub async fn save_video_path(
                item_id: ItemId,
                workspace_id: WorkspaceId,
                video_path: PathBuf
            ) -> Result<()> {
                INSERT OR REPLACE INTO video_viewers(item_id, workspace_id, video_path)
                VALUES (?, ?, ?)
            }
        }

        query! {
            pub fn get_video_path(item_id: ItemId, workspace_id: WorkspaceId) -> Result<Option<PathBuf>> {
                SELECT video_path
                FROM video_viewers
                WHERE item_id = ? AND workspace_id = ?
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playback::{
        build_atempo_filter, compute_decode_dimensions, compute_display_dimensions,
        parse_duration_seconds, parse_ffprobe_json, parse_frame_rate, parse_hms_duration,
    };
    use fs::{FakeFs, Fs as _};
    use gpui::TestAppContext;
    use settings::SettingsStore;
    use util::rel_path::rel_path;

    fn init_test(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
    }

    #[test]
    fn test_is_video_path_extensions() {
        assert!(is_video_path(Path::new("/tmp/demo.mp4")));
        assert!(is_video_path(Path::new("/tmp/DEMO.MP4")));
        assert!(is_video_path(Path::new("/tmp/clip.mov")));
        assert!(is_video_path(Path::new("/tmp/recording.webm")));
        assert!(is_video_path(Path::new("/tmp/movie.mkv")));
        assert!(!is_video_path(Path::new("/tmp/photo.png")));
        assert!(!is_video_path(Path::new("/tmp/main.rs")));
        assert!(!is_video_path(Path::new("/tmp/no_extension")));
    }

    #[test]
    fn test_parse_ffprobe_json_handles_rotation_sar_and_attached_pic() {
        let json = br#"{
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "mjpeg",
                    "width": 500,
                    "height": 500,
                    "disposition": { "attached_pic": 1 }
                },
                {
                    "index": 1,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "sample_aspect_ratio": "1:1",
                    "avg_frame_rate": "30000/1001",
                    "r_frame_rate": "30/1",
                    "side_data_list": [{ "rotation": -90 }]
                },
                {
                    "index": 2,
                    "codec_type": "audio",
                    "codec_name": "aac"
                }
            ],
            "format": {
                "duration": "12.500000",
                "size": "4096000"
            }
        }"#;

        let metadata = parse_ffprobe_json(json, 123).unwrap();
        assert_eq!(metadata.video_stream_index, Some(1));
        assert_eq!(metadata.audio_stream_index, Some(2));
        assert_eq!(metadata.display_width, 1080);
        assert_eq!(metadata.display_height, 1920);
        assert_eq!(metadata.decode_width, 1080);
        assert_eq!(metadata.decode_height, 1920);
        assert!((metadata.fps - 29.97).abs() < 0.01);
        assert_eq!(metadata.duration, Some(Duration::from_millis(12_500)));
        assert_eq!(metadata.video_codec.as_deref(), Some("h264"));
        assert_eq!(metadata.audio_codec.as_deref(), Some("aac"));
        assert_eq!(metadata.file_size, 4_096_000);
    }

    #[test]
    fn test_dimension_and_helper_calculations() {
        assert_eq!(
            compute_display_dimensions(720, 480, Some("4:3"), 0),
            (960, 480)
        );
        assert_eq!(
            compute_display_dimensions(1920, 1080, Some("1:1"), 90),
            (1080, 1920)
        );
        assert_eq!(compute_decode_dimensions(3840, 2160, 1920), (1920, 1080));
        assert_eq!(compute_decode_dimensions(1280, 720, 1920), (1280, 720));

        assert_eq!(
            parse_frame_rate("60000/1001").map(|v| v.round()),
            Some(60.0)
        );
        assert_eq!(parse_frame_rate("0/0"), None);
        assert_eq!(
            parse_duration_seconds("65.5"),
            Some(Duration::from_millis(65_500))
        );
        assert_eq!(
            parse_hms_duration("01:02:03.500"),
            Some(Duration::from_millis(3_723_500))
        );

        assert_eq!(build_atempo_filter(1.0), None);
        assert_eq!(build_atempo_filter(1.5).as_deref(), Some("atempo=1.5000"));
        assert_eq!(
            build_atempo_filter(0.25).as_deref(),
            Some("atempo=0.5,atempo=0.5000")
        );

        assert_eq!(format_timestamp(Duration::from_secs(75)), "01:15");
        assert_eq!(format_timestamp(Duration::from_secs(3665)), "1:01:05");
        assert_eq!(format_speed(1.0), "1×");
        assert_eq!(format_speed(1.25), "1.25×");
        assert_eq!(format_speed(1.5), "1.5×");
        assert_eq!(next_playback_speed(1.0), 1.25);
        assert_eq!(next_playback_speed(2.0), 0.25);
        assert_eq!(step_playback_speed(1.0, true), 1.25);
        assert_eq!(step_playback_speed(1.0, false), 0.75);
    }

    #[gpui::test]
    async fn test_real_ffmpeg_probe_and_decode_generated_mp4(cx: &mut TestAppContext) {
        cx.background_executor.allow_parking();
        let temp_dir =
            std::env::temp_dir().join(format!("zed-video-viewer-test-{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let mp4_path = temp_dir.join("sample.mp4");

        let generate = util::command::new_command("ffmpeg")
            .args([
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=0.5:size=160x120:rate=10",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=0.5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
            ])
            .arg(&mp4_path)
            .output()
            .await;

        let Ok(output) = generate else {
            let _ = std::fs::remove_dir_all(&temp_dir);
            return;
        };
        if !output.status.success() {
            let _ = std::fs::remove_dir_all(&temp_dir);
            return;
        }

        let metadata = probe_video(&mp4_path)
            .await
            .expect("probe_video should succeed");
        assert_eq!(metadata.display_width, 160);
        assert_eq!(metadata.display_height, 120);
        assert_eq!(metadata.decode_width, 160);
        assert_eq!(metadata.decode_height, 120);
        assert!(metadata.has_video());
        assert!(metadata.has_audio());
        assert!(
            metadata
                .duration
                .is_some_and(|duration| duration >= Duration::from_millis(400))
        );

        let frame = decode_single_frame(&mp4_path, &metadata, Duration::ZERO)
            .await
            .expect("decode_single_frame should succeed")
            .expect("should return frame 0");
        assert_eq!(frame.width, 160);
        assert_eq!(frame.height, 120);
        assert_eq!(frame.bgra_bytes.len(), 160 * 120 * 4);

        // Verify streaming playback session produces frames and shuts down cleanly.
        let mut metadata_no_audio = metadata.clone();
        metadata_no_audio.audio_stream_index = None;
        let mut session = PlaybackSession::start(
            &mp4_path,
            &metadata_no_audio,
            Duration::ZERO,
            1.0,
            1.0,
            true,
            None,
        )
        .expect("PlaybackSession should start");

        let mut received_frame = false;
        for _ in 0..50 {
            match session.take_due_frame() {
                FramePollResult::Frame(decoded) => {
                    assert_eq!(decoded.width, 160);
                    assert_eq!(decoded.height, 120);
                    received_frame = true;
                    break;
                }
                FramePollResult::Waiting => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                FramePollResult::Ended => break,
                FramePollResult::Error(message) => {
                    panic!("unexpected video decoder error: {message}");
                }
            }
        }
        session.stop();
        assert!(
            received_frame,
            "PlaybackSession should deliver at least one decoded frame"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[gpui::test]
    async fn test_video_item_try_open_matches_mp4_and_rejects_non_video(cx: &mut TestAppContext) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.create_dir(Path::new("/root")).await.unwrap();
        fs.insert_file("/root/clip.mp4", vec![0u8; 16]).await;
        fs.insert_file("/root/notes.txt", b"hello".to_vec()).await;

        let project = Project::test(fs, [Path::new("/root")], cx).await;
        let worktree_id =
            cx.update(|cx| project.read(cx).worktrees(cx).next().unwrap().read(cx).id());

        let video_path = ProjectPath {
            worktree_id,
            path: rel_path("clip.mp4").into(),
        };
        let text_path = ProjectPath {
            worktree_id,
            path: rel_path("notes.txt").into(),
        };

        let text_open =
            cx.update(|cx| <VideoItem as project::ProjectItem>::try_open(&project, &text_path, cx));
        assert!(text_open.is_none());

        let video_open = cx
            .update(|cx| <VideoItem as project::ProjectItem>::try_open(&project, &video_path, cx))
            .expect("mp4 should match VideoItem::try_open")
            .await
            .expect("VideoItem should open");

        video_open.read_with(cx, |item, _| {
            assert_eq!(item.file_name.as_ref(), "clip.mp4");
            assert_eq!(item.abs_path(), Path::new("/root/clip.mp4"));
        });
    }

    #[gpui::test]
    async fn test_video_view_renders_and_controls_playback_in_window(cx: &mut TestAppContext) {
        cx.background_executor.allow_parking();
        init_test(cx);
        cx.update(|cx| {
            audio::init(cx);
        });

        let temp_dir = std::env::temp_dir().join(format!(
            "zed-video-viewer-window-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let mp4_path = temp_dir.join("sample.mp4");

        let generate = util::command::new_command("ffmpeg")
            .args([
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=0.4:size=160x120:rate=10",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&mp4_path)
            .output()
            .await;

        let Ok(output) = generate else {
            let _ = std::fs::remove_dir_all(&temp_dir);
            return;
        };
        if !output.status.success() {
            let _ = std::fs::remove_dir_all(&temp_dir);
            return;
        }

        let fs = FakeFs::new(cx.executor());
        fs.create_dir(Path::new("/root")).await.unwrap();
        let project = Project::test(fs, [Path::new("/root")], cx).await;
        let worktree_id =
            cx.update(|cx| project.read(cx).worktrees(cx).next().unwrap().read(cx).id());

        let video_item = cx.new(|_| VideoItem {
            project_path: ProjectPath {
                worktree_id,
                path: rel_path("sample.mp4").into(),
            },
            abs_path: mp4_path.clone(),
            entry_id: None,
            file_name: "sample.mp4".into(),
        });

        let (view, cx) = cx.add_window_view(|window, cx| {
            VideoView::new(video_item, project.clone(), true, window, cx)
        });

        let load_task = view.update(cx, |this, _| this.load_task.take());
        if let Some(load_task) = load_task {
            load_task.await;
        }
        cx.run_until_parked();

        view.read_with(cx, |this, _| {
            assert!(
                this.load_error.is_none(),
                "unexpected error: {:?}",
                this.load_error
            );
            assert!(this.metadata.is_some());
            assert!(this.is_playing);
        });

        // Simulate frames and exercise controls in a real window context.
        cx.update(|window, cx| {
            window.simulate_next_frame(cx);
        });
        cx.run_until_parked();

        view.update_in(cx, |this, window, cx| {
            this.toggle_playback(&TogglePlayback, window, cx);
            assert!(!this.is_playing);
            this.toggle_playback(&TogglePlayback, window, cx);
            assert!(this.is_playing);
            this.toggle_loop(&ToggleLoop, window, cx);
            assert!(this.is_looping);
            this.toggle_mute(&ToggleMute, window, cx);
            assert!(this.is_muted);
            this.cycle_speed(&CycleSpeed, window, cx);
            assert!((this.playback_rate - 1.25).abs() < 1e-3);
            this.seek_to_start(&SeekToStart, window, cx);
            this.stop_playback();
        });

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
