use gpui::{App, Context, Entity, IntoElement, ParentElement, Render, Subscription, div};
use ui::prelude::*;
use util::size::format_file_size;
use workspace::{HideStatusItem, ItemHandle, StatusItemView, Workspace};

use crate::{VideoView, playback::VideoMetadata};

pub struct VideoInfo {
    metadata: Option<VideoMetadata>,
    _observe_active_video: Option<Subscription>,
}

impl VideoInfo {
    pub fn new(_workspace: &Workspace) -> Self {
        Self {
            metadata: None,
            _observe_active_video: None,
        }
    }

    fn update_metadata(&mut self, video_view: &Entity<VideoView>, cx: &mut Context<Self>) {
        self.metadata = video_view.read(cx).metadata().cloned();
        cx.notify();
    }
}

impl Render for VideoInfo {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let Some(metadata) = self.metadata.as_ref() else {
            return div().hidden();
        };

        let mut parts = Vec::new();
        if metadata.has_video() {
            parts.push(format!(
                "{}×{}",
                metadata.display_width, metadata.display_height
            ));
            let fps_label = if (metadata.fps - metadata.fps.round()).abs() < 0.05 {
                format!("{} fps", metadata.fps.round() as u32)
            } else {
                format!("{:.2} fps", metadata.fps)
            };
            parts.push(fps_label);
        }

        if let Some(video_codec) = &metadata.video_codec {
            if let Some(audio_codec) = &metadata.audio_codec {
                parts.push(format!("{video_codec} / {audio_codec}"));
            } else {
                parts.push(video_codec.clone());
            }
        } else if let Some(audio_codec) = &metadata.audio_codec {
            parts.push(audio_codec.clone());
        }

        if metadata.file_size > 0 {
            parts.push(format_file_size(metadata.file_size, false));
        }

        div().child(Label::new(parts.join(" • ")).size(LabelSize::Small))
    }
}

impl StatusItemView for VideoInfo {
    fn set_active_pane_item(
        &mut self,
        active_pane_item: Option<&dyn ItemHandle>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self._observe_active_video = None;

        if let Some(video_view) = active_pane_item.and_then(|item| item.act_as::<VideoView>(cx)) {
            self.update_metadata(&video_view, cx);
            self._observe_active_video = Some(cx.observe(&video_view, |this, view, cx| {
                this.update_metadata(&view, cx);
            }));
        } else {
            self.metadata = None;
            cx.notify();
        }
    }

    fn hide_setting(&self, _: &App) -> Option<HideStatusItem> {
        None
    }
}
