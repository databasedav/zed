use std::cell::Cell;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use collections::{FxHasher, HashMap};
use gpui::{
    App, AvailableSpace, Background, Bounds, ContentMask, Corners, Element, ElementId, FillOptions,
    FillRule as GpuiFillRule, GlobalElementId, GlyphId, Hsla, InspectorElementId, LayoutId,
    LineCap as GpuiLineCap, LineJoin as GpuiLineJoin, PathBuilder, PathStyle, Pixels, Point,
    RenderImage, SharedString, Size, StrokeOptions, Style, TransformationMatrix, Window, point, px,
    quad, size,
};
use image::{Frame as ImageFrame, ImageBuffer, Rgba};
use parking_lot::Mutex;
use smallvec::SmallVec;
use typst::WorldExt;
use typst::layout::{Abs, Frame, FrameItem, GroupItem, Point as TypstPoint, Ratio, Transform};
use typst::model::Destination;
use typst::text::TextItem;
use typst::visualize::{
    Color, Curve, CurveItem, FillRule, FixedStroke, Geometry, Gradient, Image as TypstImage,
    ImageKind, LineCap, LineJoin, Paint, RatioOrAngle, Shape,
};
use typst_layout::PagedDocument;
use util::ResultExt;

use crate::preprocess::{
    OffsetMap, TypstInlineCodeSpan, escape_typst_plain_text, extract_inline_code_spans,
    preprocess_typst_segment,
};
use crate::world::{TypstFontCatalog, TypstRenderStyle, ZedTypstWorld, resolve_gpui_font_id};

const INTER_PAGE_GAP_PT: f64 = 8.0;

#[derive(Clone, Debug)]
pub struct TypstGlyphCluster {
    pub local_bounds: Bounds<Pixels>,
    pub rendered_range: Range<usize>,
    pub source_range: Range<usize>,
}

#[derive(Clone, Debug)]
pub struct TypstLinkRegion {
    pub local_bounds: Bounds<Pixels>,
    pub destination_url: SharedString,
    pub source_range: Option<Range<usize>>,
}

pub struct CompiledTypstLayout {
    pages: Vec<(TypstPoint, Frame)>,
    size: Size<Pixels>,
    baseline: Pixels,
    rendered_text: SharedString,
    clusters: Vec<TypstGlyphCluster>,
    links: Vec<TypstLinkRegion>,
    source_range: Range<usize>,
    decoded_images: Mutex<HashMap<u64, Arc<RenderImage>>>,
}

impl CompiledTypstLayout {
    pub fn size(&self) -> Size<Pixels> {
        self.size
    }

    pub fn baseline(&self) -> Pixels {
        self.baseline
    }

    pub fn rendered_text(&self) -> &SharedString {
        &self.rendered_text
    }

    pub fn clusters(&self) -> &[TypstGlyphCluster] {
        &self.clusters
    }

    pub fn links(&self) -> &[TypstLinkRegion] {
        &self.links
    }

    pub fn source_range(&self) -> Range<usize> {
        self.source_range.clone()
    }

    pub fn link_for_local_position(
        &self,
        local_position: Point<Pixels>,
    ) -> Option<&TypstLinkRegion> {
        self.links
            .iter()
            .rev()
            .find(|link| link.local_bounds.contains(&local_position))
    }

    pub fn source_index_for_local_position(
        &self,
        local_position: Point<Pixels>,
    ) -> Result<usize, usize> {
        if self.clusters.is_empty() {
            return Err(self.source_range.end);
        }

        for cluster in &self.clusters {
            if cluster.local_bounds.contains(&local_position) {
                let midpoint_x =
                    cluster.local_bounds.origin.x + cluster.local_bounds.size.width * 0.5;
                if local_position.x >= midpoint_x {
                    return Ok(cluster.source_range.end);
                } else {
                    return Ok(cluster.source_range.start);
                }
            }
        }

        let mut best_same_row: Option<(Pixels, usize)> = None;
        let mut best_above: Option<(Pixels, usize)> = None;

        for cluster in &self.clusters {
            let top = cluster.local_bounds.top();
            let bottom = cluster.local_bounds.bottom();
            if local_position.y >= top && local_position.y <= bottom {
                let dx = if local_position.x < cluster.local_bounds.left() {
                    cluster.local_bounds.left() - local_position.x
                } else {
                    local_position.x - cluster.local_bounds.right()
                };
                let candidate_index = if local_position.x < cluster.local_bounds.left() {
                    cluster.source_range.start
                } else {
                    cluster.source_range.end
                };
                if best_same_row.is_none_or(|(best_dx, _)| dx < best_dx) {
                    best_same_row = Some((dx, candidate_index));
                }
            } else if local_position.y > bottom {
                let dy = local_position.y - bottom;
                if best_above.is_none_or(|(best_dy, _)| dy <= best_dy) {
                    best_above = Some((dy, cluster.source_range.end));
                }
            }
        }

        if let Some((_, index)) = best_same_row {
            return Err(index);
        }
        if let Some((_, index)) = best_above {
            return Err(index);
        }
        Err(self.source_range.start)
    }

    pub fn local_bounds_for_source_range(&self, range: Range<usize>) -> Vec<Bounds<Pixels>> {
        if range.start >= range.end {
            return Vec::new();
        }
        let mut merged: Vec<Bounds<Pixels>> = Vec::new();
        for cluster in &self.clusters {
            if cluster.source_range.end <= range.start || cluster.source_range.start >= range.end {
                continue;
            }
            let bounds = cluster.local_bounds;
            if let Some(last) = merged.last_mut() {
                let same_line = (last.top() - bounds.top()).abs() <= px(3.0)
                    && (last.bottom() - bounds.bottom()).abs() <= px(3.0);
                let adjacent = (bounds.left() - last.right()).abs() <= px(8.0);
                if same_line && adjacent {
                    let left = last.left().min(bounds.left());
                    let top = last.top().min(bounds.top());
                    let right = last.right().max(bounds.right());
                    let bottom = last.bottom().max(bounds.bottom());
                    *last = Bounds::from_corners(point(left, top), point(right, bottom));
                    continue;
                }
            }
            merged.push(bounds);
        }
        merged
    }

    pub fn local_position_for_source_index(
        &self,
        source_index: usize,
    ) -> Option<(Point<Pixels>, Pixels)> {
        if source_index < self.source_range.start || source_index > self.source_range.end {
            return None;
        }
        for cluster in &self.clusters {
            if source_index <= cluster.source_range.end {
                let x = if source_index <= cluster.source_range.start {
                    cluster.local_bounds.left()
                } else {
                    cluster.local_bounds.right()
                };
                return Some((
                    point(x, cluster.local_bounds.top()),
                    cluster.local_bounds.size.height,
                ));
            }
        }
        if let Some(last) = self.clusters.last() {
            return Some((
                point(last.local_bounds.right(), last.local_bounds.top()),
                last.local_bounds.size.height,
            ));
        }
        Some((Point::default(), self.size.height.max(px(16.0))))
    }

    pub fn text_for_source_range(&self, range: Range<usize>) -> String {
        if range.start >= range.end || self.clusters.is_empty() {
            return String::new();
        }
        let mut min_rendered = usize::MAX;
        let mut max_rendered = 0usize;
        for cluster in &self.clusters {
            if cluster.source_range.end > range.start && cluster.source_range.start < range.end {
                min_rendered = min_rendered.min(cluster.rendered_range.start);
                max_rendered = max_rendered.max(cluster.rendered_range.end);
            }
        }
        if min_rendered >= max_rendered {
            return String::new();
        }
        self.rendered_text
            .get(min_rendered..max_rendered)
            .unwrap_or("")
            .to_string()
    }
}

pub struct PreparedTypstSegment {
    normalized_typst: SharedString,
    raw_fallback_text: SharedString,
    offset_map: Arc<OffsetMap>,
    source_range: Range<usize>,
    inline_code_spans: Vec<TypstInlineCodeSpan>,
    font_catalog: Arc<TypstFontCatalog>,
    cached_layouts: Mutex<SmallVec<[(TypstRenderStyle, Arc<CompiledTypstLayout>); 2]>>,
    linked_variant: Mutex<Option<(Vec<(Range<usize>, SharedString)>, Arc<PreparedTypstSegment>)>>,
}

impl std::fmt::Debug for PreparedTypstSegment {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedTypstSegment")
            .field("source_range", &self.source_range)
            .field("normalized_typst", &self.normalized_typst)
            .finish()
    }
}

impl PreparedTypstSegment {
    pub fn new(
        raw_segment_text: &str,
        source_range: Range<usize>,
        inline_code_links: &HashMap<Range<usize>, SharedString>,
        font_catalog: Arc<TypstFontCatalog>,
    ) -> Self {
        let inline_code_spans = extract_inline_code_spans(raw_segment_text, source_range.start);
        let (normalized, offset_map) =
            preprocess_typst_segment(raw_segment_text, source_range.start, inline_code_links);
        Self {
            normalized_typst: SharedString::from(normalized),
            raw_fallback_text: SharedString::from(raw_segment_text.to_string()),
            offset_map: Arc::new(offset_map),
            source_range,
            inline_code_spans,
            font_catalog,
            cached_layouts: Mutex::new(SmallVec::new()),
            linked_variant: Mutex::new(None),
        }
    }

    pub fn from_exact_typst(
        typst_code: String,
        raw_fallback_text: String,
        source_range: Range<usize>,
        font_catalog: Arc<TypstFontCatalog>,
    ) -> Self {
        let code_len = typst_code.len();
        let mut fallback_map = OffsetMap::identity(source_range.start, 0);
        if code_len > 0 && !source_range.is_empty() {
            fallback_map = OffsetMap::identity(source_range.start, source_range.len());
        }
        Self {
            normalized_typst: SharedString::from(typst_code),
            raw_fallback_text: SharedString::from(raw_fallback_text),
            offset_map: Arc::new(fallback_map),
            source_range,
            inline_code_spans: Vec::new(),
            font_catalog,
            cached_layouts: Mutex::new(SmallVec::new()),
            linked_variant: Mutex::new(None),
        }
    }

    pub fn inline_code_spans(&self) -> &[TypstInlineCodeSpan] {
        &self.inline_code_spans
    }

    pub fn with_inline_code_links(
        self: &Arc<Self>,
        inline_code_links: &HashMap<Range<usize>, SharedString>,
    ) -> Arc<Self> {
        if inline_code_links.is_empty() {
            return self.clone();
        }
        let mut sorted_key: Vec<(Range<usize>, SharedString)> = inline_code_links
            .iter()
            .map(|(range, url)| (range.clone(), url.clone()))
            .collect();
        sorted_key.sort_by_key(|(range, _)| (range.start, range.end));

        {
            let guard = self.linked_variant.lock();
            if let Some((cached_key, cached_segment)) = guard.as_ref()
                && *cached_key == sorted_key
            {
                return cached_segment.clone();
            }
        }

        let built = Arc::new(Self::new(
            &self.raw_fallback_text,
            self.source_range.clone(),
            inline_code_links,
            self.font_catalog.clone(),
        ));
        *self.linked_variant.lock() = Some((sorted_key, built.clone()));
        built
    }

    pub fn source_range(&self) -> Range<usize> {
        self.source_range.clone()
    }

    pub fn normalized_typst(&self) -> &str {
        self.normalized_typst.as_ref()
    }

    pub fn layout_for_style(&self, style: &TypstRenderStyle) -> Arc<CompiledTypstLayout> {
        let quantized_style = quantize_style(style);
        {
            let cache = self.cached_layouts.lock();
            if let Some((_, existing)) = cache.iter().find(|(key, _)| *key == quantized_style) {
                return existing.clone();
            }
        }

        let compiled = Arc::new(compile_typst_with_fallback(
            &self.normalized_typst,
            &self.raw_fallback_text,
            &self.offset_map,
            self.source_range.clone(),
            &quantized_style,
            self.font_catalog.clone(),
        ));

        let mut cache = self.cached_layouts.lock();
        if cache.len() >= 2 {
            cache.remove(0);
        }
        cache.push((quantized_style, compiled.clone()));
        compiled
    }
}

fn quantize_style(style: &TypstRenderStyle) -> TypstRenderStyle {
    let quantize_quarter = |value: f32| (value * 4.0).round() / 4.0;
    let mut quantized = style.clone();
    quantized.page_width_pt = style.page_width_pt.map(|width| (width * 2.0).round() / 2.0);
    quantized.base_font_size_pt = quantize_quarter(style.base_font_size_pt);
    quantized.code_font_size_pt = quantize_quarter(style.code_font_size_pt);
    quantized.line_height_pt = quantize_quarter(style.line_height_pt);
    quantized.paragraph_spacing_pt = quantize_quarter(style.paragraph_spacing_pt);
    quantized.heading_font_sizes_pt = style.heading_font_sizes_pt.map(quantize_quarter);
    quantized
}

fn compile_typst_with_fallback(
    normalized_typst: &str,
    raw_fallback_text: &str,
    offset_map: &OffsetMap,
    source_range: Range<usize>,
    style: &TypstRenderStyle,
    font_catalog: Arc<TypstFontCatalog>,
) -> CompiledTypstLayout {
    let preamble = style.build_preamble();
    if let Some(compiled) = try_compile_with_preamble(
        &preamble,
        normalized_typst,
        raw_fallback_text,
        offset_map,
        source_range.clone(),
        font_catalog.clone(),
        style.user_preamble.is_some(),
    ) {
        return compiled;
    }

    if style.user_preamble.is_some() {
        let builtin_preamble = style.build_builtin_preamble();
        if let Some(compiled) = try_compile_with_preamble(
            &builtin_preamble,
            normalized_typst,
            raw_fallback_text,
            offset_map,
            source_range.clone(),
            font_catalog,
            false,
        ) {
            return compiled;
        }
    }

    CompiledTypstLayout {
        pages: Vec::new(),
        size: Size::default(),
        baseline: Pixels::ZERO,
        rendered_text: SharedString::default(),
        clusters: Vec::new(),
        links: Vec::new(),
        source_range,
        decoded_images: Mutex::default(),
    }
}

fn try_compile_with_preamble(
    preamble: &str,
    normalized_typst: &str,
    raw_fallback_text: &str,
    offset_map: &OffsetMap,
    source_range: Range<usize>,
    font_catalog: Arc<TypstFontCatalog>,
    abort_on_preamble_error: bool,
) -> Option<CompiledTypstLayout> {
    if let Ok(world) = ZedTypstWorld::new(preamble, normalized_typst, font_catalog.clone()) {
        let warned = typst::compile::<PagedDocument>(&world);
        match warned.output {
            Ok(document) => {
                typst::comemo::evict(30);
                return Some(build_compiled_layout(
                    document,
                    &world,
                    offset_map,
                    source_range,
                ));
            }
            Err(diagnostics) => {
                if let Some(first_diag) = diagnostics.first() {
                    if let Some(diag_range) = world.range(first_diag.span) {
                        if abort_on_preamble_error && diag_range.start < world.preamble_byte_len() {
                            log::warn!(
                                "Error in user Typst config (typst.typ); falling back to built-in preamble: {}",
                                first_diag.message
                            );
                            return None;
                        }
                        let body_err_start =
                            diag_range.start.saturating_sub(world.preamble_byte_len());
                        if let Some(split_newline) = normalized_typst
                            .get(..body_err_start)
                            .and_then(|prefix| prefix.rfind('\n'))
                        {
                            let valid_prefix = &normalized_typst[..split_newline];
                            let unparsed_tail = &normalized_typst[split_newline..];
                            let combined = format!(
                                "{valid_prefix}\n{}",
                                escape_typst_plain_text(unparsed_tail)
                            );
                            if let Ok(split_world) =
                                ZedTypstWorld::new(preamble, &combined, font_catalog.clone())
                            {
                                if let Ok(document) =
                                    typst::compile::<PagedDocument>(&split_world).output
                                {
                                    typst::comemo::evict(30);
                                    return Some(build_compiled_layout(
                                        document,
                                        &split_world,
                                        offset_map,
                                        source_range,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let escaped_plain = escape_typst_plain_text(raw_fallback_text);
    if let Ok(plain_world) = ZedTypstWorld::new(preamble, &escaped_plain, font_catalog) {
        if let Ok(document) = typst::compile::<PagedDocument>(&plain_world).output {
            typst::comemo::evict(30);
            let identity_map = OffsetMap::identity(source_range.start, raw_fallback_text.len());
            return Some(build_compiled_layout(
                document,
                &plain_world,
                &identity_map,
                source_range,
            ));
        }
    }

    None
}

fn build_compiled_layout(
    document: PagedDocument,
    world: &ZedTypstWorld,
    offset_map: &OffsetMap,
    source_range: Range<usize>,
) -> CompiledTypstLayout {
    let mut pages = Vec::with_capacity(document.pages().len());
    let mut max_width_pt = 0.0_f64;
    let mut cursor_y_pt = 0.0_f64;

    for (page_index, page) in document.pages().iter().enumerate() {
        if page_index > 0 {
            cursor_y_pt += INTER_PAGE_GAP_PT;
        }
        let frame_size = page.frame.size();
        max_width_pt = max_width_pt.max(frame_size.x.to_pt());
        pages.push((
            TypstPoint::new(Abs::zero(), Abs::pt(cursor_y_pt)),
            page.frame.clone(),
        ));
        cursor_y_pt += frame_size.y.to_pt();
    }

    let first_baseline_pt = document
        .pages()
        .first()
        .map(|page| find_first_baseline_pt(&page.frame))
        .unwrap_or(cursor_y_pt);

    let mut extractor = FrameMetadataExtractor {
        world,
        offset_map,
        source_range: source_range.clone(),
        rendered_text: String::new(),
        clusters: Vec::new(),
        links: Vec::new(),
        high_water_source: source_range.start,
        last_baseline_y: None,
        last_right_x: None,
    };

    for (page_offset, frame) in &pages {
        let initial_transform = Transform::translate(page_offset.x, page_offset.y);
        extractor.visit_frame(frame, initial_transform);
    }

    for link in &mut extractor.links {
        let mut min_source = usize::MAX;
        let mut max_source = 0usize;
        for cluster in &extractor.clusters {
            let center = point(
                cluster.local_bounds.origin.x + cluster.local_bounds.size.width * 0.5,
                cluster.local_bounds.origin.y + cluster.local_bounds.size.height * 0.5,
            );
            if link.local_bounds.contains(&center)
                && cluster.source_range.start < cluster.source_range.end
            {
                min_source = min_source.min(cluster.source_range.start);
                max_source = max_source.max(cluster.source_range.end);
            }
        }
        if min_source < max_source {
            link.source_range = Some(min_source..max_source);
        }
    }

    CompiledTypstLayout {
        pages,
        size: size(px(max_width_pt as f32), px(cursor_y_pt as f32)),
        baseline: px(first_baseline_pt as f32),
        rendered_text: SharedString::from(extractor.rendered_text),
        clusters: extractor.clusters,
        links: extractor.links,
        source_range,
        decoded_images: Mutex::default(),
    }
}

fn find_first_baseline_pt(frame: &Frame) -> f64 {
    if frame.has_baseline() {
        return frame.baseline().to_pt();
    }
    for (pos, item) in frame.items() {
        match item {
            FrameItem::Text(_) => return pos.y.to_pt(),
            FrameItem::Group(group) => {
                if let Some(sub_y) = find_first_text_y_in_frame(&group.frame) {
                    let (_, transformed_y) = apply_typst_transform(group.transform, 0.0, sub_y);
                    return pos.y.to_pt() + transformed_y;
                }
            }
            _ => {}
        }
    }
    frame.height().to_pt()
}

fn find_first_text_y_in_frame(frame: &Frame) -> Option<f64> {
    for (pos, item) in frame.items() {
        match item {
            FrameItem::Text(_) => return Some(pos.y.to_pt()),
            FrameItem::Group(group) => {
                if let Some(sub_y) = find_first_text_y_in_frame(&group.frame) {
                    let (_, transformed_y) = apply_typst_transform(group.transform, 0.0, sub_y);
                    return Some(pos.y.to_pt() + transformed_y);
                }
            }
            _ => {}
        }
    }
    None
}

struct FrameMetadataExtractor<'a> {
    world: &'a ZedTypstWorld,
    offset_map: &'a OffsetMap,
    source_range: Range<usize>,
    rendered_text: String,
    clusters: Vec<TypstGlyphCluster>,
    links: Vec<TypstLinkRegion>,
    high_water_source: usize,
    last_baseline_y: Option<f32>,
    last_right_x: Option<f32>,
}

impl FrameMetadataExtractor<'_> {
    fn visit_frame(&mut self, frame: &Frame, transform: Transform) {
        for (pos, item) in frame.items() {
            let item_transform = transform.pre_concat(Transform::translate(pos.x, pos.y));
            match item {
                FrameItem::Group(group) => {
                    let group_transform = item_transform.pre_concat(group.transform);
                    self.visit_frame(&group.frame, group_transform);
                }
                FrameItem::Text(text_item) => {
                    self.extract_text_item(text_item, item_transform);
                }
                FrameItem::Link(Destination::Url(url), link_size) => {
                    let (x0, y0) = apply_typst_transform(item_transform, 0.0, 0.0);
                    let (x1, y1) = apply_typst_transform(
                        item_transform,
                        link_size.x.to_pt(),
                        link_size.y.to_pt(),
                    );
                    let left = x0.min(x1) as f32;
                    let top = y0.min(y1) as f32;
                    let right = x0.max(x1) as f32;
                    let bottom = y0.max(y1) as f32;
                    self.links.push(TypstLinkRegion {
                        local_bounds: Bounds::from_corners(
                            point(px(left), px(top)),
                            point(px(right), px(bottom)),
                        ),
                        destination_url: SharedString::from(url.as_str().to_string()),
                        source_range: None,
                    });
                }
                _ => {}
            }
        }
    }

    fn extract_text_item(&mut self, text_item: &TextItem, transform: Transform) {
        if text_item.glyphs.is_empty() {
            return;
        }
        let font_size_pt = text_item.size.to_pt();
        let (_, scale_y) = transform_axis_scales(transform);
        let visual_font_size = (font_size_pt * scale_y.max(0.25)) as f32;

        let (first_world_x, first_world_y) = apply_typst_transform(transform, 0.0, 0.0);
        let first_world_x = first_world_x as f32;
        let first_world_y = first_world_y as f32;

        if let Some(prev_y) = self.last_baseline_y {
            if (first_world_y - prev_y).abs() > visual_font_size * 0.55 {
                if !self.rendered_text.is_empty() && !self.rendered_text.ends_with('\n') {
                    self.rendered_text.push('\n');
                }
            } else if let Some(prev_right_x) = self.last_right_x {
                let gap = first_world_x - prev_right_x;
                let starts_ws = text_item.text.starts_with(char::is_whitespace);
                let ends_ws = self.rendered_text.ends_with(char::is_whitespace);
                if gap > visual_font_size * 0.14 && !starts_ws && !ends_ws {
                    self.rendered_text.push(' ');
                }
            }
        }

        let preamble_len = self.world.preamble_byte_len();
        let mut cursor_x = Abs::zero();
        let mut cursor_y = Abs::zero();

        for glyph in &text_item.glyphs {
            let glyph_text_range = glyph.range();
            let glyph_str = text_item.text.get(glyph_text_range).unwrap_or("");
            let x_start_pt = (cursor_x + glyph.x_offset.at(text_item.size)).to_pt();
            let y_start_pt = -(cursor_y + glyph.y_offset.at(text_item.size)).to_pt();
            let advance_x_pt = glyph.x_advance.at(text_item.size).to_pt();

            let (world_x0, world_y0) = apply_typst_transform(transform, x_start_pt, y_start_pt);
            let (world_x1, _) =
                apply_typst_transform(transform, x_start_pt + advance_x_pt.max(1.0), y_start_pt);

            let left = world_x0.min(world_x1) as f32;
            let right = world_x0.max(world_x1).max(world_x0 + 1.0) as f32;
            let baseline_y = world_y0 as f32;
            let top = baseline_y - visual_font_size * 0.85;
            let bottom = baseline_y + visual_font_size * 0.25;

            let rendered_start = self.rendered_text.len();
            self.rendered_text.push_str(glyph_str);
            let rendered_end = self.rendered_text.len();

            let mapped_source_range = self
                .world
                .range(glyph.span.0)
                .and_then(|main_range| {
                    if main_range.start < preamble_len {
                        return None;
                    }
                    let norm_span_start = main_range.start - preamble_len;
                    let norm_span_end = main_range.end.saturating_sub(preamble_len);
                    let norm_glyph_start =
                        (norm_span_start + usize::from(glyph.span.1)).min(norm_span_end);
                    let norm_glyph_end = (norm_glyph_start + glyph_str.len().max(1))
                        .min(norm_span_end.max(norm_glyph_start));
                    Some(
                        self.offset_map
                            .map_range_to_source(norm_glyph_start..norm_glyph_end),
                    )
                })
                .unwrap_or_else(|| self.high_water_source..self.high_water_source);

            let clamped_start = mapped_source_range
                .start
                .max(self.high_water_source)
                .min(self.source_range.end);
            let clamped_end = mapped_source_range
                .end
                .max(clamped_start)
                .min(self.source_range.end);
            self.high_water_source = clamped_end;

            if !glyph_str.is_empty() {
                self.clusters.push(TypstGlyphCluster {
                    local_bounds: Bounds::from_corners(
                        point(px(left), px(top)),
                        point(px(right), px(bottom)),
                    ),
                    rendered_range: rendered_start..rendered_end,
                    source_range: clamped_start..clamped_end,
                });
            }

            cursor_x += glyph.x_advance.at(text_item.size);
            cursor_y += glyph.y_advance.at(text_item.size);
            self.last_right_x = Some(right);
            self.last_baseline_y = Some(baseline_y);
        }
    }
}

pub struct TypstFrameElement {
    segment: Arc<PreparedTypstSegment>,
    base_style: TypstRenderStyle,
    highlights: Vec<(Range<usize>, Hsla)>,
    painted_origin: Option<Rc<Cell<Option<Point<Pixels>>>>>,
    active_layout_sink: Option<Rc<Cell<Option<Arc<CompiledTypstLayout>>>>>,
}

impl TypstFrameElement {
    pub fn new(segment: Arc<PreparedTypstSegment>, base_style: TypstRenderStyle) -> Self {
        Self {
            segment,
            base_style,
            highlights: Vec::new(),
            painted_origin: None,
            active_layout_sink: None,
        }
    }

    pub fn with_highlights(mut self, highlights: Vec<(Range<usize>, Hsla)>) -> Self {
        self.highlights = highlights;
        self
    }

    pub fn with_tracking(
        mut self,
        painted_origin: Rc<Cell<Option<Point<Pixels>>>>,
        active_layout_sink: Rc<Cell<Option<Arc<CompiledTypstLayout>>>>,
    ) -> Self {
        self.painted_origin = Some(painted_origin);
        self.active_layout_sink = Some(active_layout_sink);
        self
    }
}

impl Element for TypstFrameElement {
    type RequestLayoutState = Option<Arc<CompiledTypstLayout>>;
    type PrepaintState = Arc<CompiledTypstLayout>;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        if self.base_style.tight_bounds {
            let mut auto_style = self.base_style.clone();
            auto_style.page_width_pt = None;
            let compiled = self.segment.layout_for_style(&auto_style);
            let mut element_style = Style::default();
            element_style.size.width = compiled.size.width.into();
            element_style.size.height = compiled.size.height.into();
            let layout_id = window.request_layout(element_style, [], cx);
            return (layout_id, Some(compiled));
        }

        let segment = self.segment.clone();
        let base_style = self.base_style.clone();
        let layout_id = window.request_measured_layout(
            Style::default(),
            move |known_dimensions, available_space, _window, _cx| {
                let mut measure_style = base_style.clone();
                if let Some(known_width) = known_dimensions.width {
                    measure_style.page_width_pt = Some(f32::from(known_width).max(32.0));
                } else {
                    match available_space.width {
                        AvailableSpace::Definite(avail_width) => {
                            measure_style.page_width_pt = Some(f32::from(avail_width).max(32.0));
                        }
                        AvailableSpace::MinContent => {
                            measure_style.page_width_pt = Some(160.0);
                        }
                        AvailableSpace::MaxContent => {
                            measure_style.page_width_pt = None;
                        }
                    }
                }
                let compiled = segment.layout_for_style(&measure_style);
                let width = known_dimensions.width.unwrap_or(compiled.size.width);
                let height = known_dimensions.height.unwrap_or(compiled.size.height);
                size(width, height)
            },
        );

        (layout_id, None)
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Self::PrepaintState {
        let compiled = match request_layout.take() {
            Some(compiled) => compiled,
            None => {
                let mut final_style = self.base_style.clone();
                if bounds.size.width > px(1.0) {
                    final_style.page_width_pt = Some(f32::from(bounds.size.width).max(32.0));
                }
                self.segment.layout_for_style(&final_style)
            }
        };

        if let Some(origin_cell) = &self.painted_origin {
            origin_cell.set(Some(bounds.origin));
        }
        if let Some(layout_sink) = &self.active_layout_sink {
            layout_sink.set(Some(compiled.clone()));
        }

        compiled
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        compiled: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        for (source_range, highlight_color) in &self.highlights {
            for local_rect in compiled.local_bounds_for_source_range(source_range.clone()) {
                let highlight_bounds = Bounds {
                    origin: bounds.origin + local_rect.origin,
                    size: local_rect.size,
                };
                window.paint_quad(quad(
                    highlight_bounds,
                    px(2.0),
                    *highlight_color,
                    gpui::Edges::default(),
                    Hsla::transparent_black(),
                    gpui::BorderStyle::default(),
                ));
            }
        }

        let mut painter = FramePainter {
            element_origin: bounds.origin,
            fallback_color: self.base_style.text_color,
            compiled,
            window,
            cx,
        };

        for (page_offset, frame) in &compiled.pages {
            let page_transform = Transform::translate(page_offset.x, page_offset.y);
            painter.paint_frame(frame, page_transform);
        }
    }
}

struct FramePainter<'a, 'b> {
    element_origin: Point<Pixels>,
    fallback_color: Hsla,
    compiled: &'a CompiledTypstLayout,
    window: &'a mut Window,
    cx: &'b mut App,
}

impl FramePainter<'_, '_> {
    fn paint_frame(&mut self, frame: &Frame, transform: Transform) {
        for (pos, item) in frame.items() {
            let item_transform = transform.pre_concat(Transform::translate(pos.x, pos.y));
            match item {
                FrameItem::Group(group) => self.paint_group(group, item_transform),
                FrameItem::Shape(shape, _) => self.paint_shape(shape, item_transform),
                FrameItem::Image(image, img_size, _) => {
                    self.paint_typst_image(image, *img_size, item_transform);
                }
                FrameItem::Text(text_item) => self.paint_text_item(text_item, item_transform),
                FrameItem::Link(_, _) | FrameItem::Tag(_) => {}
            }
        }
    }

    fn paint_group(&mut self, group: &GroupItem, parent_transform: Transform) {
        let group_transform = parent_transform.pre_concat(group.transform);
        if let Some(clip_curve) = &group.clip {
            let clip_bbox = clip_curve.bbox(None);
            let (x0, y0) = apply_typst_transform(
                group_transform,
                clip_bbox.min.x.to_pt(),
                clip_bbox.min.y.to_pt(),
            );
            let (x1, y1) = apply_typst_transform(
                group_transform,
                clip_bbox.max.x.to_pt(),
                clip_bbox.max.y.to_pt(),
            );
            let clip_bounds = Bounds::from_corners(
                self.element_origin + point(px(x0.min(x1) as f32), px(y0.min(y1) as f32)),
                self.element_origin + point(px(x0.max(x1) as f32), px(y0.max(y1) as f32)),
            );
            let intersected = self.window.content_mask().bounds.intersect(&clip_bounds);
            let element_origin = self.element_origin;
            let fallback_color = self.fallback_color;
            let compiled = self.compiled;
            let cx = &mut *self.cx;
            self.window.with_content_mask(
                Some(ContentMask {
                    bounds: intersected,
                }),
                |window| {
                    let mut nested = FramePainter {
                        element_origin,
                        fallback_color,
                        compiled,
                        window,
                        cx,
                    };
                    nested.paint_frame(&group.frame, group_transform);
                },
            );
        } else {
            self.paint_frame(&group.frame, group_transform);
        }
    }

    fn paint_text_item(&mut self, text_item: &TextItem, transform: Transform) {
        if text_item.glyphs.is_empty() {
            return;
        }
        let Some(font_id) =
            resolve_gpui_font_id(self.window.text_system(), &text_item.font).log_err()
        else {
            return;
        };

        let glyph_color = paint_to_hsla(&text_item.fill, self.fallback_color);
        let sx = transform.sx.get() as f32;
        let ky = transform.ky.get() as f32;
        let kx = transform.kx.get() as f32;
        let sy = transform.sy.get() as f32;
        let is_uniform_upright =
            ky.abs() < 1e-4 && kx.abs() < 1e-4 && sx > 0.0 && (sx - sy).abs() < 1e-4;

        let mut cursor_x = Abs::zero();
        let mut cursor_y = Abs::zero();
        let scale_factor = self.window.scale_factor();

        for glyph in &text_item.glyphs {
            let local_x_pt = (cursor_x + glyph.x_offset.at(text_item.size)).to_pt();
            let local_y_pt = -(cursor_y + glyph.y_offset.at(text_item.size)).to_pt();
            let (world_x, world_y) = apply_typst_transform(transform, local_x_pt, local_y_pt);
            let glyph_origin = self.element_origin + point(px(world_x as f32), px(world_y as f32));
            let glyph_id = GlyphId(u32::from(glyph.id));

            if is_uniform_upright {
                let scaled_font_size = px((text_item.size.to_pt() as f32) * sx);
                self.window
                    .paint_glyph(
                        glyph_origin,
                        font_id,
                        glyph_id,
                        scaled_font_size,
                        glyph_color,
                    )
                    .log_err();
            } else {
                let unscaled_font_size = px(text_item.size.to_pt() as f32);
                let pivot_x = f32::from(glyph_origin.x) * scale_factor;
                let pivot_y = f32::from(glyph_origin.y) * scale_factor;
                let matrix = TransformationMatrix {
                    rotation_scale: [[sx, kx], [ky, sy]],
                    translation: [
                        pivot_x - sx * pivot_x - kx * pivot_y,
                        pivot_y - ky * pivot_x - sy * pivot_y,
                    ],
                };
                self.window
                    .paint_glyph_transformed(
                        glyph_origin,
                        font_id,
                        glyph_id,
                        unscaled_font_size,
                        glyph_color,
                        matrix,
                    )
                    .log_err();
            }

            cursor_x += glyph.x_advance.at(text_item.size);
            cursor_y += glyph.y_advance.at(text_item.size);
        }
    }

    fn paint_shape(&mut self, shape: &Shape, transform: Transform) {
        let min_hairline_px = 1.0 / self.window.scale_factor().max(1.0);

        if self.try_paint_shape_as_quad(shape, transform, min_hairline_px) {
            return;
        }

        if let Some(fill_paint) = &shape.fill {
            if let Some(fill_builder) = build_fill_path(
                &shape.geometry,
                shape.fill_rule,
                transform,
                self.element_origin,
                min_hairline_px,
            ) {
                if let Some(path) = fill_builder.build().log_err() {
                    let bg = paint_to_background(fill_paint, self.fallback_color);
                    self.window.paint_path(path, bg);
                }
            }
        }

        if let Some(stroke) = &shape.stroke {
            if let Some(stroke_builder) = build_stroke_path(
                &shape.geometry,
                stroke,
                transform,
                self.element_origin,
                min_hairline_px,
            ) {
                if let Some(path) = stroke_builder.build().log_err() {
                    let bg = paint_to_background(&stroke.paint, self.fallback_color);
                    self.window.paint_path(path, bg);
                }
            }
        }
    }

    fn try_paint_shape_as_quad(
        &mut self,
        shape: &Shape,
        transform: Transform,
        min_hairline_px: f32,
    ) -> bool {
        let sx = transform.sx.get();
        let ky = transform.ky.get();
        let kx = transform.kx.get();
        let sy = transform.sy.get();
        if ky.abs() > 1e-5 || kx.abs() > 1e-5 || sx <= 0.0 || sy <= 0.0 {
            return false;
        }

        match &shape.geometry {
            Geometry::Rect(rect_size) if shape.stroke.is_none() => {
                let Some(Paint::Solid(color)) = &shape.fill else {
                    return false;
                };
                let width = rect_size.x.to_pt();
                let height = rect_size.y.to_pt();
                if width == 0.0 || height == 0.0 {
                    return true;
                }
                let (x0, y0) = apply_typst_transform(transform, 0.0, 0.0);
                let (x1, y1) = apply_typst_transform(transform, width, height);
                let left = x0.min(x1) as f32;
                let top = y0.min(y1) as f32;
                let w = ((x1 - x0).abs() as f32).max(min_hairline_px);
                let h = ((y1 - y0).abs() as f32).max(min_hairline_px);
                let bounds = Bounds {
                    origin: self.element_origin + point(px(left), px(top)),
                    size: size(px(w), px(h)),
                };
                self.window
                    .paint_quad(gpui::fill(bounds, typst_color_to_hsla(color)));
                true
            }
            Geometry::Curve(curve) if shape.stroke.is_none() && (sx - sy).abs() <= 1e-4 => {
                let Some(Paint::Solid(color)) = &shape.fill else {
                    return false;
                };
                let Some((local_bounds, local_corners)) = try_extract_rounded_rect_quad(curve)
                else {
                    return false;
                };
                let (x0, y0) =
                    apply_typst_transform(transform, local_bounds.origin.x, local_bounds.origin.y);
                let (x1, y1) = apply_typst_transform(
                    transform,
                    local_bounds.origin.x + local_bounds.size.width,
                    local_bounds.origin.y + local_bounds.size.height,
                );
                let left = x0.min(x1) as f32;
                let top = y0.min(y1) as f32;
                let w = (x1 - x0).abs() as f32;
                let h = (y1 - y0).abs() as f32;
                if w <= 0.0 || h <= 0.0 {
                    return true;
                }
                let scale = sx as f32;
                let corners = Corners {
                    top_left: px((local_corners.top_left as f32) * scale),
                    top_right: px((local_corners.top_right as f32) * scale),
                    bottom_right: px((local_corners.bottom_right as f32) * scale),
                    bottom_left: px((local_corners.bottom_left as f32) * scale),
                };
                let bounds = Bounds {
                    origin: self.element_origin + point(px(left), px(top)),
                    size: size(px(w), px(h)),
                };
                self.window.paint_quad(quad(
                    bounds,
                    corners,
                    typst_color_to_hsla(color),
                    gpui::Edges::default(),
                    Hsla::transparent_black(),
                    gpui::BorderStyle::default(),
                ));
                true
            }
            Geometry::Line(end_point) if shape.fill.is_none() => {
                let Some(stroke) = &shape.stroke else {
                    return false;
                };
                if stroke.cap == LineCap::Round
                    || stroke
                        .dash
                        .as_ref()
                        .is_some_and(|dash| !dash.array.is_empty())
                {
                    return false;
                }
                let Paint::Solid(color) = &stroke.paint else {
                    return false;
                };
                let (x0, y0) = apply_typst_transform(transform, 0.0, 0.0);
                let (x1, y1) =
                    apply_typst_transform(transform, end_point.x.to_pt(), end_point.y.to_pt());
                let dx = (x1 - x0).abs();
                let dy = (y1 - y0).abs();
                if dy <= 1e-3 && dx > 0.0 {
                    let thickness = ((stroke.thickness.to_pt() * sy) as f32).max(min_hairline_px);
                    let cap_ext = if stroke.cap == LineCap::Square {
                        thickness * 0.5
                    } else {
                        0.0
                    };
                    let left = (x0.min(x1) as f32) - cap_ext;
                    let width = (dx as f32) + cap_ext * 2.0;
                    let top = (y0 as f32) - thickness * 0.5;
                    let bounds = Bounds {
                        origin: self.element_origin + point(px(left), px(top)),
                        size: size(px(width), px(thickness)),
                    };
                    self.window
                        .paint_quad(gpui::fill(bounds, typst_color_to_hsla(color)));
                    true
                } else if dx <= 1e-3 && dy > 0.0 {
                    let thickness = ((stroke.thickness.to_pt() * sx) as f32).max(min_hairline_px);
                    let cap_ext = if stroke.cap == LineCap::Square {
                        thickness * 0.5
                    } else {
                        0.0
                    };
                    let top = (y0.min(y1) as f32) - cap_ext;
                    let height = (dy as f32) + cap_ext * 2.0;
                    let left = (x0 as f32) - thickness * 0.5;
                    let bounds = Bounds {
                        origin: self.element_origin + point(px(left), px(top)),
                        size: size(px(thickness), px(height)),
                    };
                    self.window
                        .paint_quad(gpui::fill(bounds, typst_color_to_hsla(color)));
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn paint_typst_image(
        &mut self,
        image: &TypstImage,
        img_size: typst::layout::Size,
        transform: Transform,
    ) {
        let (x0, y0) = apply_typst_transform(transform, 0.0, 0.0);
        let (x1, y1) = apply_typst_transform(transform, img_size.x.to_pt(), img_size.y.to_pt());
        let left = x0.min(x1) as f32;
        let top = y0.min(y1) as f32;
        let width = (x1 - x0).abs() as f32;
        let height = (y1 - y0).abs() as f32;
        if width <= 0.0 || height <= 0.0 {
            return;
        }
        let target_bounds = Bounds {
            origin: self.element_origin + point(px(left), px(top)),
            size: size(px(width), px(height)),
        };

        let mut hasher = FxHasher::default();
        image.hash(&mut hasher);
        let cache_key = hasher.finish();

        let render_image = {
            let cached = self.compiled.decoded_images.lock().get(&cache_key).cloned();
            match cached {
                Some(ready) => Some(ready),
                None => {
                    let decoded =
                        decode_typst_image(image, width, self.window.scale_factor(), self.cx);
                    if let Some(ref img_arc) = decoded {
                        self.compiled
                            .decoded_images
                            .lock()
                            .insert(cache_key, img_arc.clone());
                    }
                    decoded
                }
            }
        };

        if let Some(render_image) = render_image {
            self.window
                .paint_image(
                    target_bounds,
                    target_bounds,
                    Corners::default(),
                    render_image,
                    0,
                    false,
                )
                .log_err();
        }
    }
}

fn decode_typst_image(
    image: &TypstImage,
    target_width_px: f32,
    scale_factor: f32,
    cx: &App,
) -> Option<Arc<RenderImage>> {
    match image.kind() {
        ImageKind::Raster(raster) => {
            let mut rgba_buffer: ImageBuffer<Rgba<u8>, Vec<u8>> = raster.dynamic().to_rgba8();
            for pixel in rgba_buffer.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
            let frame = ImageFrame::new(rgba_buffer);
            Some(Arc::new(RenderImage::new(SmallVec::from_const([frame]))))
        }
        ImageKind::Svg(svg) => {
            let natural_width = (svg.width() as f32).max(1.0);
            let scale = (target_width_px / natural_width * scale_factor).clamp(0.25, 8.0);
            cx.svg_renderer()
                .render_single_frame(svg.data().as_slice(), scale)
                .log_err()
        }
        ImageKind::Pdf(_) => None,
    }
}

pub fn try_extract_rounded_rect_quad(curve: &Curve) -> Option<(Bounds<f64>, Corners<f64>)> {
    const ONE_MINUS_ARC_K: f64 = 1.0 - 0.5522847498;

    let items = curve.0.as_slice();
    if items.len() < 5 || items.len() > 10 {
        return None;
    }

    let bbox = curve.bbox(None);
    let min_x = bbox.min.x.to_pt();
    let min_y = bbox.min.y.to_pt();
    let max_x = bbox.max.x.to_pt();
    let max_y = bbox.max.y.to_pt();
    let width = max_x - min_x;
    let height = max_y - min_y;
    if width <= 0.0 || height <= 0.0 {
        return None;
    }

    let mut index = 0usize;

    let CurveItem::Move(p0) = items.get(index)? else {
        return None;
    };
    index += 1;
    let p0_x = p0.x.to_pt();
    let p0_y = p0.y.to_pt();
    if (p0_x - min_x).abs() > 1e-2 {
        return None;
    }
    let raw_tl = p0_y - min_y;
    if raw_tl < -1e-2 {
        return None;
    }
    let top_left = if raw_tl <= 1e-2 {
        0.0
    } else {
        let CurveItem::Cubic(c1, c2, end) = items.get(index)? else {
            return None;
        };
        index += 1;
        let tol = 0.03 * raw_tl + 1e-2;
        if (end.x.to_pt() - (min_x + raw_tl)).abs() > tol
            || (end.y.to_pt() - min_y).abs() > 1e-2
            || (c1.x.to_pt() - min_x).abs() > 1e-2
            || (c1.y.to_pt() - (min_y + raw_tl * ONE_MINUS_ARC_K)).abs() > tol
            || (c2.x.to_pt() - (min_x + raw_tl * ONE_MINUS_ARC_K)).abs() > tol
            || (c2.y.to_pt() - min_y).abs() > 1e-2
        {
            return None;
        }
        raw_tl
    };

    let CurveItem::Line(tr_start) = items.get(index)? else {
        return None;
    };
    index += 1;
    if (tr_start.y.to_pt() - min_y).abs() > 1e-2 {
        return None;
    }
    let raw_tr = max_x - tr_start.x.to_pt();
    if raw_tr < -1e-2 {
        return None;
    }
    let top_right = if raw_tr <= 1e-2 {
        0.0
    } else {
        let CurveItem::Cubic(c1, c2, end) = items.get(index)? else {
            return None;
        };
        index += 1;
        let tol = 0.03 * raw_tr + 1e-2;
        if (end.x.to_pt() - max_x).abs() > 1e-2
            || (end.y.to_pt() - (min_y + raw_tr)).abs() > tol
            || (c1.x.to_pt() - (max_x - raw_tr * ONE_MINUS_ARC_K)).abs() > tol
            || (c1.y.to_pt() - min_y).abs() > 1e-2
            || (c2.x.to_pt() - max_x).abs() > 1e-2
            || (c2.y.to_pt() - (min_y + raw_tr * ONE_MINUS_ARC_K)).abs() > tol
        {
            return None;
        }
        raw_tr
    };

    let CurveItem::Line(br_start) = items.get(index)? else {
        return None;
    };
    index += 1;
    if (br_start.x.to_pt() - max_x).abs() > 1e-2 {
        return None;
    }
    let raw_br = max_y - br_start.y.to_pt();
    if raw_br < -1e-2 {
        return None;
    }
    let bottom_right = if raw_br <= 1e-2 {
        0.0
    } else {
        let CurveItem::Cubic(c1, c2, end) = items.get(index)? else {
            return None;
        };
        index += 1;
        let tol = 0.03 * raw_br + 1e-2;
        if (end.x.to_pt() - (max_x - raw_br)).abs() > tol
            || (end.y.to_pt() - max_y).abs() > 1e-2
            || (c1.x.to_pt() - max_x).abs() > 1e-2
            || (c1.y.to_pt() - (max_y - raw_br * ONE_MINUS_ARC_K)).abs() > tol
            || (c2.x.to_pt() - (max_x - raw_br * ONE_MINUS_ARC_K)).abs() > tol
            || (c2.y.to_pt() - max_y).abs() > 1e-2
        {
            return None;
        }
        raw_br
    };

    let CurveItem::Line(bl_start) = items.get(index)? else {
        return None;
    };
    index += 1;
    if (bl_start.y.to_pt() - max_y).abs() > 1e-2 {
        return None;
    }
    let raw_bl = bl_start.x.to_pt() - min_x;
    if raw_bl < -1e-2 {
        return None;
    }
    let bottom_left = if raw_bl <= 1e-2 {
        0.0
    } else {
        let CurveItem::Cubic(c1, c2, end) = items.get(index)? else {
            return None;
        };
        index += 1;
        let tol = 0.03 * raw_bl + 1e-2;
        if (end.x.to_pt() - min_x).abs() > 1e-2
            || (end.y.to_pt() - (max_y - raw_bl)).abs() > tol
            || (c1.x.to_pt() - (min_x + raw_bl * ONE_MINUS_ARC_K)).abs() > tol
            || (c1.y.to_pt() - max_y).abs() > 1e-2
            || (c2.x.to_pt() - min_x).abs() > 1e-2
            || (c2.y.to_pt() - (max_y - raw_bl * ONE_MINUS_ARC_K)).abs() > tol
        {
            return None;
        }
        raw_bl
    };

    match &items[index..] {
        [CurveItem::Close] => {}
        [CurveItem::Line(end), CurveItem::Close]
            if (end.x.to_pt() - p0_x).abs() <= 1e-2 && (end.y.to_pt() - p0_y).abs() <= 1e-2 => {}
        _ => return None,
    }

    Some((
        Bounds {
            origin: point(min_x, min_y),
            size: size(width, height),
        },
        Corners {
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        },
    ))
}

fn build_fill_path(
    geometry: &Geometry,
    fill_rule: FillRule,
    transform: Transform,
    element_origin: Point<Pixels>,
    min_hairline_px: f32,
) -> Option<PathBuilder> {
    let gpui_rule = match fill_rule {
        FillRule::NonZero => GpuiFillRule::NonZero,
        FillRule::EvenOdd => GpuiFillRule::EvenOdd,
    };
    let mut builder = PathBuilder::fill().with_style(PathStyle::Fill(FillOptions {
        fill_rule: gpui_rule,
    }));

    match geometry {
        Geometry::Line(_) => return None,
        Geometry::Rect(rect_size) => {
            let mut width = rect_size.x.to_pt();
            let mut height = rect_size.y.to_pt();
            if width == 0.0 || height == 0.0 {
                return None;
            }
            if width.abs() > 0.0 && width.abs() < f64::from(min_hairline_px) {
                width = width.signum() * f64::from(min_hairline_px);
            }
            if height.abs() > 0.0 && height.abs() < f64::from(min_hairline_px) {
                height = height.signum() * f64::from(min_hairline_px);
            }
            let p00 = map_window_point(transform, element_origin, 0.0, 0.0);
            let p10 = map_window_point(transform, element_origin, width, 0.0);
            let p11 = map_window_point(transform, element_origin, width, height);
            let p01 = map_window_point(transform, element_origin, 0.0, height);
            builder.move_to(p00);
            builder.line_to(p10);
            builder.line_to(p11);
            builder.line_to(p01);
            builder.close();
        }
        Geometry::Curve(curve) => {
            if curve.0.is_empty() {
                return None;
            }
            append_curve_to_builder(&mut builder, curve, transform, element_origin);
        }
    }

    Some(builder)
}

fn build_stroke_path(
    geometry: &Geometry,
    stroke: &FixedStroke,
    transform: Transform,
    element_origin: Point<Pixels>,
    min_hairline_px: f32,
) -> Option<PathBuilder> {
    let (scale_x, scale_y) = transform_axis_scales(transform);
    let avg_scale = ((scale_x + scale_y) * 0.5) as f32;
    let raw_thickness = (stroke.thickness.to_pt() as f32) * avg_scale;
    if raw_thickness <= 0.0 {
        return None;
    }
    let clamped_thickness = raw_thickness.max(min_hairline_px);

    let line_cap = match stroke.cap {
        LineCap::Butt => GpuiLineCap::Butt,
        LineCap::Round => GpuiLineCap::Round,
        LineCap::Square => GpuiLineCap::Square,
    };
    let line_join = match stroke.join {
        LineJoin::Miter => GpuiLineJoin::Miter,
        LineJoin::Round => GpuiLineJoin::Round,
        LineJoin::Bevel => GpuiLineJoin::Bevel,
    };

    let stroke_options = StrokeOptions {
        line_width: clamped_thickness,
        line_cap,
        line_join,
        miter_limit: (stroke.miter_limit.get() as f32).max(1.0),
    };

    let mut builder =
        PathBuilder::stroke(px(clamped_thickness)).with_style(PathStyle::Stroke(stroke_options));

    if let Some(dash) = &stroke.dash {
        if !dash.array.is_empty() {
            let dash_pixels: Vec<Pixels> = dash
                .array
                .iter()
                .map(|len| px(((len.to_pt() as f32) * avg_scale).max(min_hairline_px)))
                .collect();
            builder = builder.dash_array(&dash_pixels);
        }
    }

    match geometry {
        Geometry::Line(end_point) => {
            let p0 = map_window_point(transform, element_origin, 0.0, 0.0);
            let p1 = map_window_point(
                transform,
                element_origin,
                end_point.x.to_pt(),
                end_point.y.to_pt(),
            );
            builder.move_to(p0);
            builder.line_to(p1);
        }
        Geometry::Rect(rect_size) => {
            let width = rect_size.x.to_pt();
            let height = rect_size.y.to_pt();
            let p00 = map_window_point(transform, element_origin, 0.0, 0.0);
            let p10 = map_window_point(transform, element_origin, width, 0.0);
            let p11 = map_window_point(transform, element_origin, width, height);
            let p01 = map_window_point(transform, element_origin, 0.0, height);
            builder.move_to(p00);
            builder.line_to(p10);
            builder.line_to(p11);
            builder.line_to(p01);
            builder.close();
        }
        Geometry::Curve(curve) => {
            if curve.0.is_empty() {
                return None;
            }
            append_curve_to_builder(&mut builder, curve, transform, element_origin);
        }
    }

    Some(builder)
}

fn append_curve_to_builder(
    builder: &mut PathBuilder,
    curve: &Curve,
    transform: Transform,
    element_origin: Point<Pixels>,
) {
    for item in &curve.0 {
        match item {
            CurveItem::Move(pt) => {
                builder.move_to(map_window_point(
                    transform,
                    element_origin,
                    pt.x.to_pt(),
                    pt.y.to_pt(),
                ));
            }
            CurveItem::Line(pt) => {
                builder.line_to(map_window_point(
                    transform,
                    element_origin,
                    pt.x.to_pt(),
                    pt.y.to_pt(),
                ));
            }
            CurveItem::Cubic(control_a, control_b, end_point) => {
                // GPUI's `cubic_bezier_to` takes `(to, control_a, control_b)`.
                let mapped_end = map_window_point(
                    transform,
                    element_origin,
                    end_point.x.to_pt(),
                    end_point.y.to_pt(),
                );
                let mapped_a = map_window_point(
                    transform,
                    element_origin,
                    control_a.x.to_pt(),
                    control_a.y.to_pt(),
                );
                let mapped_b = map_window_point(
                    transform,
                    element_origin,
                    control_b.x.to_pt(),
                    control_b.y.to_pt(),
                );
                builder.cubic_bezier_to(mapped_end, mapped_a, mapped_b);
            }
            CurveItem::Close => builder.close(),
        }
    }
}

fn map_window_point(
    transform: Transform,
    element_origin: Point<Pixels>,
    x_pt: f64,
    y_pt: f64,
) -> Point<Pixels> {
    let (world_x, world_y) = apply_typst_transform(transform, x_pt, y_pt);
    element_origin + point(px(world_x as f32), px(world_y as f32))
}

#[inline]
fn apply_typst_transform(transform: Transform, x_pt: f64, y_pt: f64) -> (f64, f64) {
    let out_x = transform.sx.get() * x_pt + transform.kx.get() * y_pt + transform.tx.to_pt();
    let out_y = transform.ky.get() * x_pt + transform.sy.get() * y_pt + transform.ty.to_pt();
    (out_x, out_y)
}

#[inline]
fn transform_axis_scales(transform: Transform) -> (f64, f64) {
    let scale_x = (transform.sx.get().powi(2) + transform.ky.get().powi(2)).sqrt();
    let scale_y = (transform.kx.get().powi(2) + transform.sy.get().powi(2)).sqrt();
    (scale_x, scale_y)
}

fn typst_color_to_hsla(color: &Color) -> Hsla {
    let rgb = color.to_rgb();
    gpui::Rgba {
        r: rgb.red.clamp(0.0, 1.0),
        g: rgb.green.clamp(0.0, 1.0),
        b: rgb.blue.clamp(0.0, 1.0),
        a: rgb.alpha.clamp(0.0, 1.0),
    }
    .into()
}

fn paint_to_hsla(paint: &Paint, fallback: Hsla) -> Hsla {
    match paint {
        Paint::Solid(color) => typst_color_to_hsla(color),
        Paint::Gradient(gradient) => {
            let sampled = gradient.sample(RatioOrAngle::Ratio(Ratio::new(0.5)));
            typst_color_to_hsla(&sampled)
        }
        Paint::Tiling(_) => fallback,
    }
}

fn paint_to_background(paint: &Paint, fallback: Hsla) -> Background {
    match paint {
        Paint::Solid(color) => typst_color_to_hsla(color).into(),
        Paint::Gradient(Gradient::Linear(linear)) if linear.stops.len() >= 2 => {
            let first = &linear.stops[0];
            let last = &linear.stops[linear.stops.len() - 1];
            let angle_deg = (linear.angle.to_deg() as f32 + 90.0).rem_euclid(360.0);
            gpui::linear_gradient(
                angle_deg,
                gpui::linear_color_stop(typst_color_to_hsla(&first.0), first.1.get() as f32),
                gpui::linear_color_stop(typst_color_to_hsla(&last.0), last.1.get() as f32),
            )
        }
        Paint::Gradient(gradient) => {
            let sampled = gradient.sample(RatioOrAngle::Ratio(Ratio::new(0.5)));
            typst_color_to_hsla(&sampled).into()
        }
        Paint::Tiling(_) => fallback.into(),
    }
}
