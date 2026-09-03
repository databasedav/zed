use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, OnceLock};

use anyhow::Result;
use collections::FxHasher;
use gpui::{
    App, AssetSource, Font as GpuiFont, FontFeatures, FontId, FontStyle as GpuiFontStyle,
    FontWeight as GpuiFontWeight, Global, Hsla, SharedString, TextSystem,
};
use parking_lot::RwLock;
use typst::diag::{FileError, FileResult};
use typst::foundations::{Bytes, Datetime};
use typst::syntax::{FileId, RootedPath, Source, VirtualPath, VirtualRoot};
use typst::text::{Font, FontBook, FontInfo, FontInstance, FontStyle, Tag};
use typst::utils::LazyHash;
use typst::{Library, LibraryExt, World};

static SHARED_LIBRARY: LazyLock<LazyHash<Library>> =
    LazyLock::new(|| LazyHash::new(Library::default()));

static BASE_FONT_CATALOG: LazyLock<Arc<TypstFontCatalog>> =
    LazyLock::new(|| Arc::new(TypstFontCatalog::build(None)));

static ASSET_FONT_CATALOG: OnceLock<Arc<TypstFontCatalog>> = OnceLock::new();

static USER_TYPST_PREAMBLE: RwLock<Option<SharedString>> = RwLock::new(None);

pub fn set_user_typst_preamble(contents: Option<String>) -> bool {
    let normalized = contents.and_then(|raw| {
        if raw.trim().is_empty() {
            None
        } else {
            Some(SharedString::from(raw))
        }
    });
    let mut guard = USER_TYPST_PREAMBLE.write();
    if *guard == normalized {
        return false;
    }
    *guard = normalized;
    true
}

pub fn user_typst_preamble() -> Option<SharedString> {
    USER_TYPST_PREAMBLE.read().clone()
}

struct GlobalTypstFontCatalog(Arc<TypstFontCatalog>);

impl Global for GlobalTypstFontCatalog {}

enum FontSlotSource {
    Loaded(Option<Font>),
    SharedBytes { data: Bytes, index: u32 },
    SystemFile { path: PathBuf, index: u32 },
}

struct FontSlot {
    source: FontSlotSource,
    loaded: OnceLock<Option<Font>>,
}

impl FontSlot {
    fn from_loaded(font: Font) -> Self {
        Self {
            source: FontSlotSource::Loaded(Some(font.clone())),
            loaded: OnceLock::from(Some(font)),
        }
    }

    fn from_shared_bytes(data: Bytes, index: u32) -> Self {
        Self {
            source: FontSlotSource::SharedBytes { data, index },
            loaded: OnceLock::new(),
        }
    }

    fn from_system_file(path: PathBuf, index: u32) -> Self {
        Self {
            source: FontSlotSource::SystemFile { path, index },
            loaded: OnceLock::new(),
        }
    }

    fn get(&self) -> Option<Font> {
        self.loaded
            .get_or_init(|| match &self.source {
                FontSlotSource::Loaded(font) => font.clone(),
                FontSlotSource::SharedBytes { data, index } => Font::new(data.clone(), *index),
                FontSlotSource::SystemFile { path, index } => {
                    let bytes = std::fs::read(path).ok()?;
                    Font::new(Bytes::new(bytes), *index)
                }
            })
            .clone()
    }
}

pub struct TypstFontCatalog {
    book: LazyHash<FontBook>,
    slots: Vec<FontSlot>,
    system_bytes_cache: RwLock<collections::HashMap<PathBuf, Bytes>>,
}

impl TypstFontCatalog {
    pub fn global_or_init(cx: &mut App) -> Arc<Self> {
        if let Some(existing) = cx.try_global::<GlobalTypstFontCatalog>() {
            return existing.0.clone();
        }
        let asset_source = cx.asset_source().clone();
        let catalog = Self::shared(asset_source.as_ref());
        cx.set_global(GlobalTypstFontCatalog(catalog.clone()));
        catalog
    }

    pub fn shared(asset_source: &dyn AssetSource) -> Arc<Self> {
        if let Some(existing) = ASSET_FONT_CATALOG.get() {
            return existing.clone();
        }
        let has_asset_fonts = asset_source
            .list("fonts")
            .ok()
            .is_some_and(|paths| !paths.is_empty());
        if has_asset_fonts {
            ASSET_FONT_CATALOG
                .get_or_init(|| Arc::new(Self::build(Some(asset_source))))
                .clone()
        } else {
            BASE_FONT_CATALOG.clone()
        }
    }

    pub fn base() -> Arc<Self> {
        BASE_FONT_CATALOG.clone()
    }

    pub fn build(asset_source: Option<&dyn AssetSource>) -> Self {
        let mut book = FontBook::new();
        let mut slots = Vec::new();

        for font_bytes in typst_assets::fonts() {
            let bytes = Bytes::new(font_bytes);
            for font in Font::iter(bytes) {
                book.push(font.info().clone());
                slots.push(FontSlot::from_loaded(font));
            }
        }

        if let Some(asset_source) = asset_source {
            load_zed_asset_fonts(asset_source, &mut book, &mut slots);
        }

        let mut system_db = fontdb::Database::new();
        system_db.load_system_fonts();
        for face in system_db.faces() {
            let maybe_info = system_db
                .with_face_data(face.id, |font_data, face_index| {
                    FontInfo::new(font_data, face_index)
                })
                .flatten();
            let Some(info) = maybe_info.map(normalize_typst_font_info) else {
                continue;
            };
            match &face.source {
                fontdb::Source::File(path) => {
                    book.push(info);
                    slots.push(FontSlot::from_system_file(path.clone(), face.index));
                }
                fontdb::Source::Binary(data) | fontdb::Source::SharedFile(_, data) => {
                    let shared_vec = data.as_ref().as_ref().to_vec();
                    let bytes = Bytes::new(shared_vec);
                    book.push(info);
                    slots.push(FontSlot::from_shared_bytes(bytes, face.index));
                }
            }
        }

        Self {
            book: LazyHash::new(book),
            slots,
            system_bytes_cache: RwLock::default(),
        }
    }

    pub fn book(&self) -> &LazyHash<FontBook> {
        &self.book
    }

    pub fn font(&self, index: usize) -> Option<Font> {
        let slot = self.slots.get(index)?;
        if let FontSlotSource::SystemFile {
            path,
            index: face_index,
        } = &slot.source
        {
            if let Some(ready) = slot.loaded.get() {
                return ready.clone();
            }
            let cached_bytes = { self.system_bytes_cache.read().get(path).cloned() };
            let bytes = match cached_bytes {
                Some(bytes) => bytes,
                None => {
                    let raw = std::fs::read(path).ok()?;
                    let bytes = Bytes::new(raw);
                    self.system_bytes_cache
                        .write()
                        .insert(path.clone(), bytes.clone());
                    bytes
                }
            };
            return slot
                .loaded
                .get_or_init(|| Font::new(bytes, *face_index))
                .clone();
        }
        slot.get()
    }
}

fn load_zed_asset_fonts(
    asset_source: &dyn AssetSource,
    book: &mut FontBook,
    slots: &mut Vec<FontSlot>,
) {
    let Ok(font_paths) = asset_source.list("fonts") else {
        return;
    };
    for font_path in font_paths {
        if !font_path.ends_with(".ttf") && !font_path.ends_with(".otf") {
            continue;
        }
        let Ok(Some(font_bytes)) = asset_source.load(&font_path) else {
            continue;
        };
        let bytes = match font_bytes {
            Cow::Borrowed(static_slice) => Bytes::new(static_slice),
            Cow::Owned(owned_vec) => Bytes::new(owned_vec),
        };
        for font in Font::iter(bytes) {
            book.push(normalize_typst_font_info(font.info().clone()));
            slots.push(FontSlot::from_loaded(font));
        }
    }
}

fn normalize_typst_font_info(mut info: FontInfo) -> FontInfo {
    const LEGACY_WEIGHT_SUFFIXES: &[&str] = &[
        "smbld", "sembld", "exbld", "extbld", "dmibld", "demibld", "ultrabld", "ubld", "bld",
    ];
    let lower = info.family.to_ascii_lowercase();
    for suffix in LEGACY_WEIGHT_SUFFIXES {
        if let Some(prefix) = lower.strip_suffix(suffix) {
            let trimmed_len = prefix.trim_end_matches([' ', '-', '_']).len();
            if trimmed_len > 0 && trimmed_len < prefix.len() {
                info.family.truncate(trimmed_len);
                break;
            }
        }
    }
    info
}

pub fn resolve_gpui_font_id(
    text_system: &TextSystem,
    font_instance: &FontInstance,
) -> Result<FontId> {
    let font = font_instance.font();
    let wght_tag = Tag::from_bytes(b"wght");
    let effective_weight = font_instance
        .variations()
        .0
        .iter()
        .find_map(|(tag, value)| (*tag == wght_tag).then_some(value.0))
        .unwrap_or_else(|| font.info().variant.weight.to_number() as f32)
        .clamp(100.0, 900.0);
    let quantized_weight = effective_weight.round() as u16;

    let mut hasher = FxHasher::default();
    font.data().hash(&mut hasher);
    quantized_weight.hash(&mut hasher);
    let cache_key = hasher.finish();
    let face_index = font.index();

    text_system.register_font_face(cache_key, face_index, || {
        let font_bytes = Cow::Owned(font.data().as_slice().to_vec());
        let info = font.info();
        let gpui_style = match info.variant.style {
            FontStyle::Normal => GpuiFontStyle::Normal,
            FontStyle::Italic => GpuiFontStyle::Italic,
            FontStyle::Oblique => GpuiFontStyle::Oblique,
        };
        let gpui_weight = GpuiFontWeight(effective_weight);
        let fallback_descriptor = GpuiFont {
            family: SharedString::from(info.family.clone()),
            features: FontFeatures::default(),
            fallbacks: None,
            weight: gpui_weight,
            style: gpui_style,
        };
        (font_bytes, fallback_descriptor)
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypstRenderStyle {
    pub text_color: Hsla,
    pub muted_text_color: Hsla,
    pub link_color: Hsla,
    pub code_background_color: Hsla,
    pub border_color: Hsla,
    pub table_header_background_color: Hsla,
    pub table_row_stripe_color: Hsla,
    pub font_family: SharedString,
    pub monospace_font_family: SharedString,
    pub base_font_size_pt: f32,
    pub code_font_size_pt: f32,
    pub line_height_pt: f32,
    pub paragraph_spacing_pt: f32,
    pub heading_font_sizes_pt: [f32; 6],
    pub page_width_pt: Option<f32>,
    pub tight_bounds: bool,
    pub user_preamble: Option<SharedString>,
}

impl Default for TypstRenderStyle {
    fn default() -> Self {
        Self {
            text_color: gpui::hsla(0.0, 0.0, 0.9, 1.0),
            muted_text_color: gpui::hsla(0.0, 0.0, 0.65, 1.0),
            link_color: gpui::hsla(0.58, 0.75, 0.6, 1.0),
            code_background_color: gpui::hsla(0.0, 0.0, 0.2, 0.45),
            border_color: gpui::hsla(0.0, 0.0, 0.4, 0.45),
            table_header_background_color: gpui::hsla(0.0, 0.0, 0.16, 0.65),
            table_row_stripe_color: gpui::hsla(0.0, 0.0, 0.14, 0.35),
            font_family: SharedString::new_static("IBM Plex Sans"),
            monospace_font_family: SharedString::new_static("Lilex"),
            base_font_size_pt: 14.0,
            code_font_size_pt: 12.25,
            line_height_pt: 18.2,
            paragraph_spacing_pt: 8.0,
            heading_font_sizes_pt: [17.1, 16.0, 15.0, 14.0, 13.3, 12.6],
            page_width_pt: None,
            tight_bounds: false,
            user_preamble: None,
        }
    }
}

impl TypstRenderStyle {
    pub fn build_builtin_preamble(&self) -> String {
        let width_expr = match self.page_width_pt {
            Some(width_pt) => format!("{:.2}pt", width_pt.max(32.0)),
            None => "auto".to_string(),
        };
        let text_hex = hsla_to_typst_hex(self.text_color);
        let muted_hex = hsla_to_typst_hex(self.muted_text_color);
        let link_hex = hsla_to_typst_hex(self.link_color);
        let mut link_underline_color = self.link_color;
        link_underline_color.a *= 0.5;
        let link_underline_hex = hsla_to_typst_hex(link_underline_color);
        let code_bg_hex = hsla_to_typst_hex(self.code_background_color);
        let border_hex = hsla_to_typst_hex(self.border_color);
        let table_head_bg_hex = hsla_to_typst_hex(self.table_header_background_color);
        let table_stripe_bg_hex = hsla_to_typst_hex(self.table_row_stripe_color);
        let primary_font = normalize_typst_font_family(&self.font_family, false);
        let mono_font = normalize_typst_font_family(&self.monospace_font_family, true);
        let font_size = self.base_font_size_pt.max(6.0);
        let code_size = self.code_font_size_pt.max(6.0);

        if self.tight_bounds {
            return format!(
                "#set page(width: {width_expr}, height: auto, margin: (x: 0.5pt, y: 1pt), fill: none)\n\
                 #set text(font: ({primary_font}, \"IBM Plex Sans\", \"Libertinus Serif\"), fill: rgb(\"{text_hex}\"), size: {font_size:.2}pt, hyphenate: false)\n"
            );
        }

        let line_height_pt = self.line_height_pt.max(font_size * 1.25);
        // `0.86em - (-0.19em) = 1.05em` matches Zed's Markdown inline code chip bounds
        // while keeping baseline-to-baseline distance equal to `line_height_pt`.
        let leading_pt = (line_height_pt - font_size * 1.05).max(2.0);
        let par_spacing_pt = leading_pt + self.paragraph_spacing_pt.max(4.0);
        let heading_above_pt = leading_pt + (font_size * 0.65);
        let heading_below_pt = leading_pt + (font_size * 0.28);
        let list_item_spacing_pt = leading_pt + (font_size * 0.20);
        let list_block_spacing_pt = leading_pt + (font_size * 0.28);
        let [h1_pt, h2_pt, h3_pt, h4_pt, h5_pt, h6_pt] = self.heading_font_sizes_pt;
        // Typst's built-in `RawElem` show-set applies `size: 0.8em` first:
        // - Inline code uses `1.25em` (`0.8 * 1.25 = 1.0em`) + `weight: "regular"` + `#highlight(...)`
        //   to match GPUI's `StyledText` + `RenderedLine::paint_code_chips` in Zed's Markdown renderer.
        // - Block raw uses `1.25 * (code_size / font_size)` to match fenced code block buffer font size.
        let block_code_scale = (code_size / font_size).clamp(0.75, 1.05);
        let block_raw_em = 1.25 * block_code_scale;

        format!(
            "#let zed-text = rgb(\"{text_hex}\")\n\
             #let zed-muted = rgb(\"{muted_hex}\")\n\
             #let zed-link = rgb(\"{link_hex}\")\n\
             #let zed-link-underline = rgb(\"{link_underline_hex}\")\n\
             #let zed-code-bg = rgb(\"{code_bg_hex}\")\n\
             #let zed-border = rgb(\"{border_hex}\")\n\
             #let zed-table-head-bg = rgb(\"{table_head_bg_hex}\")\n\
             #let zed-table-stripe-bg = rgb(\"{table_stripe_bg_hex}\")\n\
             #let zed-ui-font = {primary_font}\n\
             #let zed-mono-font = {mono_font}\n\
             #let zed-ui-size = {font_size:.2}pt\n\
             #let zed-code-size = {code_size:.2}pt\n\
             #let zed-line-height = {line_height_pt:.2}pt\n\
             #let zed-leading = {leading_pt:.2}pt\n\
             #let zed-par-spacing = {par_spacing_pt:.2}pt\n\
             #set page(width: {width_expr}, height: auto, margin: 0pt, fill: none)\n\
             #set text(font: (zed-ui-font, \"IBM Plex Sans\", \"Libertinus Serif\"), fill: zed-text, size: zed-ui-size, top-edge: 0.86em, bottom-edge: -0.19em, hyphenate: false)\n\
             #set strong(delta: 0)\n\
             #show strong: set text(weight: \"semibold\")\n\
             #set par(leading: zed-leading, spacing: zed-par-spacing, justify: false)\n\
             #show heading: set block(above: {heading_above_pt:.2}pt, below: {heading_below_pt:.2}pt)\n\
             #show heading: set par(leading: 0.22em)\n\
             #show heading.where(level: 1): set text(size: {h1_pt:.2}pt, weight: \"semibold\")\n\
             #show heading.where(level: 2): set text(size: {h2_pt:.2}pt, weight: \"semibold\")\n\
             #show heading.where(level: 3): set text(size: {h3_pt:.2}pt, weight: \"semibold\")\n\
             #show heading.where(level: 4): set text(size: {h4_pt:.2}pt, weight: \"semibold\")\n\
             #show heading.where(level: 5): set text(size: {h5_pt:.2}pt, weight: \"semibold\")\n\
             #show heading.where(level: 6): set text(size: {h6_pt:.2}pt, weight: \"semibold\")\n\
             #set list(marker: [•], indent: 6pt, body-indent: 5pt, spacing: {list_item_spacing_pt:.2}pt)\n\
             #set enum(numbering: \"1.\", full: false, indent: 6pt, body-indent: 5pt, spacing: {list_item_spacing_pt:.2}pt)\n\
             #show list: set block(above: {list_block_spacing_pt:.2}pt, below: {list_block_spacing_pt:.2}pt)\n\
             #show enum: set block(above: {list_block_spacing_pt:.2}pt, below: {list_block_spacing_pt:.2}pt)\n\
             #set underline(stroke: 1pt + zed-link-underline, offset: 0.19em, evade: true)\n\
             #show link: set text(fill: zed-link)\n\
             #show link: underline\n\
             #set highlight(fill: zed-code-bg, stroke: none, radius: 4pt, extent: 1.25pt, top-edge: 0.86em, bottom-edge: -0.19em)\n\
             #show raw.where(block: false): set text(font: (zed-mono-font, \"Lilex\", \"DejaVu Sans Mono\"), size: 1.25em, weight: \"regular\", top-edge: 0.86em, bottom-edge: -0.19em)\n\
             #show raw.where(block: false): highlight\n\
             #show raw.where(block: true): set text(font: (zed-mono-font, \"Lilex\", \"DejaVu Sans Mono\"), size: {block_raw_em:.4}em, weight: \"regular\")\n\
             #show raw.where(block: true): block.with(width: 100%, fill: zed-code-bg, stroke: 0.75pt + zed-border, inset: 8pt, radius: 4pt)\n\
             #set table(stroke: 0.75pt + zed-border, inset: (x: 8pt, y: 6pt), align: horizon, fill: (_, y) => if y == 0 {{ zed-table-head-bg }} else if calc.odd(y) {{ zed-table-stripe-bg }} else {{ none }})\n\
             #show table.cell.where(y: 0): set text(weight: \"semibold\")\n\
             #set quote(block: true)\n\
             #show quote.where(block: true): set text(fill: zed-muted)\n\
             #show quote.where(block: true): block.with(stroke: (left: 2pt + zed-border), inset: (left: 10pt, y: 2pt))\n"
        )
    }

    pub fn build_preamble(&self) -> String {
        let mut preamble = self.build_builtin_preamble();
        if !self.tight_bounds
            && let Some(user_preamble) = &self.user_preamble
        {
            if !preamble.ends_with('\n') {
                preamble.push('\n');
            }
            preamble.push_str(user_preamble.as_ref());
            if !preamble.ends_with('\n') {
                preamble.push('\n');
            }
        }
        preamble
    }
}

fn normalize_typst_font_family(family: &str, monospace: bool) -> String {
    let resolved = match family {
        ".ZedSans" | ".SystemUIFont" | "system-ui" | "sans-serif" | "" => {
            if monospace {
                "Lilex"
            } else {
                "IBM Plex Sans"
            }
        }
        ".ZedMono" | "monospace" | "ui-monospace" => "Lilex",
        other => other,
    };
    let escaped = resolved.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

pub fn hsla_to_typst_hex(color: Hsla) -> String {
    let rgba = color.to_rgb();
    let red = (rgba.r.clamp(0.0, 1.0) * 255.0).round() as u8;
    let green = (rgba.g.clamp(0.0, 1.0) * 255.0).round() as u8;
    let blue = (rgba.b.clamp(0.0, 1.0) * 255.0).round() as u8;
    let alpha = (rgba.a.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{red:02x}{green:02x}{blue:02x}{alpha:02x}")
}

pub struct ZedTypstWorld {
    main_id: FileId,
    main_source: Source,
    preamble_byte_len: usize,
    font_catalog: Arc<TypstFontCatalog>,
}

impl ZedTypstWorld {
    pub fn new(preamble: &str, body: &str, font_catalog: Arc<TypstFontCatalog>) -> Result<Self> {
        let virtual_path = VirtualPath::new("main.typ")
            .map_err(|error| anyhow::anyhow!("invalid virtual path for main.typ: {error:?}"))?;
        let main_id = RootedPath::new(VirtualRoot::Project, virtual_path).intern();
        let mut full_text = String::with_capacity(preamble.len() + 1 + body.len());
        full_text.push_str(preamble);
        if !preamble.is_empty() && !preamble.ends_with('\n') {
            full_text.push('\n');
        }
        let preamble_byte_len = full_text.len();
        full_text.push_str(body);
        let main_source = Source::new(main_id, full_text);
        Ok(Self {
            main_id,
            main_source,
            preamble_byte_len,
            font_catalog,
        })
    }

    pub fn preamble_byte_len(&self) -> usize {
        self.preamble_byte_len
    }

    pub fn main_source(&self) -> &Source {
        &self.main_source
    }
}

impl World for ZedTypstWorld {
    fn library(&self) -> &LazyHash<Library> {
        &SHARED_LIBRARY
    }

    fn book(&self) -> &LazyHash<FontBook> {
        self.font_catalog.book()
    }

    fn main(&self) -> FileId {
        self.main_id
    }

    fn source(&self, id: FileId) -> FileResult<Source> {
        if id == self.main_id {
            Ok(self.main_source.clone())
        } else {
            // Deny arbitrary filesystem reads from untrusted agent/document Typst code.
            Err(FileError::AccessDenied)
        }
    }

    fn file(&self, _id: FileId) -> FileResult<Bytes> {
        Err(FileError::AccessDenied)
    }

    fn font(&self, index: usize) -> Option<Font> {
        self.font_catalog.font(index)
    }

    fn today(&self, _offset: Option<typst::foundations::Duration>) -> Option<Datetime> {
        None
    }
}
