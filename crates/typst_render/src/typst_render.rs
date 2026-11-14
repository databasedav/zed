mod element;
mod preprocess;
mod world;

pub use element::{
    CompiledTypstLayout, PreparedTypstSegment, TypstFrameElement, TypstGlyphCluster,
    TypstLinkRegion, try_extract_rounded_rect_quad,
};
pub use preprocess::{
    OffsetMap, TypstCodeBlock, TypstDocumentSegment, TypstInlineCodeSpan, convert_math_to_typst,
    escape_typst_plain_text, extract_inline_code_spans, looks_like_latex_math,
    preprocess_typst_segment, segment_typst_document,
};
pub use typst;
pub use typst_layout;
pub use typst_syntax;
pub use world::{
    TypstFontCatalog, TypstRenderStyle, ZedTypstWorld, hsla_to_typst_hex, resolve_gpui_font_id,
    set_user_typst_preamble, user_typst_preamble,
};

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Arc;

    use collections::HashMap;
    use gpui::{ParentElement as _, Render, Styled as _, TestAppContext, Window, div, px, size};

    use super::*;

    #[test]
    fn compiles_native_typst_math_and_arrows() {
        let catalog = TypstFontCatalog::base();
        let source = "$ -> \"Mind LLM\" -> $";
        let segment =
            PreparedTypstSegment::new(source, 0..source.len(), &HashMap::default(), catalog);
        let layout = segment.layout_for_style(&TypstRenderStyle::default());

        assert!(layout.size().width > px(10.0));
        assert!(layout.size().height > px(5.0));
        assert!(
            layout.rendered_text().contains('→'),
            "expected right arrow in rendered text, got {:?}",
            layout.rendered_text()
        );
        assert!(layout.rendered_text().contains("Mind LLM"));
    }

    #[test]
    fn converts_legacy_latex_math_via_mitex() {
        let catalog = TypstFontCatalog::base();
        let source = "Flow: $\\rightarrow$ Mind LLM $\\rightarrow$ output and $\\frac{a + b}{2}$";
        let segment =
            PreparedTypstSegment::new(source, 0..source.len(), &HashMap::default(), catalog);
        let layout = segment.layout_for_style(&TypstRenderStyle::default());

        assert!(
            layout.rendered_text().contains("Flow:"),
            "expected Flow: in {:?}",
            layout.rendered_text()
        );
        assert!(
            layout.rendered_text().contains('→'),
            "expected arrow from \\rightarrow in {:?}",
            layout.rendered_text()
        );
        assert!(layout.rendered_text().contains("Mind LLM"));
    }

    #[test]
    fn recovers_gracefully_during_streaming_incomplete_tokens() {
        let catalog = TypstFontCatalog::base();
        let full = "= Architecture\n\n*Bold streaming* with #link(\"file:///tmp/test.rs\")[test.rs] and $ sum_(i=1)^n x_i $";
        for cutoff in 1..full.len() {
            if !full.is_char_boundary(cutoff) {
                continue;
            }
            let prefix = &full[..cutoff];
            let segment = PreparedTypstSegment::new(
                prefix,
                0..prefix.len(),
                &HashMap::default(),
                catalog.clone(),
            );
            let layout = segment.layout_for_style(&TypstRenderStyle {
                page_width_pt: Some(420.0),
                ..TypstRenderStyle::default()
            });
            assert!(
                layout.size().height >= px(0.0),
                "streaming prefix failed at cutoff {cutoff}: {prefix:?}"
            );
        }
    }

    #[test]
    fn segments_fenced_code_blocks_and_preserves_typst_tables() {
        let doc = "= Summary\n\n#table(\n  columns: 2,\n  [*Key*], [*Value*],\n  [Mode], [Typst],\n)\n\n```rust path/to/lib.rs#L1-L3\nfn main() {}\n```\n\nAfter code.";
        let segments = segment_typst_document(doc);
        assert_eq!(segments.len(), 3);
        assert!(matches!(
            &segments[0],
            TypstDocumentSegment::TypstMarkup { .. }
        ));
        match &segments[1] {
            TypstDocumentSegment::FencedCodeBlock(block) => {
                assert_eq!(
                    block.info_string.as_deref(),
                    Some("rust path/to/lib.rs#L1-L3")
                );
                assert_eq!(&doc[block.content_range.clone()], "fn main() {}\n");
            }
            other => panic!("expected fenced code block, got {other:?}"),
        }
        assert!(matches!(
            &segments[2],
            TypstDocumentSegment::TypstMarkup { .. }
        ));
    }

    #[test]
    fn denies_arbitrary_filesystem_reads() {
        let catalog = TypstFontCatalog::base();
        let world =
            ZedTypstWorld::new("", "#read(\"/etc/passwd\")", catalog).expect("world creation");
        let compiled = typst::compile::<typst_layout::PagedDocument>(&world);
        assert!(
            compiled.output.is_err(),
            "expected hermetic world to deny #read(\"/etc/passwd\")"
        );
    }

    struct TestTypstView {
        segment: Arc<PreparedTypstSegment>,
    }

    impl Render for TestTypstView {
        fn render(
            &mut self,
            _window: &mut Window,
            _cx: &mut gpui::Context<Self>,
        ) -> impl gpui::IntoElement {
            div().w(px(500.0)).child(TypstFrameElement::new(
                self.segment.clone(),
                TypstRenderStyle {
                    page_width_pt: Some(500.0),
                    ..TypstRenderStyle::default()
                },
            ))
        }
    }

    #[gpui::test]
    fn renders_typst_glyphs_and_shapes_in_gpui_window(cx: &mut TestAppContext) {
        let source = "= Native Typst\n\n$ -> \"Mind LLM\" -> $\n\n#table(columns: 2, [A], [B])\n\n#circle(radius: 8pt, fill: blue)";
        let segment = cx.update(|cx| {
            let catalog = TypstFontCatalog::global_or_init(cx);
            let mut inline_links: HashMap<Range<usize>, gpui::SharedString> = HashMap::default();
            let _ = &mut inline_links;
            Arc::new(PreparedTypstSegment::new(
                source,
                0..source.len(),
                &inline_links,
                catalog,
            ))
        });

        let (_view, cx) = cx.add_window_view(|_window, _cx| TestTypstView {
            segment: segment.clone(),
        });
        cx.simulate_resize(size(px(600.0), px(400.0)));
        cx.run_until_parked();

        cx.update(|_window, cx| {
            let world = ZedTypstWorld::new("", "Math $x$", TypstFontCatalog::base())
                .expect("world creation");
            let doc = typst::compile::<typst_layout::PagedDocument>(&world)
                .output
                .expect("document compiles");
            let mut found_font = false;
            for (_, item) in doc.pages()[0].frame.items() {
                if let typst::layout::FrameItem::Text(text_item) = item {
                    let font_id = resolve_gpui_font_id(cx.text_system(), &text_item.font)
                        .expect("register_font_face succeeds on platform text system");
                    let _ = font_id;
                    found_font = true;
                }
            }
            assert!(found_font);
        });
    }

    #[test]
    fn compiles_real_world_agent_typst_tables_and_prose_hazards() {
        let catalog = TypstFontCatalog::base();
        let source = "\
Right, it didn't fire. It was cancelled at 04:15:58, the moment your “how did you set the timer?” message arrived. The log says `Timer cancelled: context done.`

= Gemini has reset

I checked manually:

#table(columns: 3,
[Group], [_5-hour limit_], [_Weekly limit_],
[Gemini], [_100% left_, next reset 09:31], [56% left, resets Thu Nov 20, 18:31],
[Claude / GPT], [70% left, next reset 09:05], [60% left, resets Wed Nov 19, 20:00],
)

The Claude/GPT weekly quota has kept dropping fast: from $72 to $60 in about 25 minutes ($100 cap).
See @aaro, PR #1234, C#, #[gpui::test], user@example.com, or type @ to include context.

#table(columns: 2,
[1) Unmatched paren :(], [PR #42 & @aaro & **bold**],
)";
        let segment =
            PreparedTypstSegment::new(source, 0..source.len(), &HashMap::default(), catalog);
        let layout = segment.layout_for_style(&TypstRenderStyle {
            page_width_pt: Some(520.0),
            ..TypstRenderStyle::default()
        });
        let rendered = layout.rendered_text();

        assert!(
            !rendered.contains("#table"),
            "expected #table to render natively instead of plain-text fallback, got {rendered:?}"
        );
        assert!(
            !rendered.contains("= Gemini has reset"),
            "expected heading to render natively instead of plain-text fallback, got {rendered:?}"
        );
        assert!(rendered.contains("Gemini has reset"));
        assert!(rendered.contains("5-hour limit"));
        assert!(rendered.contains("@aaro"));
        assert!(rendered.contains("#1234"));
        assert!(rendered.contains("$72"));
        assert!(rendered.contains("$60"));
        assert!(rendered.contains("$100"));
        assert!(rendered.contains("#42"));
    }

    #[test]
    fn compiles_legacy_markdown_headings_lists_and_rules_without_fallback() {
        let catalog = TypstFontCatalog::base();
        let source = "\
### Quota Limits
- **Gemini Models**
  *(Models within this group: Gemini 4 Carbon, Gemini 4 Argon)*
  - **Weekly Limit Remaining:** 56% remaining
* Bullet item with leading star
---
After rule and math $2x + 1 = 5$.";
        let segment =
            PreparedTypstSegment::new(source, 0..source.len(), &HashMap::default(), catalog);
        let layout = segment.layout_for_style(&TypstRenderStyle {
            page_width_pt: Some(520.0),
            ..TypstRenderStyle::default()
        });
        let rendered = layout.rendered_text();

        assert!(!rendered.contains("###"));
        assert!(!rendered.contains("**Gemini"));
        assert!(rendered.contains("Quota Limits"));
        assert!(rendered.contains("Gemini Models"));
        assert!(rendered.contains("Bullet item with leading star"));
    }

    struct LocalFontsAssetSource(std::path::PathBuf);

    impl gpui::AssetSource for LocalFontsAssetSource {
        fn load(&self, path: &str) -> anyhow::Result<Option<std::borrow::Cow<'static, [u8]>>> {
            let full = self.0.join(path);
            match std::fs::read(full) {
                Ok(bytes) => Ok(Some(std::borrow::Cow::Owned(bytes))),
                Err(_) => Ok(None),
            }
        }

        fn list(&self, path: &str) -> anyhow::Result<Vec<gpui::SharedString>> {
            let mut results = Vec::new();
            let root = self.0.join(path);
            fn walk(
                base: &std::path::Path,
                dir: &std::path::Path,
                out: &mut Vec<gpui::SharedString>,
            ) {
                let Ok(entries) = std::fs::read_dir(dir) else {
                    return;
                };
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_dir() {
                        walk(base, &p, out);
                    } else if let Ok(rel) = p.strip_prefix(base) {
                        out.push(gpui::SharedString::from(
                            rel.to_string_lossy().replace('\\', "/"),
                        ));
                    }
                }
            }
            walk(&self.0, &root, &mut results);
            Ok(results)
        }
    }

    fn collect_text_items<'a>(
        frame: &'a typst::layout::Frame,
        out: &mut Vec<&'a typst::text::TextItem>,
    ) {
        for (_, item) in frame.items() {
            match item {
                typst::layout::FrameItem::Text(text_item) => out.push(text_item),
                typst::layout::FrameItem::Group(group) => collect_text_items(&group.frame, out),
                _ => {}
            }
        }
    }

    fn collect_shapes<'a>(
        frame: &'a typst::layout::Frame,
        parent_offset: typst::layout::Point,
        out: &mut Vec<(typst::layout::Point, &'a typst::visualize::Shape)>,
    ) {
        for (pos, item) in frame.items() {
            let abs_pos =
                typst::layout::Point::new(parent_offset.x + pos.x, parent_offset.y + pos.y);
            match item {
                typst::layout::FrameItem::Shape(shape, _) => out.push((abs_pos, shape)),
                typst::layout::FrameItem::Group(group) => {
                    collect_shapes(&group.frame, abs_pos, out)
                }
                _ => {}
            }
        }
    }

    #[test]
    fn resolves_ibm_plex_semibold_for_bold_and_headings_and_aligns_inline_code_chips_like_markdown()
    {
        let assets_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets");
        let asset_source = LocalFontsAssetSource(assets_dir);
        let catalog = Arc::new(TypstFontCatalog::build(Some(&asset_source)));

        let style = TypstRenderStyle {
            base_font_size_pt: 22.0,
            code_font_size_pt: 18.0,
            line_height_pt: 28.6,
            paragraph_spacing_pt: 8.0,
            heading_font_sizes_pt: [26.8, 25.1, 23.5, 22.0, 20.9, 19.8],
            page_width_pt: Some(640.0),
            ..TypstRenderStyle::default()
        };

        let raw = "== HeadingTitle\n\nPlainWord *BoldWord `BoldCode`* `InlineCode` and ~\\$8.50 with ~~OldText~~.";
        let (normalized, _) = preprocess_typst_segment(raw, 0, &HashMap::default());
        let world = ZedTypstWorld::new(&style.build_preamble(), &normalized, catalog.clone())
            .expect("world creation");
        let doc = typst::compile::<typst_layout::PagedDocument>(&world)
            .output
            .expect("typst compilation succeeds");

        let mut text_items = Vec::new();
        collect_text_items(&doc.pages()[0].frame, &mut text_items);

        let heading_item = text_items
            .iter()
            .find(|item| item.text.contains("HeadingTitle"))
            .expect("heading text item");
        assert!(
            heading_item.font.font().info().variant.weight.to_number() >= 600,
            "expected heading in IBM Plex Sans to use SemiBold (>=600), got {:?}",
            heading_item.font.font().info()
        );

        let plain_item = text_items
            .iter()
            .find(|item| item.text.contains("PlainWord"))
            .expect("plain text item");
        assert_eq!(
            plain_item.font.font().info().variant.weight.to_number(),
            400
        );

        let bold_item = text_items
            .iter()
            .find(|item| item.text.contains("BoldWord"))
            .expect("bold text item");
        assert!(
            bold_item.font.font().info().variant.weight.to_number() >= 600,
            "expected *BoldWord* in IBM Plex Sans to use SemiBold (>=600), got {:?}",
            bold_item.font.font().info()
        );

        let bold_code_item = text_items
            .iter()
            .find(|item| item.text.contains("BoldCode"))
            .expect("inline code inside bold");
        assert_eq!(
            bold_code_item.font.font().info().variant.weight.to_number(),
            400,
            "expected inline code inside *...* to stay regular weight (400) like Zed Markdown"
        );

        let code_item = text_items
            .iter()
            .find(|item| item.text.contains("InlineCode"))
            .expect("inline code text item");
        let code_pt = code_item.size.to_pt();
        assert!(
            (code_pt - 22.0).abs() < 0.05,
            "expected inline code size 22.0pt (1.0em matching surrounding prose like GPUI StyledText), got {code_pt:.3}pt"
        );

        let mut shapes = Vec::new();
        collect_shapes(
            &doc.pages()[0].frame,
            typst::layout::Point::zero(),
            &mut shapes,
        );
        let mut extracted_chip_quads = 0usize;
        for (_, shape) in &shapes {
            if let typst::visualize::Geometry::Curve(curve) = &shape.geometry
                && let Some((bounds, corners)) = try_extract_rounded_rect_quad(curve)
            {
                assert!(
                    (corners.top_left - 4.0).abs() < 0.05,
                    "expected 4pt chip corner radius, got {corners:?}"
                );
                assert!(
                    (bounds.size.height - 23.1).abs() < 0.5,
                    "expected symmetric ~23.1pt chip height at 22pt prose, got {:.2}pt",
                    bounds.size.height
                );
                extracted_chip_quads += 1;
            }
        }
        assert!(
            extracted_chip_quads >= 2,
            "expected inline code #highlight shapes to extract cleanly as GPU SDF quads, got {extracted_chip_quads}"
        );

        let segment = PreparedTypstSegment::new(raw, 0..raw.len(), &HashMap::default(), catalog);
        let compiled = segment.layout_for_style(&style);
        assert!(
            compiled.rendered_text().contains("~$8.50"),
            "expected tilde before currency to be preserved instead of swallowed as nbsp, got {:?}",
            compiled.rendered_text()
        );
        assert!(
            !compiled.rendered_text().contains("~~"),
            "expected ~~OldText~~ to convert to #strike, got {:?}",
            compiled.rendered_text()
        );
        assert!(compiled.rendered_text().contains("OldText"));
    }

    #[test]
    fn live_user_preamble_overrides_style_without_stacking_and_falls_back_on_syntax_error() {
        let catalog = TypstFontCatalog::base();
        let raw = "Hello `code_span` world.\n\nSecond paragraph.";
        let segment =
            PreparedTypstSegment::new(raw, 0..raw.len(), &HashMap::default(), catalog.clone());

        let base_style = TypstRenderStyle {
            base_font_size_pt: 20.0,
            line_height_pt: 26.0,
            page_width_pt: Some(400.0),
            ..TypstRenderStyle::default()
        };
        let default_layout = segment.layout_for_style(&base_style);

        let mut custom_style = base_style.clone();
        custom_style.user_preamble = Some(gpui::SharedString::from(
            "#set par(spacing: 40pt)\n#set highlight(radius: 2pt, extent: 3pt)\n",
        ));
        let overridden_layout = segment.layout_for_style(&custom_style);
        assert!(
            overridden_layout.size().height > default_layout.size().height,
            "expected live user_preamble override to invalidate cache and increase height ({:?} vs {:?})",
            overridden_layout.size(),
            default_layout.size()
        );

        let (normalized, _) =
            preprocess_typst_segment("Only `one_chip` here.", 0, &HashMap::default());
        let custom_world =
            ZedTypstWorld::new(&custom_style.build_preamble(), &normalized, catalog).unwrap();
        let custom_doc = typst::compile::<typst_layout::PagedDocument>(&custom_world)
            .output
            .unwrap();
        let mut custom_shapes = Vec::new();
        collect_shapes(
            &custom_doc.pages()[0].frame,
            typst::layout::Point::zero(),
            &mut custom_shapes,
        );
        assert_eq!(
            custom_shapes.len(),
            1,
            "expected #set highlight override in typst.typ to replace chip style rather than stacking duplicate shapes"
        );
        if let typst::visualize::Geometry::Curve(curve) = &custom_shapes[0].1.geometry {
            let (_, corners) = try_extract_rounded_rect_quad(curve).expect("rounded rect chip");
            assert!(
                (corners.top_left - 2.0).abs() < 0.05,
                "expected overridden 2pt radius from user_preamble, got {corners:?}"
            );
        } else {
            panic!("expected rounded rect Curve for inline code chip");
        }

        let mut broken_style = base_style;
        broken_style.user_preamble = Some(gpui::SharedString::from(
            "#show raw.where(block: false): highlight.with(radius: \n",
        ));
        let recovered_layout = segment.layout_for_style(&broken_style);
        assert!(
            recovered_layout.rendered_text().contains("code_span"),
            "expected syntax error in user typst.typ to fall back cleanly to built-in preamble, got {:?}",
            recovered_layout.rendered_text()
        );
        assert_eq!(recovered_layout.size(), default_layout.size());
    }
}
