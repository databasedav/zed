mod element;
mod preprocess;
mod world;

pub use element::{
    CompiledTypstLayout, PreparedTypstSegment, TypstCompileDiagnostic, TypstFrameElement,
    TypstGlyphCluster, TypstLinkRegion, format_typst_diagnostics, try_extract_rounded_rect_quad,
    validate_typst_document,
};
pub use preprocess::{
    OffsetMap, TypstCodeBlock, TypstDocumentSegment, TypstInlineCodeSpan, convert_math_to_typst,
    escape_typst_plain_text, extract_inline_code_spans, looks_like_latex_math,
    preprocess_typst_segment, preprocess_typst_segment_strict, segment_typst_document,
};
pub use typst;
pub use typst_layout;
pub use typst_syntax;
pub use world::{
    TypstFontCatalog, TypstRenderStyle, ZedTypstWorld, hsla_to_typst_hex, is_library_math_symbol,
    resolve_gpui_font_id, set_user_typst_preamble, user_typst_preamble,
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
            (code_pt - 19.8).abs() < 0.05,
            "expected inline code size 19.8pt (0.90 * zed-ui-size), got {code_pt:.3}pt"
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
                    (bounds.size.height - 20.0).abs() < 0.5,
                    "expected symmetric ~20.0pt chip height at 22pt prose, got {:.2}pt",
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
            "#show raw.where(block: false): set text(size: 0.90 * zed-ui-size)\n#set par(spacing: 40pt)\n#set highlight(radius: 2pt, extent: 3pt)\n",
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
        let mut custom_text_items = Vec::new();
        collect_text_items(&custom_doc.pages()[0].frame, &mut custom_text_items);
        let custom_code_item = custom_text_items
            .iter()
            .find(|item| item.text.contains("one_chip"))
            .expect("inline code item in overridden doc");
        assert!(
            (custom_code_item.size.to_pt() - 18.0).abs() < 0.05,
            "expected user_preamble #show raw size override to resolve cleanly to 18.0pt without multiplying, got {:.3}pt",
            custom_code_item.size.to_pt()
        );
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

    #[test]
    fn unwraps_backticked_typst_links_and_preserves_explicit_line_range_links() {
        let catalog = TypstFontCatalog::base();
        let style = TypstRenderStyle {
            page_width_pt: Some(640.0),
            ..TypstRenderStyle::default()
        };

        let source = "\
#table(
  columns: 3,
  [*Artifact / Crate*], [*Capabilities*], [*Gates*],
  [`#link(\"file:///workspace/crates/crate-wire/src/crate_wire.rs\")[crate-wire]` & #link(\"file:///workspace/crates/crate-crypto/src/crate_crypto.rs#L10-L25\")[`crate_crypto.rs`]],
  [64 B `#[repr(C)]` Pod header, `$Delta^2 x / Delta t^2$` ($Delta^2 x / Delta t^2$)],
  [Doc placeholder `#link(\"https://...\")[label]` stays literal code],
)";

        let crypto_tick_start = source
            .find("`crate_crypto.rs`")
            .expect("inline code inside explicit link");
        let crypto_tick_end = crypto_tick_start + "`crate_crypto.rs`".len();

        let extracted_spans = extract_inline_code_spans(source, 0);
        assert!(
            !extracted_spans
                .iter()
                .any(|span| span.full_range == (crypto_tick_start..crypto_tick_end)),
            "expected extract_inline_code_spans to skip inline code already inside #link(...)"
        );

        let mut inline_code_links: HashMap<Range<usize>, gpui::SharedString> = HashMap::default();
        inline_code_links.insert(
            crypto_tick_start..crypto_tick_end,
            gpui::SharedString::from("file:///workspace/crates/crate-crypto/src/crate_crypto.rs"),
        );

        let (normalized, _) = preprocess_typst_segment(source, 0, &inline_code_links);
        let world = ZedTypstWorld::new(&style.build_preamble(), &normalized, catalog.clone())
            .expect("world creation");
        assert!(
            typst::compile::<typst_layout::PagedDocument>(&world)
                .output
                .is_ok(),
            "expected table with unwrapped links and explicit line-range link to compile directly: {normalized}"
        );

        let segment =
            PreparedTypstSegment::new(source, 0..source.len(), &inline_code_links, catalog);
        let layout = segment.layout_for_style(&style);
        let rendered = layout.rendered_text();

        assert!(
            !rendered.contains("#link(\"file:///workspace"),
            "expected backtick-wrapped real #link(...) to unwrap into a clickable link, got {rendered:?}"
        );
        assert!(
            rendered.contains("#link(\"https://...\")"),
            "expected documentation placeholder `#link(\"https://...\")[label]` to remain literal code, got {rendered:?}"
        );

        let link_urls: Vec<&str> = layout
            .links()
            .iter()
            .map(|region| region.destination_url.as_ref())
            .collect();
        assert!(
            link_urls.contains(&"file:///workspace/crates/crate-wire/src/crate_wire.rs"),
            "expected unwrapped crate-wire link destination in {link_urls:?}"
        );
        assert!(
            link_urls
                .contains(&"file:///workspace/crates/crate-crypto/src/crate_crypto.rs#L10-L25"),
            "expected explicit #L10-L25 link destination to survive without inline_code_links clobbering, got {link_urls:?}"
        );
        assert!(
            !link_urls.contains(&"file:///workspace/crates/crate-crypto/src/crate_crypto.rs"),
            "expected inline_code_links not to double-wrap `#link(...)[`crate_crypto.rs`]`, got {link_urls:?}"
        );
    }

    #[test]
    fn compiles_all_real_world_llm_typst_syntax_hazards_without_fallback() {
        let catalog = TypstFontCatalog::base();
        let style = TypstRenderStyle {
            page_width_pt: Some(640.0),
            ..TypstRenderStyle::default()
        };

        let source = "\
= Section Intro

#figure([Box], caption: [Flow]) <fig-flow>

See @fig-flow for `<timer_notification>` (<timer_notification>), <2% of headset owners, \"<11 ms reflex\", and <4 px column <-> sync.

+ *Mode 1 $leftrightarrow$ 2 & stripped LaTeX symbols:* $x leq y geq z neq 0$, $a pm b propto infty$.
+ *Nested raw ticks:* pretty-printed ` ```json {\"ok\": true} ``` ` code block and `` `double_tick` ``.
+ *Underscores & brackets:* #link(\"file:///workspace/src/parity.py\")[_parity_cancel], __init__, and [unclosed *star inside bracket].
+ * Coding-agent \"Stop-and-Ask\" triggers in `AGENTS.md`*:
  - Hard-requiring agents to *stop and ask you* before changing schemas.
+ * stationary Datum Rail = ATW Horizon Invariant:* the top horizontal bar *never moves or tilts*—mirroring 2 * 3 and width*height, while JSON * strictly at the outer HTTP edge* stays clean.
+ *Stray backtick math closer & mixed currency:* billing window $<= 5\"m\"`, `+ 90s` grace, $0 egress with $O(1)$ lookup at $0.015/GB ($/GB) and $1,500–$3,000.
+ *URLs & comment tokens:* Visit https://zed.dev/docs or file:///tmp/report.txt and never trip on bare // or /* in prose.

| Gate | Status |
| :--- | :---: |
| `crate-wire` | Passed |
";

        let (normalized, _) = preprocess_typst_segment(source, 0, &HashMap::default());
        let world = ZedTypstWorld::new(&style.build_preamble(), &normalized, catalog.clone())
            .expect("world creation");
        let compiled = typst::compile::<typst_layout::PagedDocument>(&world);
        assert!(
            compiled.output.is_ok(),
            "expected zero Typst compile errors without fallback: {:?}\nnormalized:\n{normalized}",
            compiled.output.err()
        );

        let segment =
            PreparedTypstSegment::new(source, 0..source.len(), &HashMap::default(), catalog);
        let layout = segment.layout_for_style(&style);
        let rendered = layout.rendered_text();

        assert!(rendered.contains("<2% of headset owners"));
        assert!(rendered.contains("<11 ms reflex"));
        assert!(
            rendered.contains('↔'),
            "expected $leftrightarrow$ arrow in {rendered:?}"
        );
        assert!(rendered.contains("_parity_cancel"));
        assert!(rendered.contains("stop and ask you"));
        assert!(rendered.contains("never moves or tilts"));
        assert!(rendered.contains("$0 egress"));
        assert!(rendered.contains("crate-wire"));
        assert!(rendered.contains("Passed"));
        assert!(
            !rendered.contains("| :--- |"),
            "expected Markdown pipe table to convert to native #table, got {rendered:?}"
        );

        let escaped = escape_typst_plain_text("file:///tmp/a//b/*c*/");
        let fallback_world =
            ZedTypstWorld::new(&style.build_preamble(), &escaped, TypstFontCatalog::base())
                .expect("fallback world");
        let fallback_doc = typst::compile::<typst_layout::PagedDocument>(&fallback_world)
            .output
            .expect("escaped plain-text fallback compiles");
        let mut fallback_items = Vec::new();
        collect_text_items(&fallback_doc.pages()[0].frame, &mut fallback_items);
        let fallback_rendered: String = fallback_items
            .iter()
            .map(|item| item.text.as_str())
            .collect();
        assert_eq!(fallback_rendered, "file:///tmp/a//b/*c*/");
    }

    #[test]
    fn compiles_digit_led_degree_math_inside_tables_and_prose() {
        let source = "\
= Why It Generalizes: Foundation Priors + Per-Scene Inverse Graphics

1. *Test-time inverse-graphics optimization:*
   - Solver: $ min_theta sum_j w_j norm(Pi_K (\"FK\"_j (theta_t)) - x_(t, j))^2 + lambda_\"bone\" norm(Delta L)^2 $

= Where Iterating on One Case _Can_ Still Overfit

Even without training weights, a 2D shoulder-width ratio heuristic mistook crossed arms for a $135 degree$ back-spin.

#table(
  columns: 2,
  [*Guardrail*], [*How It Prevents Single-Case Overfitting*],
  [*3-Domain Inner Loop*], [Every iteration improves #link(\"file:///tmp/09_solo_dance.mp4\")[`09_solo_dance.mp4`] at $+1.0$.],
  [*Off-Axis 3D Anti-Cheat*], [`eval_recreation.py` scores both `Source Camera` *and* $plus.minus 45 degree$ and $90 degree$ orbited views.],
)

= Live Parallel Status

All three background engineering lanes (#link(\"file:///tmp/demo_voice_ai.rs#L151-L176\")[`ExpressionWeights`]) are running at `90 Hz`.";

        let diags = validate_typst_document(source);
        assert!(
            diags.is_empty(),
            "expected zero diagnostics on digit-led degree math + #table, got: {diags:?}"
        );

        let catalog = TypstFontCatalog::base();
        let segment =
            PreparedTypstSegment::new(source, 0..source.len(), &HashMap::default(), catalog);
        let layout = segment.layout_for_style(&TypstRenderStyle {
            page_width_pt: Some(640.0),
            ..TypstRenderStyle::default()
        });
        let rendered = layout.rendered_text();
        assert!(
            rendered.contains("135°"),
            "expected 135° in rendered output, got {rendered:?}"
        );
        assert!(
            rendered.contains("90°"),
            "expected 90° in rendered output, got {rendered:?}"
        );
        assert!(
            !rendered.contains("#table("),
            "expected #table(...) to render natively instead of plain text, got {rendered:?}"
        );
        assert_eq!(layout.links().len(), 2);
    }

    #[test]
    fn isolates_mid_document_table_syntax_errors_to_offending_block() {
        let source = "\
= Valid Intro Heading

Intro math $ x_i <= y_i $ and #link(\"file:///tmp/intro.rs\")[`intro.rs`].

#table(
  columns: 2,
  [*Header A*], [*Header B*],
  [#nonexistent_typst_fn()], [Broken cell],
)

= Valid Trailing Section

Trailing #link(\"file:///tmp/trailing.rs\")[`trailing.rs`] and *bold status* still render.";

        let diags = validate_typst_document(source);
        assert!(
            !diags.is_empty(),
            "strict validator must report #nonexistent_typst_fn()"
        );

        let catalog = TypstFontCatalog::base();
        let segment =
            PreparedTypstSegment::new(source, 0..source.len(), &HashMap::default(), catalog);
        let layout = segment.layout_for_style(&TypstRenderStyle {
            page_width_pt: Some(640.0),
            ..TypstRenderStyle::default()
        });
        let rendered = layout.rendered_text();
        assert!(
            !rendered.contains("= Valid Intro Heading"),
            "intro heading must remain rich Typst instead of escaped '= ', got {rendered:?}"
        );
        assert!(
            !rendered.contains("= Valid Trailing Section"),
            "trailing heading must remain rich Typst instead of escaped '= ', got {rendered:?}"
        );
        assert!(
            rendered.contains("#nonexistent_typst_fn()"),
            "broken table block should fall back cleanly to plain text, got {rendered:?}"
        );
        let link_urls: Vec<&str> = layout
            .links()
            .iter()
            .map(|link| link.destination_url.as_ref())
            .collect();
        assert_eq!(
            link_urls,
            vec!["file:///tmp/intro.rs", "file:///tmp/trailing.rs"]
        );
    }

    #[test]
    fn strict_validator_catches_unclosed_delimiters_and_maps_line_numbers() {
        let valid_doc = "= Clean Output\n\nMath $2 times 25 = 50$ and prose $100.\n\n```rust\nfn broken_rust_inside_fence( {\n```\n";
        assert!(validate_typst_document(valid_doc).is_empty());

        let unclosed_table = "= Heading\n\nParagraph.\n\n#table(\n  columns: 2,\n  [A], [B]\n";
        let diags = validate_typst_document(unclosed_table);
        assert!(
            !diags.is_empty(),
            "expected strict validator to catch unclosed #table("
        );
        let formatted = format_typst_diagnostics(&diags);
        assert!(formatted.contains("Line "), "got formatted: {formatted}");

        let bad_math = "= Heading\n\nInline $1 + unknown_symbol_xyz$ fails.";
        let math_diags = validate_typst_document(bad_math);
        assert_eq!(math_diags.len(), 1);
        assert_eq!(math_diags[0].line, 3);
        assert!(math_diags[0].message.contains("unknown variable"));

        let unclosed_fence = "= Heading\n\n```rust\nlet x = 1;\n";
        let fence_diags = validate_typst_document(unclosed_fence);
        assert_eq!(fence_diags[0].line, 3);
        assert!(
            fence_diags[0]
                .message
                .contains("unclosed fenced code block")
        );
    }
}
