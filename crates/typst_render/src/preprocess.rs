use std::ops::Range;

use gpui::SharedString;

#[derive(Clone, Debug, Default)]
pub struct OffsetMap {
    entries: Vec<OffsetEntry>,
    source_end: usize,
}

#[derive(Clone, Copy, Debug)]
struct OffsetEntry {
    normalized_start: usize,
    normalized_end: usize,
    source_start: usize,
    source_end: usize,
}

impl OffsetMap {
    pub fn identity(base_offset: usize, byte_len: usize) -> Self {
        let mut map = Self {
            entries: Vec::with_capacity(1),
            source_end: base_offset + byte_len,
        };
        if byte_len > 0 {
            map.entries.push(OffsetEntry {
                normalized_start: 0,
                normalized_end: byte_len,
                source_start: base_offset,
                source_end: base_offset + byte_len,
            });
        }
        map
    }

    pub fn map_to_source(&self, normalized_offset: usize) -> usize {
        let index = self
            .entries
            .partition_point(|entry| entry.normalized_end <= normalized_offset);
        let Some(entry) = self.entries.get(index) else {
            return self.source_end;
        };
        if normalized_offset <= entry.normalized_start {
            return entry.source_start;
        }
        let norm_span = entry.normalized_end.saturating_sub(entry.normalized_start);
        let source_span = entry.source_end.saturating_sub(entry.source_start);
        if norm_span == source_span {
            entry.source_start + (normalized_offset - entry.normalized_start)
        } else {
            entry.source_start
        }
    }

    pub fn map_range_to_source(&self, range: Range<usize>) -> Range<usize> {
        let start = self.map_to_source(range.start);
        let end = self.map_to_source(range.end).max(start);
        start..end
    }

    pub fn source_end(&self) -> usize {
        self.source_end
    }
}

struct NormalizedBuilder {
    text: String,
    entries: Vec<OffsetEntry>,
    base_offset: usize,
    source_len: usize,
}

impl NormalizedBuilder {
    fn new(base_offset: usize, source_len: usize) -> Self {
        Self {
            text: String::with_capacity(source_len + 32),
            entries: Vec::new(),
            base_offset,
            source_len,
        }
    }

    fn push_passthrough(&mut self, slice: &str, local_start: usize) {
        if slice.is_empty() {
            return;
        }
        let normalized_start = self.text.len();
        self.text.push_str(slice);
        let normalized_end = self.text.len();
        let source_start = self.base_offset + local_start;
        let source_end = source_start + slice.len();

        if let Some(last) = self.entries.last_mut() {
            let last_is_identity = (last.normalized_end - last.normalized_start)
                == (last.source_end - last.source_start);
            if last_is_identity
                && last.normalized_end == normalized_start
                && last.source_end == source_start
            {
                last.normalized_end = normalized_end;
                last.source_end = source_end;
                return;
            }
        }

        self.entries.push(OffsetEntry {
            normalized_start,
            normalized_end,
            source_start,
            source_end,
        });
    }

    fn push_replaced(&mut self, replacement: &str, local_range: Range<usize>) {
        if replacement.is_empty() {
            return;
        }
        let normalized_start = self.text.len();
        self.text.push_str(replacement);
        let normalized_end = self.text.len();
        self.entries.push(OffsetEntry {
            normalized_start,
            normalized_end,
            source_start: self.base_offset + local_range.start,
            source_end: self.base_offset + local_range.end,
        });
    }

    fn finish(self) -> (String, OffsetMap) {
        (
            self.text,
            OffsetMap {
                entries: self.entries,
                source_end: self.base_offset + self.source_len,
            },
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypstCodeBlock {
    pub full_range: Range<usize>,
    pub content_range: Range<usize>,
    pub info_string: Option<SharedString>,
    pub is_unclosed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypstInlineCodeSpan {
    pub full_range: Range<usize>,
    pub content_range: Range<usize>,
    pub text: SharedString,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypstDocumentSegment {
    TypstMarkup { source_range: Range<usize> },
    FencedCodeBlock(TypstCodeBlock),
}

pub fn segment_typst_document(source: &str) -> Vec<TypstDocumentSegment> {
    let mut segments = Vec::new();
    let bytes = source.as_bytes();
    let total_len = bytes.len();
    let mut markup_start = 0;
    let mut cursor = 0;

    while cursor < total_len {
        let line_start = cursor;
        let mut indent = cursor;
        while indent < total_len && (bytes[indent] == b' ' || bytes[indent] == b'\t') {
            indent += 1;
        }

        let fence_char = match bytes.get(indent).copied() {
            Some(b'`') | Some(b'~') => bytes[indent],
            _ => {
                cursor = next_line_start(bytes, cursor);
                continue;
            }
        };

        let mut fence_end = indent;
        while fence_end < total_len && bytes[fence_end] == fence_char {
            fence_end += 1;
        }
        let fence_len = fence_end - indent;
        if fence_len < 3 {
            cursor = next_line_start(bytes, cursor);
            continue;
        }

        let info_line_end = find_line_end(bytes, fence_end);
        let raw_info = &source[fence_end..info_line_end];
        if fence_char == b'`' && raw_info.contains('`') {
            cursor = next_line_start(bytes, cursor);
            continue;
        }
        let trimmed_info = raw_info.trim();
        let info_string =
            (!trimmed_info.is_empty()).then(|| SharedString::from(trimmed_info.to_string()));
        let content_start = next_line_start(bytes, fence_end);

        let mut scan = content_start;
        let mut closing_fence_line_start = None;
        let mut block_end = total_len;
        while scan < total_len {
            let candidate_line_start = scan;
            let mut candidate_indent = scan;
            while candidate_indent < total_len
                && (bytes[candidate_indent] == b' ' || bytes[candidate_indent] == b'\t')
            {
                candidate_indent += 1;
            }
            let mut close_end = candidate_indent;
            while close_end < total_len && bytes[close_end] == fence_char {
                close_end += 1;
            }
            if close_end - candidate_indent >= fence_len {
                let close_line_end = find_line_end(bytes, close_end);
                if source[close_end..close_line_end].trim().is_empty() {
                    closing_fence_line_start = Some(candidate_line_start);
                    block_end = next_line_start(bytes, close_end);
                    break;
                }
            }
            scan = next_line_start(bytes, scan);
        }

        let is_native_typst_eval_fence = matches!(
            trimmed_info.split_whitespace().next(),
            Some("typst-render" | "typ-render")
        );

        if is_native_typst_eval_fence {
            if markup_start < line_start && !source[markup_start..line_start].trim().is_empty() {
                segments.push(TypstDocumentSegment::TypstMarkup {
                    source_range: markup_start..line_start,
                });
            }
            let inner_end = closing_fence_line_start.unwrap_or(total_len);
            if content_start < inner_end {
                segments.push(TypstDocumentSegment::TypstMarkup {
                    source_range: content_start..inner_end,
                });
            }
            markup_start = block_end;
            cursor = block_end;
            continue;
        }

        if markup_start < line_start && !source[markup_start..line_start].trim().is_empty() {
            segments.push(TypstDocumentSegment::TypstMarkup {
                source_range: markup_start..line_start,
            });
        }

        let content_end = closing_fence_line_start.unwrap_or(total_len);
        segments.push(TypstDocumentSegment::FencedCodeBlock(TypstCodeBlock {
            full_range: line_start..block_end,
            content_range: content_start..content_end,
            info_string,
            is_unclosed: closing_fence_line_start.is_none(),
        }));

        markup_start = block_end;
        cursor = block_end;
    }

    if markup_start < total_len && !source[markup_start..total_len].trim().is_empty() {
        segments.push(TypstDocumentSegment::TypstMarkup {
            source_range: markup_start..total_len,
        });
    }

    segments
}

pub fn extract_inline_code_spans(
    segment_text: &str,
    segment_base_offset: usize,
) -> Vec<TypstInlineCodeSpan> {
    let mut spans = Vec::new();
    let bytes = segment_text.as_bytes();
    let len = bytes.len();
    let mut index = 0;
    let mut mode_stack: Vec<PreprocessMode> = Vec::new();

    while index < len {
        let in_markup = mode_stack.last().is_none_or(|mode| {
            matches!(
                *mode,
                PreprocessMode::MarkupBracket { .. } | PreprocessMode::LinkBodyBracket { .. }
            )
        });
        let inside_explicit_link = mode_stack
            .iter()
            .any(|mode| matches!(mode, PreprocessMode::LinkBodyBracket { .. }));

        match bytes[index] {
            b'\\' => {
                index += next_escaped_step(segment_text, index);
            }
            b'"' if !in_markup => {
                let (str_end, _) = scan_typst_string(bytes, index);
                index = str_end;
            }
            b'$' if in_markup => {
                if is_prose_currency_dollar(segment_text, index) {
                    index += 1;
                    continue;
                }
                let (math_end, _, _, _) = scan_dollar_math(bytes, index);
                index = math_end;
            }
            b'[' if in_markup => {
                let prev_is_ident_or_call = index > 0
                    && matches!(
                        bytes[index - 1],
                        b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'-' | b')' | b']'
                    );
                if !prev_is_ident_or_call
                    && let Some((_, _, full_end)) = try_parse_markdown_link(segment_text, index)
                {
                    index = full_end;
                    continue;
                }
                if matches!(
                    mode_stack.last(),
                    Some(
                        PreprocessMode::MarkupBracket { .. }
                            | PreprocessMode::LinkBodyBracket { .. }
                    )
                ) {
                    mode_stack.push(PreprocessMode::MarkupBracket {
                        emph_open: false,
                        strong_open: false,
                    });
                }
                index += 1;
            }
            b']' if in_markup
                && matches!(
                    mode_stack.last(),
                    Some(
                        PreprocessMode::MarkupBracket { .. }
                            | PreprocessMode::LinkBodyBracket { .. }
                    )
                ) =>
            {
                mode_stack.pop();
                index += 1;
            }
            b'#' => {
                if let Some(after_ident) = scan_typst_hash_expr(segment_text, index + 1) {
                    let ident = &segment_text[(index + 1)..after_ident];
                    index = after_ident;
                    if bytes.get(index) == Some(&b'(') {
                        if ident == "link" {
                            mode_stack.push(PreprocessMode::LinkArgsParen);
                        } else {
                            mode_stack.push(PreprocessMode::CodeParen);
                        }
                        index += 1;
                    } else if bytes.get(index) == Some(&b'{') {
                        mode_stack.push(PreprocessMode::CodeBrace);
                        index += 1;
                    } else if bytes.get(index) == Some(&b'[') {
                        if ident == "link" {
                            mode_stack.push(PreprocessMode::LinkBodyBracket {
                                emph_open: false,
                                strong_open: false,
                            });
                        } else {
                            mode_stack.push(PreprocessMode::MarkupBracket {
                                emph_open: false,
                                strong_open: false,
                            });
                        }
                        index += 1;
                    }
                    continue;
                }
                if bytes.get(index + 1) == Some(&b'{') {
                    mode_stack.push(PreprocessMode::CodeBrace);
                    index += 2;
                    continue;
                }
                if bytes.get(index + 1) == Some(&b'(') {
                    mode_stack.push(PreprocessMode::CodeParen);
                    index += 2;
                    continue;
                }
                index += 1;
            }
            b'`' => {
                let start = index;
                let mut tick_count = 0;
                while index + tick_count < len && bytes[index + tick_count] == b'`' {
                    tick_count += 1;
                }
                let (raw_end, closed) = scan_inline_backticks(bytes, start);
                if closed {
                    let content_start = start + tick_count;
                    let content_end = raw_end - tick_count;
                    let inner = &segment_text[content_start..content_end];
                    let is_backticked_link =
                        tick_count == 1 && try_unwrap_backticked_typst_link(inner).is_some();
                    if in_markup
                        && !inside_explicit_link
                        && !is_backticked_link
                        && !inner.contains('\n')
                        && !inner.is_empty()
                    {
                        spans.push(TypstInlineCodeSpan {
                            full_range: (segment_base_offset + start)
                                ..(segment_base_offset + raw_end),
                            content_range: (segment_base_offset + content_start)
                                ..(segment_base_offset + content_end),
                            text: SharedString::from(inner.to_string()),
                        });
                    }
                }
                index = raw_end;
            }
            byte if !in_markup => {
                match byte {
                    b'(' => mode_stack.push(PreprocessMode::CodeParen),
                    b')' => {
                        if mode_stack.last() == Some(&PreprocessMode::LinkArgsParen) {
                            mode_stack.pop();
                            index += 1;
                            if bytes.get(index) == Some(&b'[') {
                                mode_stack.push(PreprocessMode::LinkBodyBracket {
                                    emph_open: false,
                                    strong_open: false,
                                });
                                index += 1;
                            }
                            continue;
                        }
                        if mode_stack.last() == Some(&PreprocessMode::CodeParen) {
                            mode_stack.pop();
                        }
                    }
                    b'{' => mode_stack.push(PreprocessMode::CodeBrace),
                    b'}' => {
                        if mode_stack.last() == Some(&PreprocessMode::CodeBrace) {
                            mode_stack.pop();
                        }
                    }
                    b'[' => {
                        if mode_stack.last() == Some(&PreprocessMode::LinkArgsParen) {
                            mode_stack.push(PreprocessMode::LinkBodyBracket {
                                emph_open: false,
                                strong_open: false,
                            });
                        } else {
                            mode_stack.push(PreprocessMode::MarkupBracket {
                                emph_open: false,
                                strong_open: false,
                            });
                        }
                    }
                    _ => {}
                }
                index += 1;
            }
            _ => {
                if let Some(character) = segment_text
                    .get(index..)
                    .and_then(|rest| rest.chars().next())
                {
                    index += character.len_utf8();
                } else {
                    index += 1;
                }
            }
        }
    }

    spans
}

pub fn looks_like_latex_math(math_content: &str) -> bool {
    let trimmed = math_content.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.contains("^{") || trimmed.contains("_{") {
        return true;
    }
    let bytes = trimmed.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            let cmd_start = index + 1;
            let mut cmd_end = cmd_start;
            while cmd_end < bytes.len() && bytes[cmd_end].is_ascii_alphabetic() {
                cmd_end += 1;
            }
            if cmd_end - cmd_start >= 2 {
                return true;
            }
            index = cmd_end.max(index + 1);
        } else {
            index += 1;
        }
    }
    false
}

pub fn convert_math_to_typst(math_content: &str, display_mode: bool) -> String {
    let trimmed = math_content.trim();
    if trimmed.is_empty() {
        return if display_mode {
            "$ $".to_string()
        } else {
            "$$".to_string()
        };
    }

    let body = if looks_like_latex_math(trimmed) {
        match mitex::convert_math(trimmed, None) {
            Ok(converted) => converted.trim().to_string(),
            Err(_) => escape_typst_math_fallback(trimmed),
        }
    } else {
        trimmed.to_string()
    };

    if display_mode {
        format!("$ {body} $")
    } else {
        format!("${body}$")
    }
}

fn escape_typst_math_fallback(raw: &str) -> String {
    let escaped = raw.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

pub fn escape_typst_plain_text(raw: &str) -> String {
    let mut escaped = String::with_capacity(raw.len() + 16);
    for character in raw.chars() {
        match character {
            '\\' | '#' | '$' | '*' | '_' | '`' | '[' | ']' | '<' | '>' | '@' | '=' | '~' | '/' => {
                escaped.push('\\');
                escaped.push(character);
            }
            _ => escaped.push(character),
        }
    }
    escaped
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreprocessMode {
    CodeParen,
    LinkArgsParen,
    CodeBrace,
    MarkupBracket { emph_open: bool, strong_open: bool },
    LinkBodyBracket { emph_open: bool, strong_open: bool },
}

pub fn preprocess_typst_segment(
    segment_text: &str,
    base_offset: usize,
    inline_code_links: &collections::HashMap<Range<usize>, SharedString>,
) -> (String, OffsetMap) {
    preprocess_typst_segment_inner(segment_text, base_offset, inline_code_links, false, true)
}

fn preprocess_typst_segment_inner(
    segment_text: &str,
    base_offset: usize,
    inline_code_links: &collections::HashMap<Range<usize>, SharedString>,
    inside_link_override: bool,
    apply_streaming_tail: bool,
) -> (String, OffsetMap) {
    let mut builder = NormalizedBuilder::new(base_offset, segment_text.len());
    let bytes = segment_text.as_bytes();
    let len = bytes.len();
    let mut cursor = 0;
    let mut at_line_start = true;
    let mut mode_stack: Vec<PreprocessMode> = Vec::new();
    let mut root_emph_open = false;
    let mut root_strong_open = false;

    while cursor < len {
        let in_markup = mode_stack.last().is_none_or(|mode| {
            matches!(
                *mode,
                PreprocessMode::MarkupBracket { .. } | PreprocessMode::LinkBodyBracket { .. }
            )
        });
        let inside_explicit_link = inside_link_override
            || mode_stack
                .iter()
                .any(|mode| matches!(mode, PreprocessMode::LinkBodyBracket { .. }));

        if in_markup && at_line_start {
            if mode_stack.is_empty()
                && let Some((consumed, table_replacement)) = try_convert_markdown_pipe_table(
                    segment_text,
                    cursor,
                    base_offset,
                    inline_code_links,
                )
            {
                builder.push_replaced(&table_replacement, cursor..(cursor + consumed));
                cursor += consumed;
                at_line_start = true;
                continue;
            }
            if let Some((consumed, replacement)) =
                try_normalize_markdown_line_prefix(segment_text, cursor)
            {
                builder.push_replaced(&replacement, cursor..(cursor + consumed));
                cursor += consumed;
                at_line_start = false;
                continue;
            }
        }

        let current_byte = bytes[cursor];

        if current_byte == b'\\' {
            if in_markup && bytes.get(cursor + 1) == Some(&b'(') {
                if let Some(close_pos) = find_closing_subslice(segment_text, cursor + 2, "\\)") {
                    let math_inner = &segment_text[(cursor + 2)..close_pos];
                    let converted = convert_math_to_typst(math_inner, false);
                    let end = close_pos + 2;
                    builder.push_replaced(&converted, cursor..end);
                    cursor = end;
                    at_line_start = false;
                    continue;
                }
            }
            if in_markup && bytes.get(cursor + 1) == Some(&b'[') {
                if let Some(close_pos) = find_closing_subslice(segment_text, cursor + 2, "\\]") {
                    let math_inner = &segment_text[(cursor + 2)..close_pos];
                    let converted = convert_math_to_typst(math_inner, true);
                    let end = close_pos + 2;
                    builder.push_replaced(&converted, cursor..end);
                    cursor = end;
                    at_line_start = false;
                    continue;
                }
            }

            let step = next_escaped_step(segment_text, cursor);
            builder.push_passthrough(&segment_text[cursor..(cursor + step)], cursor);
            at_line_start = segment_text[cursor..(cursor + step)].ends_with('\n');
            cursor += step;
            continue;
        }

        if current_byte == b'`' {
            let mut tick_count = 0;
            while cursor + tick_count < len && bytes[cursor + tick_count] == b'`' {
                tick_count += 1;
            }
            let (raw_end, closed) = scan_inline_backticks(bytes, cursor);
            let global_range = (base_offset + cursor)..(base_offset + raw_end);
            if in_markup && closed {
                let inner = &segment_text[(cursor + tick_count)..(raw_end - tick_count)];
                if tick_count == 1
                    && !inside_explicit_link
                    && let Some((raw_url, label)) = try_unwrap_backticked_typst_link(inner)
                {
                    let formatted_link = if !label.is_empty()
                        && !label.contains('`')
                        && !label.contains('#')
                        && !label.contains('*')
                        && !label.contains('$')
                        && !label.contains('[')
                        && !label.contains(']')
                    {
                        format!("#link(\"{raw_url}\")[`{label}`]")
                    } else {
                        let (normalized_label, _) = preprocess_typst_segment_inner(
                            label,
                            base_offset + cursor + 1,
                            inline_code_links,
                            true,
                            false,
                        );
                        format!("#link(\"{raw_url}\")[{normalized_label}]")
                    };
                    builder.push_replaced(&formatted_link, cursor..raw_end);
                    cursor = raw_end;
                    at_line_start = false;
                    continue;
                }

                let needs_raw_func =
                    !inner.contains('\n') && (tick_count >= 2 || inner.contains('`'));
                let code_expr = if needs_raw_func {
                    let trimmed_inner = if inner.len() >= 2
                        && inner.starts_with(' ')
                        && inner.ends_with(' ')
                        && inner.bytes().any(|byte| byte != b' ')
                    {
                        &inner[1..(inner.len() - 1)]
                    } else {
                        inner
                    };
                    Some(format!(
                        "#raw(\"{}\", block: false)",
                        escape_typst_string(trimmed_inner)
                    ))
                } else {
                    None
                };

                if !inside_explicit_link
                    && let Some(link_url) = inline_code_links.get(&global_range)
                {
                    let raw_body = code_expr
                        .as_deref()
                        .unwrap_or(&segment_text[cursor..raw_end]);
                    let escaped_url = escape_typst_string(link_url.as_ref());
                    let linked = format!("#link(\"{escaped_url}\")[{raw_body}]");
                    builder.push_replaced(&linked, cursor..raw_end);
                    cursor = raw_end;
                    at_line_start = false;
                    continue;
                }

                if let Some(raw_func_call) = code_expr {
                    builder.push_replaced(&raw_func_call, cursor..raw_end);
                    cursor = raw_end;
                    at_line_start = false;
                    continue;
                }
            }
            builder.push_passthrough(&segment_text[cursor..raw_end], cursor);
            if !closed {
                let closing_ticks = "`".repeat(tick_count);
                builder.push_replaced(&closing_ticks, raw_end..raw_end);
            }
            at_line_start = segment_text[cursor..raw_end].ends_with('\n');
            cursor = raw_end;
            continue;
        }

        if !in_markup && current_byte == b'"' {
            let (str_end, closed) = scan_typst_string(bytes, cursor);
            builder.push_passthrough(&segment_text[cursor..str_end], cursor);
            if !closed {
                builder.push_replaced("\"", str_end..str_end);
            }
            cursor = str_end;
            at_line_start = false;
            continue;
        }

        if in_markup && current_byte == b'$' {
            if is_prose_currency_dollar(segment_text, cursor) {
                builder.push_replaced("\\$", cursor..(cursor + 1));
                cursor += 1;
                at_line_start = false;
                continue;
            }

            let (math_end, is_double, closed, closed_by_backtick) = scan_dollar_math(bytes, cursor);
            let delim_len = if is_double { 2 } else { 1 };
            let close_len = if closed_by_backtick {
                1
            } else if closed {
                delim_len
            } else {
                0
            };
            let inner_end = math_end - close_len;
            let inner = &segment_text[(cursor + delim_len)..inner_end];

            let standalone_on_line = is_standalone_on_line(segment_text, cursor, math_end);
            let has_both_spaces = inner.starts_with(char::is_whitespace)
                && inner.ends_with(char::is_whitespace)
                && inner.trim().len() < inner.len();
            let display_mode = is_double || (has_both_spaces && standalone_on_line);

            if !is_double
                && closed
                && !closed_by_backtick
                && !looks_like_latex_math(inner)
                && (!has_both_spaces || standalone_on_line)
            {
                builder.push_passthrough(&segment_text[cursor..math_end], cursor);
            } else {
                let closed_inner = close_unbalanced_math_parens(inner);
                let converted = convert_math_to_typst(&closed_inner, display_mode);
                builder.push_replaced(&converted, cursor..math_end);
            }
            cursor = math_end;
            at_line_start = false;
            continue;
        }

        if in_markup && current_byte == b'<' {
            let label = scan_typst_label_ref(segment_text, cursor + 1);
            let closes_with_gt =
                !label.is_empty() && bytes.get(cursor + 1 + label.len()) == Some(&b'>');
            let has_matching_ref = closes_with_gt && segment_text.contains(&format!("@{label}"));
            if !has_matching_ref {
                builder.push_replaced("\\<", cursor..(cursor + 1));
                cursor += 1;
                at_line_start = false;
                continue;
            }
        }

        if in_markup && current_byte == b'@' {
            let label = scan_typst_label_ref(segment_text, cursor + 1);
            let has_matching_label =
                !label.is_empty() && segment_text.contains(&format!("<{label}>"));
            if !has_matching_label {
                builder.push_replaced("\\@", cursor..(cursor + 1));
                cursor += 1;
                at_line_start = false;
                continue;
            }
        }

        if in_markup
            && !inside_explicit_link
            && matches!(current_byte, b'h' | b'f')
            && (cursor == 0
                || matches!(
                    bytes[cursor - 1],
                    b' ' | b'\t' | b'\n' | b'\r' | b'(' | b'[' | b':' | b',' | b';'
                ))
            && let Some(url_end) = scan_bare_prose_url(segment_text, cursor)
        {
            let raw_url = &segment_text[cursor..url_end];
            let escaped_url = escape_typst_string(raw_url);
            let escaped_label = escape_typst_plain_text(raw_url);
            let linked = format!("#link(\"{escaped_url}\")[{escaped_label}]");
            builder.push_replaced(&linked, cursor..url_end);
            cursor = url_end;
            at_line_start = false;
            continue;
        }

        if in_markup && current_byte == b'/' {
            if bytes.get(cursor + 1) == Some(&b'/') {
                builder.push_replaced("/\\/", cursor..(cursor + 2));
                cursor += 2;
                at_line_start = false;
                continue;
            }
            if bytes.get(cursor + 1) == Some(&b'*') {
                builder.push_replaced("/\\*", cursor..(cursor + 2));
                cursor += 2;
                at_line_start = false;
                continue;
            }
        }

        if in_markup && current_byte == b'*' {
            if bytes.get(cursor + 1) == Some(&b'*') {
                if let Some(close_pos) =
                    find_markdown_double_delimiter_end(segment_text, cursor + 2, b'*')
                {
                    builder.push_replaced("*", cursor..(cursor + 2));
                    let inner = &segment_text[(cursor + 2)..close_pos];
                    let (normalized_inner, _) = preprocess_typst_segment_inner(
                        inner,
                        base_offset + cursor + 2,
                        inline_code_links,
                        inside_explicit_link,
                        false,
                    );
                    builder.push_replaced(&normalized_inner, (cursor + 2)..close_pos);
                    builder.push_replaced("*", close_pos..(close_pos + 2));
                    cursor = close_pos + 2;
                    at_line_start = false;
                    continue;
                }
            } else {
                let prev_char = segment_text[..cursor].chars().next_back();
                let next_char = segment_text
                    .get((cursor + 1)..)
                    .and_then(|rest| rest.chars().next());
                if is_typst_in_word(prev_char, next_char) {
                    builder.push_passthrough("*", cursor);
                } else {
                    let prev_ws = prev_char.is_none_or(char::is_whitespace);
                    let next_ws = next_char.is_none_or(char::is_whitespace);
                    let prev_wordy = prev_char.is_some_and(is_typst_word_char);
                    let can_close = !prev_ws;
                    let can_open = !next_ws
                        && !prev_wordy
                        && !matches!(prev_char, Some('`' | ')' | ']' | '}'));
                    let strong_open = current_strong_open(&mut mode_stack, &mut root_strong_open);
                    if *strong_open && can_close {
                        *strong_open = false;
                        builder.push_passthrough("*", cursor);
                    } else if !*strong_open
                        && can_open
                        && has_closing_delimiter_in_scope(segment_text, cursor + 1, b'*')
                    {
                        *strong_open = true;
                        builder.push_passthrough("*", cursor);
                    } else {
                        builder.push_replaced("\\*", cursor..(cursor + 1));
                    }
                }
                cursor += 1;
                at_line_start = false;
                continue;
            }
        }

        if in_markup && current_byte == b'_' {
            let prev_char = segment_text[..cursor].chars().next_back();
            let next_char = segment_text
                .get((cursor + 1)..)
                .and_then(|rest| rest.chars().next());
            if prev_char == Some('_') || next_char == Some('_') {
                builder.push_replaced("\\_", cursor..(cursor + 1));
            } else if is_typst_in_word(prev_char, next_char) {
                builder.push_passthrough("_", cursor);
            } else {
                let prev_wordy = prev_char.is_some_and(is_typst_word_char);
                let next_wordy = next_char.is_some_and(is_typst_word_char);
                let can_close =
                    prev_char.is_some_and(|character| !character.is_whitespace()) && !next_wordy;
                let can_open = next_char.is_some_and(|character| !character.is_whitespace())
                    && !prev_wordy
                    && !matches!(prev_char, Some('`' | ')' | ']' | '}'));
                let is_code_ident = is_leading_underscore_identifier(segment_text, cursor);
                let can_pair = !is_code_ident
                    && has_closing_delimiter_in_scope(segment_text, cursor + 1, b'_');
                let emph_open = current_emph_open(&mut mode_stack, &mut root_emph_open);
                if *emph_open && can_close {
                    *emph_open = false;
                    builder.push_passthrough("_", cursor);
                } else if !*emph_open && can_open && can_pair {
                    *emph_open = true;
                    builder.push_passthrough("_", cursor);
                } else {
                    builder.push_replaced("\\_", cursor..(cursor + 1));
                }
            }
            cursor += 1;
            at_line_start = false;
            continue;
        }

        if in_markup && current_byte == b'~' {
            if bytes.get(cursor + 1) == Some(&b'~') {
                if let Some(close_pos) =
                    find_markdown_double_delimiter_end(segment_text, cursor + 2, b'~')
                {
                    builder.push_replaced("#strike[", cursor..(cursor + 2));
                    let inner = &segment_text[(cursor + 2)..close_pos];
                    let (normalized_inner, _) = preprocess_typst_segment_inner(
                        inner,
                        base_offset + cursor + 2,
                        inline_code_links,
                        inside_explicit_link,
                        false,
                    );
                    builder.push_replaced(&normalized_inner, (cursor + 2)..close_pos);
                    builder.push_replaced("]", close_pos..(close_pos + 2));
                    cursor = close_pos + 2;
                    at_line_start = false;
                    continue;
                }
            }
            // In Typst markup, bare `~` is a non-breaking space (`\u{00A0}`), which
            // unexpectedly swallows approximate tildes like `~$8.50` or `~5ms`.
            builder.push_replaced("\\~", cursor..(cursor + 1));
            cursor += 1;
            at_line_start = false;
            continue;
        }

        if in_markup && current_byte == b'[' {
            let prev_is_ident_or_call = cursor > 0
                && matches!(
                    bytes[cursor - 1],
                    b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'-' | b')' | b']'
                );
            if !prev_is_ident_or_call {
                if let Some((label_range, url_str, full_end)) =
                    try_parse_markdown_link(segment_text, cursor)
                {
                    let escaped_url = escape_typst_string(&url_str);
                    let label_slice = &segment_text[label_range.clone()];
                    let (normalized_label, _) = preprocess_typst_segment_inner(
                        label_slice,
                        base_offset + label_range.start,
                        inline_code_links,
                        true,
                        false,
                    );
                    let replacement = format!("#link(\"{escaped_url}\")[{normalized_label}]");
                    builder.push_replaced(&replacement, cursor..full_end);
                    cursor = full_end;
                    at_line_start = false;
                    continue;
                }
            }
            if matches!(
                mode_stack.last(),
                Some(PreprocessMode::MarkupBracket { .. } | PreprocessMode::LinkBodyBracket { .. })
            ) {
                mode_stack.push(PreprocessMode::MarkupBracket {
                    emph_open: false,
                    strong_open: false,
                });
                builder.push_passthrough("[", cursor);
                cursor += 1;
                at_line_start = false;
                continue;
            }
        }

        if in_markup
            && current_byte == b']'
            && matches!(
                mode_stack.last(),
                Some(PreprocessMode::MarkupBracket { .. } | PreprocessMode::LinkBodyBracket { .. })
            )
        {
            mode_stack.pop();
            builder.push_passthrough("]", cursor);
            cursor += 1;
            at_line_start = false;
            continue;
        }

        if current_byte == b'#' {
            if let Some(after_ident) = scan_typst_hash_expr(segment_text, cursor + 1) {
                let ident = &segment_text[(cursor + 1)..after_ident];
                builder.push_passthrough(&segment_text[cursor..after_ident], cursor);
                cursor = after_ident;
                if bytes.get(cursor) == Some(&b'(') {
                    if ident == "link" {
                        mode_stack.push(PreprocessMode::LinkArgsParen);
                    } else {
                        mode_stack.push(PreprocessMode::CodeParen);
                    }
                    builder.push_passthrough("(", cursor);
                    cursor += 1;
                } else if bytes.get(cursor) == Some(&b'{') {
                    mode_stack.push(PreprocessMode::CodeBrace);
                    builder.push_passthrough("{", cursor);
                    cursor += 1;
                } else if bytes.get(cursor) == Some(&b'[') {
                    if ident == "link" {
                        mode_stack.push(PreprocessMode::LinkBodyBracket {
                            emph_open: false,
                            strong_open: false,
                        });
                    } else {
                        mode_stack.push(PreprocessMode::MarkupBracket {
                            emph_open: false,
                            strong_open: false,
                        });
                    }
                    builder.push_passthrough("[", cursor);
                    cursor += 1;
                }
                at_line_start = false;
                continue;
            }
            if bytes.get(cursor + 1) == Some(&b'{') {
                mode_stack.push(PreprocessMode::CodeBrace);
                builder.push_passthrough("#{", cursor);
                cursor += 2;
                at_line_start = false;
                continue;
            }
            if bytes.get(cursor + 1) == Some(&b'(') {
                mode_stack.push(PreprocessMode::CodeParen);
                builder.push_passthrough("#(", cursor);
                cursor += 2;
                at_line_start = false;
                continue;
            }
            if in_markup {
                builder.push_replaced("\\#", cursor..(cursor + 1));
                cursor += 1;
                at_line_start = false;
                continue;
            }
        }

        if !in_markup {
            match current_byte {
                b'(' => mode_stack.push(PreprocessMode::CodeParen),
                b')' => {
                    if mode_stack.last() == Some(&PreprocessMode::LinkArgsParen) {
                        mode_stack.pop();
                        builder.push_passthrough(")", cursor);
                        cursor += 1;
                        if bytes.get(cursor) == Some(&b'[') {
                            mode_stack.push(PreprocessMode::LinkBodyBracket {
                                emph_open: false,
                                strong_open: false,
                            });
                            builder.push_passthrough("[", cursor);
                            cursor += 1;
                        }
                        at_line_start = false;
                        continue;
                    }
                    if mode_stack.last() == Some(&PreprocessMode::CodeParen) {
                        mode_stack.pop();
                    }
                }
                b'{' => mode_stack.push(PreprocessMode::CodeBrace),
                b'}' => {
                    if mode_stack.last() == Some(&PreprocessMode::CodeBrace) {
                        mode_stack.pop();
                    }
                }
                b'[' => {
                    if mode_stack.last() == Some(&PreprocessMode::LinkArgsParen) {
                        mode_stack.push(PreprocessMode::LinkBodyBracket {
                            emph_open: false,
                            strong_open: false,
                        });
                    } else {
                        mode_stack.push(PreprocessMode::MarkupBracket {
                            emph_open: false,
                            strong_open: false,
                        });
                    }
                }
                _ => {}
            }
        }

        if current_byte == b'\n' {
            if in_markup && is_parbreak_at(bytes, cursor) {
                *current_emph_open(&mut mode_stack, &mut root_emph_open) = false;
                *current_strong_open(&mut mode_stack, &mut root_strong_open) = false;
            }
            builder.push_passthrough("\n", cursor);
            cursor += 1;
            at_line_start = true;
            continue;
        }

        let character_len = segment_text
            .get(cursor..)
            .and_then(|rest| rest.chars().next())
            .map_or(1, char::len_utf8);
        let next_cursor = cursor + character_len;
        if !segment_text[cursor..next_cursor]
            .chars()
            .all(|ch| ch == ' ' || ch == '\t')
        {
            at_line_start = false;
        }
        builder.push_passthrough(&segment_text[cursor..next_cursor], cursor);
        cursor = next_cursor;
    }

    if apply_streaming_tail {
        let closing_tail = compute_streaming_closing_tail(&builder.text);
        if !closing_tail.is_empty() {
            builder.push_replaced(&closing_tail, len..len);
        }
    }

    builder.finish()
}

fn try_normalize_markdown_line_prefix(text: &str, line_start: usize) -> Option<(usize, String)> {
    let rest = text.get(line_start..)?;
    let bytes = rest.as_bytes();
    let mut indent = 0;
    while indent < bytes.len() && (bytes[indent] == b' ' || bytes[indent] == b'\t') {
        indent += 1;
    }
    let indent_str = &rest[..indent];
    let after_indent = &rest[indent..];

    if after_indent.starts_with('#') {
        let hash_count = after_indent
            .bytes()
            .take_while(|byte| *byte == b'#')
            .count();
        if (1..=6).contains(&hash_count) {
            let after_hashes = &after_indent[hash_count..];
            if after_hashes.starts_with(' ') || after_hashes.starts_with('\t') {
                let equals = "=".repeat(hash_count);
                return Some((indent + hash_count, format!("{indent_str}{equals}")));
            }
        }
    }

    if let Some(after_dash) = after_indent
        .strip_prefix("- [ ] ")
        .or_else(|| after_indent.strip_prefix("* [ ] "))
    {
        let consumed = rest.len() - after_dash.len();
        return Some((consumed, format!("{indent_str}- ☐ ")));
    }
    if let Some(after_dash) = after_indent
        .strip_prefix("- [x] ")
        .or_else(|| after_indent.strip_prefix("- [X] "))
        .or_else(|| after_indent.strip_prefix("* [x] "))
        .or_else(|| after_indent.strip_prefix("* [X] "))
    {
        let consumed = rest.len() - after_dash.len();
        return Some((consumed, format!("{indent_str}- ☑ ")));
    }

    if let Some(after_rule) = after_indent.strip_prefix("---") {
        let line_end = after_rule.find('\n').unwrap_or(after_rule.len());
        if after_rule[..line_end]
            .chars()
            .all(|ch| ch == '-' || ch == ' ' || ch == '\t' || ch == '\r')
        {
            let consumed = indent + 3 + line_end;
            return Some((
                consumed,
                format!("{indent_str}#line(length: 100%, stroke: 0.5pt)"),
            ));
        }
    }

    if let Some(after_star) = after_indent.strip_prefix("* ") {
        let line_end = after_star.find('\n').unwrap_or(after_star.len());
        if !after_star[..line_end].contains('*') {
            return Some((indent + 2, format!("{indent_str}- ")));
        }
    }

    if let Some(after_gt) = after_indent.strip_prefix("> ") {
        let line_end = after_gt.find('\n').unwrap_or(after_gt.len());
        let quote_line = &after_gt[..line_end];
        if !quote_line.trim().starts_with("[!") {
            let (normalized_quote, _) = preprocess_typst_segment_inner(
                quote_line,
                0,
                &collections::HashMap::default(),
                false,
                false,
            );
            let consumed = indent + 2 + line_end;
            return Some((consumed, format!("{indent_str}#quote[{normalized_quote}]")));
        }
    }

    None
}

fn current_strong_open<'a>(
    mode_stack: &'a mut [PreprocessMode],
    root_strong_open: &'a mut bool,
) -> &'a mut bool {
    if let Some(
        PreprocessMode::MarkupBracket { strong_open, .. }
        | PreprocessMode::LinkBodyBracket { strong_open, .. },
    ) = mode_stack.last_mut()
    {
        return strong_open;
    }
    root_strong_open
}

fn current_emph_open<'a>(
    mode_stack: &'a mut [PreprocessMode],
    root_emph_open: &'a mut bool,
) -> &'a mut bool {
    if let Some(
        PreprocessMode::MarkupBracket { emph_open, .. }
        | PreprocessMode::LinkBodyBracket { emph_open, .. },
    ) = mode_stack.last_mut()
    {
        return emph_open;
    }
    root_emph_open
}

fn is_leading_underscore_identifier(text: &str, underscore_pos: usize) -> bool {
    let Some(rest) = text.get((underscore_pos + 1)..) else {
        return false;
    };
    if rest.starts_with('_') {
        return true;
    }
    let mut chars = rest.chars().peekable();
    let Some(first) = chars.peek().copied() else {
        return false;
    };
    if !first.is_ascii_alphanumeric() {
        return false;
    }
    while let Some(character) = chars.peek().copied() {
        if character.is_ascii_alphanumeric() {
            chars.next();
        } else {
            break;
        }
    }
    if chars.next() != Some('_') {
        return false;
    }
    let Some(after_mid_underscore) = chars.peek().copied() else {
        return false;
    };
    if !after_mid_underscore.is_ascii_alphanumeric() {
        return false;
    }
    let mut last_was_underscore = false;
    for character in chars {
        if character.is_ascii_alphanumeric() {
            last_was_underscore = false;
        } else if character == '_' {
            last_was_underscore = true;
        } else {
            break;
        }
    }
    !last_was_underscore
}

fn has_closing_delimiter_in_scope(text: &str, start: usize, delimiter: u8) -> bool {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut cursor = start;
    let mut bracket_depth = 0usize;
    let mut paren_depth = 0usize;
    let mut brace_depth = 0usize;

    while cursor < len {
        let byte = bytes[cursor];
        match byte {
            b'\\' => {
                cursor += next_escaped_step(text, cursor);
                continue;
            }
            b'`' => {
                let (end, _) = scan_inline_backticks(bytes, cursor);
                cursor = end;
                continue;
            }
            b'$' => {
                if is_prose_currency_dollar(text, cursor) {
                    cursor += 1;
                } else {
                    let (end, _, _, _) = scan_dollar_math(bytes, cursor);
                    cursor = end;
                }
                continue;
            }
            b'[' => {
                bracket_depth += 1;
                cursor += 1;
                continue;
            }
            b']' => {
                if bracket_depth > 0 {
                    bracket_depth -= 1;
                    cursor += 1;
                    continue;
                }
                return false;
            }
            b'(' if paren_depth > 0 => {
                paren_depth += 1;
                cursor += 1;
                continue;
            }
            b')' if paren_depth > 0 => {
                paren_depth -= 1;
                cursor += 1;
                continue;
            }
            b'{' if brace_depth > 0 => {
                brace_depth += 1;
                cursor += 1;
                continue;
            }
            b'}' if brace_depth > 0 => {
                brace_depth -= 1;
                cursor += 1;
                continue;
            }
            b'#' => {
                if let Some(after_ident) = scan_typst_hash_expr(text, cursor + 1) {
                    cursor = after_ident;
                    if bytes.get(cursor) == Some(&b'(') {
                        paren_depth += 1;
                        cursor += 1;
                    } else if bytes.get(cursor) == Some(&b'{') {
                        brace_depth += 1;
                        cursor += 1;
                    } else if bytes.get(cursor) == Some(&b'[') {
                        bracket_depth += 1;
                        cursor += 1;
                    }
                    continue;
                }
            }
            b'\n'
                if is_parbreak_at(bytes, cursor)
                    && bracket_depth == 0
                    && paren_depth == 0
                    && brace_depth == 0 =>
            {
                return false;
            }
            _ => {}
        }

        if bracket_depth == 0 && paren_depth == 0 && brace_depth == 0 && byte == delimiter {
            let prev_char = text[..cursor].chars().next_back();
            let next_char = text
                .get((cursor + 1)..)
                .and_then(|rest| rest.chars().next());
            if !is_typst_in_word(prev_char, next_char) {
                if delimiter == b'*' {
                    if bytes.get(cursor + 1) == Some(&b'*') {
                        cursor += 2;
                        continue;
                    }
                    let prev_non_ws = prev_char.is_some_and(|character| !character.is_whitespace());
                    if prev_non_ws {
                        return true;
                    }
                } else if delimiter == b'_' {
                    let prev_non_ws = prev_char.is_some_and(|character| !character.is_whitespace());
                    let next_wordy = next_char.is_some_and(is_typst_word_char);
                    if prev_non_ws && !next_wordy {
                        return true;
                    }
                }
            }
        }

        cursor += text
            .get(cursor..)
            .and_then(|rest| rest.chars().next())
            .map_or(1, char::len_utf8);
    }

    true
}

fn try_unwrap_backticked_typst_link(inner: &str) -> Option<(&str, &str)> {
    let trimmed = inner.trim();
    if trimmed.contains('\n') || !trimmed.starts_with("#link(") || !trimmed.ends_with(']') {
        return None;
    }
    let bytes = trimmed.as_bytes();
    let mut cursor = "#link(".len();
    while cursor < bytes.len() && (bytes[cursor] == b' ' || bytes[cursor] == b'\t') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'"') {
        return None;
    }
    let quote_start = cursor;
    let (quote_end, closed_quote) = scan_typst_string(bytes, quote_start);
    if !closed_quote || quote_end < quote_start + 2 {
        return None;
    }
    let raw_url = trimmed.get((quote_start + 1)..(quote_end - 1))?;
    if !is_real_link_target(raw_url) {
        return None;
    }
    cursor = quote_end;
    while cursor < bytes.len() && (bytes[cursor] == b' ' || bytes[cursor] == b'\t') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b')') {
        return None;
    }
    cursor += 1;
    while cursor < bytes.len() && (bytes[cursor] == b' ' || bytes[cursor] == b'\t') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'[') {
        return None;
    }
    let bracket_start = cursor;
    let mut depth = 0usize;
    let mut scan = bracket_start;
    while scan < bytes.len() {
        match bytes[scan] {
            b'\\' => {
                scan = (scan + 2).min(bytes.len());
                continue;
            }
            b'`' => {
                let (end, _) = scan_inline_backticks(bytes, scan);
                scan = end;
                continue;
            }
            b'[' => depth += 1,
            b']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    if scan + 1 != bytes.len() {
                        return None;
                    }
                    let label = trimmed.get((bracket_start + 1)..scan)?;
                    return Some((raw_url, label));
                }
            }
            _ => {}
        }
        scan += 1;
    }
    None
}

fn is_real_link_target(url: &str) -> bool {
    let has_scheme = url.starts_with("file://")
        || url.starts_with("https://")
        || url.starts_with("http://")
        || url.starts_with("mailto:");
    if !has_scheme {
        return false;
    }
    !url.contains("...")
        && !url.contains("example.com")
        && !url.contains("/path/to/")
        && !url.contains("absolute/path")
}

fn scan_bare_prose_url(text: &str, start: usize) -> Option<usize> {
    let rest = text.get(start..)?;
    let prefix_len = if rest.starts_with("https://") {
        "https://".len()
    } else if rest.starts_with("http://") {
        "http://".len()
    } else if rest.starts_with("file:///") {
        "file:///".len()
    } else {
        return None;
    };

    let bytes = text.as_bytes();
    let mut cursor = start + prefix_len;
    let mut open_parens = 0usize;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        if byte.is_ascii_whitespace()
            || matches!(byte, b'<' | b'>' | b'"' | b'\'' | b'`' | b'[' | b']')
        {
            break;
        }
        if byte == b'(' {
            open_parens += 1;
        } else if byte == b')' {
            if open_parens == 0 {
                break;
            }
            open_parens -= 1;
        }
        cursor += 1;
    }

    while cursor > start + prefix_len
        && matches!(bytes[cursor - 1], b'.' | b',' | b';' | b':' | b'!' | b'?')
    {
        cursor -= 1;
    }

    (cursor > start + prefix_len).then_some(cursor)
}

fn try_convert_markdown_pipe_table(
    text: &str,
    table_start: usize,
    base_offset: usize,
    inline_code_links: &collections::HashMap<Range<usize>, SharedString>,
) -> Option<(usize, String)> {
    let bytes = text.as_bytes();
    let header_line_end = find_line_end(bytes, table_start);
    let header_line = text.get(table_start..header_line_end)?;
    if !header_line.trim_start().starts_with('|') {
        return None;
    }
    let sep_line_start = next_line_start(bytes, table_start);
    if sep_line_start >= bytes.len() || sep_line_start == table_start {
        return None;
    }
    let sep_line_end = find_line_end(bytes, sep_line_start);
    let sep_line = text.get(sep_line_start..sep_line_end)?;
    if !sep_line.trim_start().starts_with('|') {
        return None;
    }

    let sep_cells = split_markdown_pipe_row(sep_line, sep_line_start)?;
    if sep_cells.is_empty() {
        return None;
    }
    let mut alignments = Vec::with_capacity(sep_cells.len());
    for (cell_slice, _) in &sep_cells {
        let trimmed = cell_slice.trim();
        let without_left = trimmed.strip_prefix(':');
        let core = without_left.unwrap_or(trimmed);
        let without_right = core.strip_suffix(':');
        let dashes = without_right.unwrap_or(core);
        if dashes.len() < 3 || !dashes.bytes().all(|byte| byte == b'-') {
            return None;
        }
        let align = match (without_left.is_some(), without_right.is_some()) {
            (true, true) => "center",
            (false, true) => "right",
            (true, false) => "left",
            (false, false) => "auto",
        };
        alignments.push(align);
    }

    let column_count = alignments.len();
    let header_cells = split_markdown_pipe_row(header_line, table_start)?;
    if header_cells.len() != column_count {
        return None;
    }

    let mut rows = vec![header_cells];
    let mut consumed_end = next_line_start(bytes, sep_line_start);
    while consumed_end < bytes.len() {
        let row_end = find_line_end(bytes, consumed_end);
        let row_line = &text[consumed_end..row_end];
        if !row_line.trim_start().starts_with('|') {
            break;
        }
        let Some(parsed_cells) = split_markdown_pipe_row(row_line, consumed_end) else {
            break;
        };
        rows.push(parsed_cells);
        consumed_end = next_line_start(bytes, consumed_end);
    }

    let align_spec = alignments.join(", ");
    let mut rendered_table = format!("#table(columns: {column_count}, align: ({align_spec}),");
    for row in rows {
        for col_index in 0..column_count {
            if let Some((cell_text, cell_offset)) = row.get(col_index) {
                let leading_ws = cell_text.len() - cell_text.trim_start().len();
                let trimmed_cell = cell_text.trim();
                let (normalized_cell, _) = preprocess_typst_segment_inner(
                    trimmed_cell,
                    base_offset + cell_offset + leading_ws,
                    inline_code_links,
                    false,
                    false,
                );
                rendered_table.push_str(&format!(" [{normalized_cell}],"));
            } else {
                rendered_table.push_str(" [],");
            }
        }
    }
    rendered_table.push(')');
    if consumed_end <= bytes.len() && text[table_start..consumed_end].ends_with('\n') {
        rendered_table.push('\n');
    }

    Some((consumed_end - table_start, rendered_table))
}

fn split_markdown_pipe_row(line: &str, line_offset: usize) -> Option<Vec<(&str, usize)>> {
    let trimmed_end = line.trim_end_matches([' ', '\t', '\r']);
    let leading_ws = trimmed_end.len() - trimmed_end.trim_start_matches([' ', '\t']).len();
    let trimmed = &trimmed_end[leading_ws..];
    if !trimmed.starts_with('|') {
        return None;
    }
    let bytes = trimmed.as_bytes();
    let mut cells = Vec::new();
    let mut cell_start = 1usize;
    let mut cursor = 1usize;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'\\' => {
                cursor = (cursor + 2).min(bytes.len());
                continue;
            }
            b'`' => {
                let (end, _) = scan_inline_backticks(bytes, cursor);
                cursor = end;
                continue;
            }
            b'|' => {
                cells.push((
                    &trimmed[cell_start..cursor],
                    line_offset + leading_ws + cell_start,
                ));
                cell_start = cursor + 1;
            }
            _ => {}
        }
        cursor += 1;
    }
    if cell_start < bytes.len() {
        let trailing = &trimmed[cell_start..];
        if !trailing.trim().is_empty() {
            cells.push((trailing, line_offset + leading_ws + cell_start));
        }
    }
    (!cells.is_empty()).then_some(cells)
}

fn scan_typst_label_ref(text: &str, start: usize) -> &str {
    let Some(rest) = text.get(start..) else {
        return "";
    };
    let mut end_rel = 0;
    for (offset, ch) in rest.char_indices() {
        if ch.is_alphanumeric() || matches!(ch, '_' | '-' | ':' | '.') {
            end_rel = offset + ch.len_utf8();
        } else {
            break;
        }
    }
    &rest[..end_rel]
}

fn is_prose_currency_dollar(text: &str, dollar_pos: usize) -> bool {
    let bytes = text.as_bytes();
    let Some(next_byte) = bytes.get(dollar_pos + 1).copied() else {
        return true;
    };
    if next_byte == b'$' {
        return false;
    }
    if next_byte == b'/' {
        return true;
    }
    if !next_byte.is_ascii_digit() {
        return false;
    }

    let line_end = find_line_end(bytes, dollar_pos + 2);
    let mut cursor = dollar_pos + 2;
    let mut in_string = false;
    let mut candidate_close = None;
    while cursor < line_end {
        let byte = bytes[cursor];
        if byte == b'\\' {
            cursor = (cursor + 2).min(line_end);
            continue;
        }
        if byte == b'"' {
            in_string = !in_string;
            cursor += 1;
            continue;
        }
        if !in_string && byte == b'$' {
            candidate_close = Some(cursor);
            break;
        }
        cursor += 1;
    }

    let Some(close_pos) = candidate_close else {
        return true;
    };

    if bytes
        .get(close_pos + 1)
        .is_some_and(|after| after.is_ascii_digit() || *after == b'/')
    {
        return true;
    }

    let between = &text[(dollar_pos + 1)..close_pos];
    if between.ends_with(char::is_whitespace) {
        return true;
    }

    !looks_like_digit_led_math(between)
}

fn looks_like_digit_led_math(between: &str) -> bool {
    let bytes = between.as_bytes();
    let mut number_end = 0usize;
    while number_end < bytes.len()
        && (bytes[number_end].is_ascii_digit()
            || bytes[number_end] == b','
            || bytes[number_end] == b'.')
    {
        number_end += 1;
    }
    if let Some(after_num) = bytes.get(number_end).copied() {
        if matches!(after_num, b'k' | b'K' | b'm' | b'M' | b'b' | b'B')
            && bytes
                .get(number_end + 1)
                .is_none_or(|next| !next.is_ascii_alphabetic())
        {
            return false;
        }
        if after_num == b'/'
            && bytes
                .get(number_end + 1)
                .is_some_and(|next| next.is_ascii_alphanumeric())
        {
            return false;
        }
    }

    let mut stack: Vec<char> = Vec::new();
    let mut unquoted = String::with_capacity(between.len());
    let mut in_string = false;
    let mut escaped = false;
    for character in between.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if character == '\\' {
            escaped = true;
            continue;
        }
        if character == '"' {
            in_string = !in_string;
            unquoted.push(' ');
            continue;
        }
        if in_string {
            continue;
        }
        match character {
            '(' => stack.push(')'),
            '[' => stack.push(']'),
            '{' => stack.push('}'),
            ')' | ']' | '}' => {
                if stack.pop() != Some(character) {
                    return false;
                }
            }
            _ => {}
        }
        unquoted.push(character);
    }
    if in_string || !stack.is_empty() {
        return false;
    }

    let mut word = String::new();
    for character in unquoted.chars().chain(std::iter::once(' ')) {
        if character.is_ascii_alphabetic() {
            word.push(character);
        } else if !word.is_empty() {
            if word.len() >= 2 && !is_known_math_word(&word) {
                return false;
            }
            word.clear();
        }
    }

    true
}

fn is_known_math_word(word: &str) -> bool {
    matches!(
        word,
        "times"
            | "div"
            | "approx"
            | "plus"
            | "minus"
            | "pm"
            | "mp"
            | "leq"
            | "geq"
            | "neq"
            | "in"
            | "not"
            | "mod"
            | "rem"
            | "log"
            | "ln"
            | "lg"
            | "exp"
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "sec"
            | "csc"
            | "cot"
            | "min"
            | "max"
            | "sup"
            | "inf"
            | "lim"
            | "sum"
            | "prod"
            | "product"
            | "integral"
            | "sqrt"
            | "root"
            | "abs"
            | "norm"
            | "floor"
            | "ceil"
            | "round"
            | "quad"
            | "wide"
            | "dif"
            | "diff"
            | "Delta"
            | "Gamma"
            | "Lambda"
            | "Omega"
            | "Phi"
            | "Pi"
            | "Psi"
            | "Sigma"
            | "Theta"
            | "alpha"
            | "beta"
            | "gamma"
            | "delta"
            | "epsilon"
            | "zeta"
            | "eta"
            | "theta"
            | "iota"
            | "kappa"
            | "lambda"
            | "mu"
            | "nu"
            | "xi"
            | "pi"
            | "rho"
            | "sigma"
            | "tau"
            | "upsilon"
            | "phi"
            | "chi"
            | "psi"
            | "omega"
            | "infinity"
            | "infty"
            | "arrow"
            | "rightarrow"
            | "leftarrow"
            | "leftrightarrow"
            | "Rightarrow"
            | "Leftarrow"
            | "Leftrightarrow"
            | "iff"
            | "implies"
            | "propto"
            | "subset"
            | "supset"
            | "cup"
            | "cap"
            | "emptyset"
            | "forall"
            | "exists"
            | "nabla"
            | "partial"
            | "cdot"
            | "dots"
            | "circle"
            | "deg"
    )
}

fn try_parse_markdown_link(
    text: &str,
    bracket_start: usize,
) -> Option<(Range<usize>, String, usize)> {
    let bytes = text.as_bytes();
    if bytes.get(bracket_start) != Some(&b'[') {
        return None;
    }
    let mut depth = 1usize;
    let mut cursor = bracket_start + 1;
    while cursor < bytes.len() && depth > 0 {
        match bytes[cursor] {
            b'\\' => {
                cursor += 2;
                continue;
            }
            b'`' => {
                let (end, _) = scan_inline_backticks(bytes, cursor);
                cursor = end;
                continue;
            }
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            b'\n' => return None,
            _ => {}
        }
        cursor += 1;
    }
    if depth != 0 || bytes.get(cursor) != Some(&b']') {
        return None;
    }
    let label_end = cursor;
    if bytes.get(label_end + 1) != Some(&b'(') {
        return None;
    }

    let url_start = label_end + 2;
    let mut url_cursor = url_start;
    let mut paren_depth = 1usize;
    while url_cursor < bytes.len() && paren_depth > 0 {
        match bytes[url_cursor] {
            b'\\' => {
                url_cursor += 2;
                continue;
            }
            b'(' => paren_depth += 1,
            b')' => {
                paren_depth -= 1;
                if paren_depth == 0 {
                    break;
                }
            }
            b'\n' | b' ' | b'\t' => return None,
            _ => {}
        }
        url_cursor += 1;
    }
    if paren_depth != 0 || bytes.get(url_cursor) != Some(&b')') {
        return None;
    }

    let url = text.get(url_start..url_cursor)?.trim();
    if url.is_empty() {
        return None;
    }
    let looks_like_url = url.contains("://")
        || url.starts_with('/')
        || url.starts_with('#')
        || url.starts_with("mailto:")
        || url.starts_with("./")
        || url.starts_with("../");
    if !looks_like_url {
        return None;
    }

    Some((
        (bracket_start + 1)..label_end,
        url.to_string(),
        url_cursor + 1,
    ))
}

fn find_markdown_double_delimiter_end(text: &str, start: usize, delim: u8) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes
        .get(start)
        .copied()
        .is_none_or(|byte| byte.is_ascii_whitespace() || byte == delim)
    {
        return None;
    }
    let mut cursor = start;
    while cursor + 1 < bytes.len() {
        match bytes[cursor] {
            b'\\' => {
                cursor += 2;
                continue;
            }
            b'`' => {
                let (end, _) = scan_inline_backticks(bytes, cursor);
                cursor = end;
                continue;
            }
            b'\n' if bytes.get(cursor + 1) == Some(&b'\n') => return None,
            byte if byte == delim && bytes.get(cursor + 1) == Some(&delim) => {
                if cursor > start && !bytes[cursor - 1].is_ascii_whitespace() {
                    return Some(cursor);
                }
            }
            _ => {}
        }
        cursor += 1;
    }
    None
}

fn scan_typst_hash_expr(text: &str, ident_start: usize) -> Option<usize> {
    let rest = text.get(ident_start..)?;
    let mut chars = rest.char_indices();
    let (_, first) = chars.next()?;
    if !first.is_alphabetic() && first != '_' {
        return None;
    }
    let mut end_rel = first.len_utf8();
    for (offset, ch) in chars {
        if ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
            end_rel = offset + ch.len_utf8();
        } else {
            break;
        }
    }
    Some(ident_start + end_rel)
}

fn compute_streaming_closing_tail(normalized: &str) -> String {
    let bytes = normalized.as_bytes();
    let len = bytes.len();
    let mut cursor = 0;
    let mut delimiters: Vec<char> = Vec::new();
    let mut hash_call_ready = false;

    while cursor < len {
        let byte = bytes[cursor];
        let in_code_mode = matches!(delimiters.last(), Some(&')' | &'}'));
        let in_math_mode = delimiters.last() == Some(&'$');
        match byte {
            b'\\' => {
                hash_call_ready = false;
                cursor += next_escaped_step(normalized, cursor);
                continue;
            }
            b'`' => {
                hash_call_ready = false;
                let (end, _) = scan_inline_backticks(bytes, cursor);
                cursor = end;
                continue;
            }
            b'"' => {
                hash_call_ready = false;
                if in_code_mode || in_math_mode {
                    let (end, closed) = scan_typst_string(bytes, cursor);
                    if !closed {
                        delimiters.push('"');
                    }
                    cursor = end;
                    continue;
                }
            }
            b'$' => {
                hash_call_ready = false;
                if in_math_mode {
                    delimiters.pop();
                } else if !in_code_mode {
                    delimiters.push('$');
                }
                cursor += 1;
                continue;
            }
            b'#' if !in_math_mode => {
                if let Some(after_ident) = scan_typst_hash_expr(normalized, cursor + 1) {
                    hash_call_ready = true;
                    cursor = after_ident;
                    continue;
                }
                hash_call_ready = true;
                cursor += 1;
                continue;
            }
            b'(' if !in_math_mode => {
                if in_code_mode || hash_call_ready {
                    delimiters.push(')');
                }
                hash_call_ready = false;
            }
            b')' if !in_math_mode => {
                hash_call_ready = false;
                if delimiters.last() == Some(&')') {
                    delimiters.pop();
                }
            }
            b'{' if !in_math_mode => {
                if in_code_mode || hash_call_ready {
                    delimiters.push('}');
                }
                hash_call_ready = false;
            }
            b'}' if !in_math_mode => {
                hash_call_ready = false;
                if delimiters.last() == Some(&'}') {
                    delimiters.pop();
                }
            }
            b'[' if !in_math_mode => {
                hash_call_ready = false;
                delimiters.push(']');
            }
            b']' if !in_math_mode => {
                hash_call_ready = false;
                while matches!(delimiters.last(), Some(&'*' | &'_')) {
                    delimiters.pop();
                }
                if delimiters.last() == Some(&']') {
                    delimiters.pop();
                }
            }
            b'\n' if !in_math_mode && !in_code_mode && is_parbreak_at(bytes, cursor) => {
                hash_call_ready = false;
                while matches!(delimiters.last(), Some(&'*' | &'_')) {
                    delimiters.pop();
                }
            }
            b'*' if !in_math_mode && !in_code_mode => {
                hash_call_ready = false;
                let prev_char = normalized[..cursor].chars().next_back();
                let next_char = normalized
                    .get((cursor + 1)..)
                    .and_then(|rest| rest.chars().next());
                if !is_typst_in_word(prev_char, next_char) {
                    if let Some(pos) = delimiters
                        .iter()
                        .rposition(|delim| matches!(*delim, '*' | ']' | ')' | '}'))
                        && delimiters[pos] == '*'
                    {
                        delimiters.remove(pos);
                    } else {
                        delimiters.push('*');
                    }
                }
            }
            b'_' if !in_math_mode && !in_code_mode => {
                hash_call_ready = false;
                let prev_char = normalized[..cursor].chars().next_back();
                let next_char = normalized
                    .get((cursor + 1)..)
                    .and_then(|rest| rest.chars().next());
                if !is_typst_in_word(prev_char, next_char) {
                    if let Some(pos) = delimiters
                        .iter()
                        .rposition(|delim| matches!(*delim, '_' | ']' | ')' | '}'))
                        && delimiters[pos] == '_'
                    {
                        delimiters.remove(pos);
                    } else {
                        delimiters.push('_');
                    }
                }
            }
            _ => {
                hash_call_ready = false;
            }
        }
        cursor += normalized
            .get(cursor..)
            .and_then(|rest| rest.chars().next())
            .map_or(1, char::len_utf8);
    }

    delimiters.into_iter().rev().collect()
}

fn close_unbalanced_math_parens(math: &str) -> String {
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for ch in math.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '(' => stack.push(')'),
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            ')' | '}' | ']' => {
                if stack.last() == Some(&ch) {
                    stack.pop();
                }
            }
            _ => {}
        }
    }
    if !in_string && stack.is_empty() {
        return math.to_string();
    }
    let mut result = math.to_string();
    if in_string {
        result.push('"');
    }
    for closer in stack.into_iter().rev() {
        result.push(closer);
    }
    result
}

fn scan_inline_backticks(bytes: &[u8], start: usize) -> (usize, bool) {
    let len = bytes.len();
    let mut tick_count = 0;
    while start + tick_count < len && bytes[start + tick_count] == b'`' {
        tick_count += 1;
    }
    let mut cursor = start + tick_count;
    while cursor < len {
        if bytes[cursor] == b'`' {
            let mut close_count = 0;
            while cursor + close_count < len && bytes[cursor + close_count] == b'`' {
                close_count += 1;
            }
            if close_count == tick_count {
                return (cursor + close_count, true);
            }
            cursor += close_count;
        } else {
            cursor += 1;
        }
    }
    (len, false)
}

fn scan_typst_string(bytes: &[u8], start: usize) -> (usize, bool) {
    let len = bytes.len();
    let mut cursor = start + 1;
    while cursor < len {
        match bytes[cursor] {
            b'\\' => cursor = (cursor + 2).min(len),
            b'"' => return (cursor + 1, true),
            _ => cursor += 1,
        }
    }
    (len, false)
}

fn scan_dollar_math(bytes: &[u8], start: usize) -> (usize, bool, bool, bool) {
    let len = bytes.len();
    let is_double = bytes.get(start + 1) == Some(&b'$');
    let content_start = start + if is_double { 2 } else { 1 };
    let starts_with_ws = bytes
        .get(content_start)
        .is_some_and(|byte| byte.is_ascii_whitespace());
    let mut cursor = content_start;
    let mut in_string = false;
    while cursor < len {
        let byte = bytes[cursor];
        if byte == b'\\' {
            cursor = (cursor + 2).min(len);
            continue;
        }
        if byte == b'"' {
            in_string = !in_string;
            cursor += 1;
            continue;
        }
        if !in_string {
            if is_double {
                if byte == b'$' && bytes.get(cursor + 1) == Some(&b'$') {
                    return (cursor + 2, true, true, false);
                }
            } else if byte == b'$' {
                return (cursor + 1, false, true, false);
            } else if byte == b'`'
                && cursor > content_start
                && !bytes[cursor - 1].is_ascii_whitespace()
                && bytes.get(cursor + 1).is_none_or(|after| {
                    matches!(
                        *after,
                        b' ' | b'\t' | b'\n' | b'\r' | b',' | b')' | b']' | b'.' | b';' | b':'
                    )
                })
            {
                return (cursor + 1, false, true, true);
            }

            if byte == b'\n' {
                if !is_double && !starts_with_ws {
                    return (cursor, false, false, false);
                }
                if bytes.get(cursor + 1) == Some(&b'\n') {
                    return (cursor, is_double, false, false);
                }
            }
        }
        cursor += 1;
    }
    (len, is_double, false, false)
}

fn is_standalone_on_line(text: &str, start: usize, end: usize) -> bool {
    let before = &text[..start];
    let line_prefix = match before.rfind('\n') {
        Some(newline_index) => &before[(newline_index + 1)..],
        None => before,
    };
    if !line_prefix.trim().is_empty() {
        return false;
    }
    let after = &text[end..];
    let line_suffix = match after.find('\n') {
        Some(newline_index) => &after[..newline_index],
        None => after,
    };
    line_suffix.trim().is_empty()
}

fn is_parbreak_at(bytes: &[u8], newline_pos: usize) -> bool {
    if bytes.get(newline_pos) != Some(&b'\n') {
        return false;
    }
    let mut index = newline_pos + 1;
    while index < bytes.len() {
        match bytes[index] {
            b' ' | b'\t' | b'\r' => index += 1,
            b'\n' => return true,
            _ => return false,
        }
    }
    false
}

fn is_typst_in_word(prev: Option<char>, next: Option<char>) -> bool {
    prev.is_some_and(is_typst_word_char) && next.is_some_and(is_typst_word_char)
}

fn is_typst_word_char(character: char) -> bool {
    character.is_alphanumeric() && !is_cjk_character(character)
}

fn is_cjk_character(character: char) -> bool {
    matches!(
        character,
        '\u{3040}'..='\u{30FF}'
            | '\u{31F0}'..='\u{31FF}'
            | '\u{3400}'..='\u{4DBF}'
            | '\u{4E00}'..='\u{9FFF}'
            | '\u{AC00}'..='\u{D7AF}'
            | '\u{F900}'..='\u{FAFF}'
            | '\u{20000}'..='\u{2FA1F}'
    )
}

fn find_closing_subslice(text: &str, start: usize, needle: &str) -> Option<usize> {
    let rest = text.get(start..)?;
    rest.find(needle).map(|offset| start + offset)
}

fn escape_typst_string(raw: &str) -> String {
    raw.replace('\\', "\\\\").replace('"', "\\\"")
}

fn next_escaped_step(text: &str, backslash_pos: usize) -> usize {
    let next_char_len = text
        .get((backslash_pos + 1)..)
        .and_then(|rest| rest.chars().next())
        .map_or(0, char::len_utf8);
    1 + next_char_len
}

fn find_line_end(bytes: &[u8], start: usize) -> usize {
    let mut index = start;
    while index < bytes.len() && bytes[index] != b'\n' && bytes[index] != b'\r' {
        index += 1;
    }
    index
}

fn next_line_start(bytes: &[u8], start: usize) -> usize {
    let mut index = start;
    while index < bytes.len() && bytes[index] != b'\n' {
        index += 1;
    }
    if index < bytes.len() && bytes[index] == b'\n' {
        index + 1
    } else {
        index
    }
}
