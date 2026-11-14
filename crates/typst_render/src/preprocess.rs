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

    while index < len {
        match bytes[index] {
            b'\\' => {
                index += 1;
                if let Some(character) = segment_text
                    .get(index..)
                    .and_then(|rest| rest.chars().next())
                {
                    index += character.len_utf8();
                }
            }
            b'$' => {
                let is_double = bytes.get(index + 1) == Some(&b'$');
                index += if is_double { 2 } else { 1 };
                while index < len {
                    if bytes[index] == b'\\' {
                        index += 2;
                        continue;
                    }
                    if is_double {
                        if bytes.get(index) == Some(&b'$') && bytes.get(index + 1) == Some(&b'$') {
                            index += 2;
                            break;
                        }
                    } else if bytes[index] == b'$' {
                        index += 1;
                        break;
                    }
                    index += 1;
                }
            }
            b'`' => {
                let start = index;
                let mut tick_count = 0;
                while index + tick_count < len && bytes[index + tick_count] == b'`' {
                    tick_count += 1;
                }
                index += tick_count;
                let content_start = index;
                let mut found_end = None;
                while index < len {
                    if bytes[index] == b'`' {
                        let mut close_count = 0;
                        while index + close_count < len && bytes[index + close_count] == b'`' {
                            close_count += 1;
                        }
                        if close_count == tick_count {
                            found_end = Some((index, index + close_count));
                            index += close_count;
                            break;
                        }
                        index += close_count;
                    } else {
                        index += 1;
                    }
                }
                if let Some((content_end, full_end)) = found_end {
                    let inner = &segment_text[content_start..content_end];
                    if !inner.contains('\n') && !inner.is_empty() {
                        spans.push(TypstInlineCodeSpan {
                            full_range: (segment_base_offset + start)
                                ..(segment_base_offset + full_end),
                            content_range: (segment_base_offset + content_start)
                                ..(segment_base_offset + content_end),
                            text: SharedString::from(inner.to_string()),
                        });
                    }
                }
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
            '\\' | '#' | '$' | '*' | '_' | '`' | '[' | ']' | '<' | '>' | '@' | '=' | '~' => {
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
    CodeBrace,
    MarkupBracket,
}

pub fn preprocess_typst_segment(
    segment_text: &str,
    base_offset: usize,
    inline_code_links: &collections::HashMap<Range<usize>, SharedString>,
) -> (String, OffsetMap) {
    let mut builder = NormalizedBuilder::new(base_offset, segment_text.len());
    let bytes = segment_text.as_bytes();
    let len = bytes.len();
    let mut cursor = 0;
    let mut at_line_start = true;
    let mut mode_stack: Vec<PreprocessMode> = Vec::new();

    while cursor < len {
        let in_markup = mode_stack
            .last()
            .is_none_or(|mode| *mode == PreprocessMode::MarkupBracket);

        if in_markup && at_line_start {
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
            let (raw_end, closed) = scan_inline_backticks(bytes, cursor);
            let global_range = (base_offset + cursor)..(base_offset + raw_end);
            if in_markup && closed {
                if let Some(link_url) = inline_code_links.get(&global_range) {
                    let raw_slice = &segment_text[cursor..raw_end];
                    let escaped_url = escape_typst_string(link_url.as_ref());
                    let linked = format!("#link(\"{escaped_url}\")[{raw_slice}]");
                    builder.push_replaced(&linked, cursor..raw_end);
                    cursor = raw_end;
                    at_line_start = false;
                    continue;
                }
            }
            builder.push_passthrough(&segment_text[cursor..raw_end], cursor);
            if !closed {
                let tick_count = bytes[cursor..]
                    .iter()
                    .take_while(|byte| **byte == b'`')
                    .count();
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

            let (math_end, is_double, closed) = scan_dollar_math(bytes, cursor);
            let delim_len = if is_double { 2 } else { 1 };
            let inner_end = if closed {
                math_end - delim_len
            } else {
                math_end
            };
            let inner = &segment_text[(cursor + delim_len)..inner_end];

            let standalone_on_line = is_standalone_on_line(segment_text, cursor, math_end);
            let has_both_spaces = inner.starts_with(char::is_whitespace)
                && inner.ends_with(char::is_whitespace)
                && inner.trim().len() < inner.len();
            let display_mode = is_double || (has_both_spaces && standalone_on_line);

            if !is_double
                && closed
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

        if in_markup && current_byte == b'*' && bytes.get(cursor + 1) == Some(&b'*') {
            if let Some(close_pos) =
                find_markdown_double_delimiter_end(segment_text, cursor + 2, b'*')
            {
                builder.push_replaced("*", cursor..(cursor + 2));
                let inner = &segment_text[(cursor + 2)..close_pos];
                let (normalized_inner, _) =
                    preprocess_typst_segment(inner, base_offset + cursor + 2, inline_code_links);
                builder.push_replaced(&normalized_inner, (cursor + 2)..close_pos);
                builder.push_replaced("*", close_pos..(close_pos + 2));
                cursor = close_pos + 2;
                at_line_start = false;
                continue;
            }
        }

        if in_markup && current_byte == b'~' {
            if bytes.get(cursor + 1) == Some(&b'~') {
                if let Some(close_pos) =
                    find_markdown_double_delimiter_end(segment_text, cursor + 2, b'~')
                {
                    builder.push_replaced("#strike[", cursor..(cursor + 2));
                    let inner = &segment_text[(cursor + 2)..close_pos];
                    let (normalized_inner, _) = preprocess_typst_segment(
                        inner,
                        base_offset + cursor + 2,
                        inline_code_links,
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
                    let (normalized_label, _) = preprocess_typst_segment(
                        label_slice,
                        base_offset + label_range.start,
                        inline_code_links,
                    );
                    let replacement = format!("#link(\"{escaped_url}\")[{normalized_label}]");
                    builder.push_replaced(&replacement, cursor..full_end);
                    cursor = full_end;
                    at_line_start = false;
                    continue;
                }
            }
            if mode_stack.last() == Some(&PreprocessMode::MarkupBracket) {
                mode_stack.push(PreprocessMode::MarkupBracket);
                builder.push_passthrough("[", cursor);
                cursor += 1;
                at_line_start = false;
                continue;
            }
        }

        if in_markup
            && current_byte == b']'
            && mode_stack.last() == Some(&PreprocessMode::MarkupBracket)
        {
            mode_stack.pop();
            builder.push_passthrough("]", cursor);
            cursor += 1;
            at_line_start = false;
            continue;
        }

        if current_byte == b'#' {
            if let Some(after_ident) = scan_typst_hash_expr(segment_text, cursor + 1) {
                builder.push_passthrough(&segment_text[cursor..after_ident], cursor);
                cursor = after_ident;
                if bytes.get(cursor) == Some(&b'(') {
                    mode_stack.push(PreprocessMode::CodeParen);
                    builder.push_passthrough("(", cursor);
                    cursor += 1;
                } else if bytes.get(cursor) == Some(&b'{') {
                    mode_stack.push(PreprocessMode::CodeBrace);
                    builder.push_passthrough("{", cursor);
                    cursor += 1;
                } else if bytes.get(cursor) == Some(&b'[') {
                    mode_stack.push(PreprocessMode::MarkupBracket);
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
                b'[' => mode_stack.push(PreprocessMode::MarkupBracket),
                _ => {}
            }
        }

        if current_byte == b'\n' {
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

    let closing_tail = compute_streaming_closing_tail(&builder.text);
    if !closing_tail.is_empty() {
        builder.push_replaced(&closing_tail, len..len);
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
            let escaped_inner = quote_line.replace('[', "\\[").replace(']', "\\]");
            let consumed = indent + 2 + line_end;
            return Some((consumed, format!("{indent_str}#quote[{escaped_inner}]")));
        }
    }

    None
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
    if bytes.get(dollar_pos + 1) == Some(&b'$') {
        return false;
    }
    if !bytes
        .get(dollar_pos + 1)
        .is_some_and(|next| next.is_ascii_digit())
    {
        return false;
    }

    let mut cursor = dollar_pos + 2;
    let mut in_string = false;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        if byte == b'\\' {
            cursor = (cursor + 2).min(bytes.len());
            continue;
        }
        if byte == b'\n' && bytes.get(cursor + 1) == Some(&b'\n') {
            return true;
        }
        if byte == b'"' {
            in_string = !in_string;
            cursor += 1;
            continue;
        }
        if !in_string && byte == b'$' {
            let followed_by_digit = bytes
                .get(cursor + 1)
                .is_some_and(|after| after.is_ascii_digit());
            return followed_by_digit;
        }
        cursor += 1;
    }

    true
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
    let mut open_strong = false;
    let mut open_emph = false;
    let mut hash_call_ready = false;

    while cursor < len {
        let byte = bytes[cursor];
        let in_code_mode = matches!(delimiters.last(), Some(&')' | &'}'));
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
                let in_code_or_math = delimiters
                    .last()
                    .is_some_and(|top| matches!(*top, ')' | '}' | '$'));
                if in_code_or_math {
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
                if delimiters.last() == Some(&'$') {
                    delimiters.pop();
                } else if !in_code_mode {
                    delimiters.push('$');
                }
                cursor += 1;
                continue;
            }
            b'#' if delimiters.last() != Some(&'$') => {
                if let Some(after_ident) = scan_typst_hash_expr(normalized, cursor + 1) {
                    hash_call_ready = true;
                    cursor = after_ident;
                    continue;
                }
                hash_call_ready = true;
                cursor += 1;
                continue;
            }
            b'(' if delimiters.last() != Some(&'$') => {
                if in_code_mode || hash_call_ready {
                    delimiters.push(')');
                }
                hash_call_ready = false;
            }
            b')' => {
                hash_call_ready = false;
                if delimiters.last() == Some(&')') {
                    delimiters.pop();
                }
            }
            b'{' if delimiters.last() != Some(&'$') => {
                if in_code_mode || hash_call_ready {
                    delimiters.push('}');
                }
                hash_call_ready = false;
            }
            b'}' => {
                hash_call_ready = false;
                if delimiters.last() == Some(&'}') {
                    delimiters.pop();
                }
            }
            b'[' if delimiters.last() != Some(&'$') => {
                hash_call_ready = false;
                delimiters.push(']');
            }
            b']' => {
                hash_call_ready = false;
                if delimiters.last() == Some(&']') {
                    delimiters.pop();
                }
            }
            b'*' if delimiters.last() != Some(&'$') && !in_code_mode => {
                hash_call_ready = false;
                let at_list_marker = is_at_line_indent(bytes, cursor)
                    && bytes.get(cursor + 1).is_some_and(|next| *next == b' ');
                if !at_list_marker {
                    if open_strong {
                        open_strong = false;
                    } else if bytes
                        .get(cursor + 1)
                        .is_some_and(|next| !next.is_ascii_whitespace())
                    {
                        open_strong = true;
                    }
                }
            }
            b'_' if delimiters.last() != Some(&'$') && !in_code_mode => {
                hash_call_ready = false;
                let prev_alnum = cursor > 0 && bytes[cursor - 1].is_ascii_alphanumeric();
                let next_alnum = bytes
                    .get(cursor + 1)
                    .is_some_and(|next| next.is_ascii_alphanumeric());
                if !(prev_alnum && next_alnum) {
                    if open_emph {
                        open_emph = false;
                    } else if bytes
                        .get(cursor + 1)
                        .is_some_and(|next| !next.is_ascii_whitespace())
                    {
                        open_emph = true;
                    }
                }
            }
            b'\n' => {
                hash_call_ready = false;
                if bytes.get(cursor + 1) == Some(&b'\n') {
                    open_strong = false;
                    open_emph = false;
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

    let mut tail = String::new();
    if open_emph {
        tail.push('_');
    }
    if open_strong {
        tail.push('*');
    }
    for closer in delimiters.into_iter().rev() {
        tail.push(closer);
    }
    tail
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

fn scan_dollar_math(bytes: &[u8], start: usize) -> (usize, bool, bool) {
    let len = bytes.len();
    let is_double = bytes.get(start + 1) == Some(&b'$');
    let mut cursor = start + if is_double { 2 } else { 1 };
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
                    return (cursor + 2, true, true);
                }
            } else if byte == b'$' {
                return (cursor + 1, false, true);
            }
        }
        cursor += 1;
    }
    (len, is_double, false)
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

fn is_at_line_indent(bytes: &[u8], pos: usize) -> bool {
    let mut index = pos;
    while index > 0 {
        let prev = bytes[index - 1];
        if prev == b'\n' {
            return true;
        }
        if prev != b' ' && prev != b'\t' {
            return false;
        }
        index -= 1;
    }
    true
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
