use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use agent_client_protocol::v2::schema as acp_v2;
use anyhow::{Result, anyhow};
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use regex::RegexBuilder;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};

use crate::{
    AgentMessageContent, AgentTool, CompactionInfo, DbThreadMetadata, Message, ThreadEnvironment,
    ToolCallEventStream, ToolInput, UserMessageContent,
};

const DEFAULT_QUERY_LIMIT: usize = 25;
const MAX_QUERY_LIMIT: usize = 100;
const DEFAULT_MAX_CONTENT_CHARS: usize = 2_000;
const MAX_ALLOWED_CONTENT_CHARS: usize = 16_000;

fn deserialize_optional_session_id<'de, D>(
    deserializer: D,
) -> Result<Option<acp_v2::SessionId>, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(value) = Option::<serde_json::Value>::deserialize(deserializer)? else {
        return Ok(None);
    };

    if value
        .as_str()
        .is_some_and(|session_id| session_id.trim().is_empty())
    {
        return Ok(None);
    }

    serde_json::from_value(value)
        .map(Some)
        .map_err(serde::de::Error::custom)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ThreadHistoryEntryKindFilter {
    UserMessage,
    AssistantMessage,
    Thinking,
    ToolCall,
    Compaction,
    SubagentResult,
}

/// Query verbatim message history from the current thread (including messages
/// hidden behind context compaction) or list and search past threads stored in
/// Zed's thread database.
///
/// ### Usage modes
/// - **Pre-compaction recall in the current thread:** Omit `thread_id` (and
///   leave `list_threads` false) to inspect verbatim messages in this thread.
///   Pass `before_latest_compaction: true`, `search_regex`, `tool_names`,
///   `entry_kinds`, or `from_index` / `to_index` to retrieve exact prior tool
///   outputs, diffs, errors, or discussion without re-reading everything.
/// - **List or search saved threads:** Pass `list_threads: true` or
///   `thread_query` (regex/substring matched against thread titles and
///   workspace folder paths) to discover prior thread IDs.
/// - **Inspect a specific past thread:** Pass `thread_id` along with any
///   combination of `search_regex`, `entry_kinds`, `tool_names`, `from_index`,
///   `to_index`, `limit`, and `tail`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct QueryThreadHistoryToolInput {
    /// Specific thread session ID to inspect. Omit to query the current thread
    /// (unless `list_threads` or `thread_query` is set).
    #[serde(default, deserialize_with = "deserialize_optional_session_id")]
    pub thread_id: Option<acp_v2::SessionId>,

    /// When `true` (or when `thread_query` is provided without `thread_id`),
    /// returns matching thread metadata from Zed's saved thread database.
    #[serde(default)]
    pub list_threads: bool,

    /// Optional regex or substring matched against thread titles and workspace
    /// folder paths when listing saved threads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_query: Option<String>,

    /// When listing threads, only include threads whose workspace folder paths
    /// overlap the current project.
    #[serde(default)]
    pub workspace_only: bool,

    /// When listing threads, also include child subagent threads (default: `false`).
    #[serde(default)]
    pub include_subagents: bool,

    /// When querying messages in a thread, clamp the scan range to messages
    /// strictly before the latest compaction boundary (`0..latest_compaction_index`).
    #[serde(default)]
    pub before_latest_compaction: bool,

    /// Entry categories to include when querying messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_kinds: Option<Vec<ThreadHistoryEntryKindFilter>>,

    /// Filter `tool_call` entries to tools whose name matches one of these names
    /// (e.g. `["edit_file", "terminal", "read_file"]`). Automatically restricts
    /// to `tool_call` entries unless `entry_kinds` also specifies other kinds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_names: Option<Vec<String>>,

    /// Regular expression (Rust `regex` syntax) matched against message text,
    /// attached mentions, thinking blocks, tool inputs/outputs, compaction
    /// summaries, and subagent results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_regex: Option<String>,

    /// Whether `search_regex` and `thread_query` matching are case-sensitive
    /// (default: `false`).
    #[serde(default)]
    pub case_sensitive: bool,

    /// Inclusive 0-based `message_index` in the thread to start scanning from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_index: Option<usize>,

    /// Exclusive 0-based `message_index` in the thread to stop scanning at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_index: Option<usize>,

    /// Maximum number of matching threads or entries to return (default: 25, max: 100).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,

    /// When `true` (default when `from_index`, `to_index`, and `search_regex` are
    /// omitted), returns the most recent `limit` matching entries; when `false`,
    /// returns the earliest `limit` matching entries forward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail: Option<bool>,

    /// Maximum characters per entry text/input/output field before truncation
    /// (default: 2000, max: 16000).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_content_chars: Option<usize>,
}

impl QueryThreadHistoryToolInput {
    pub fn is_list_request(&self) -> bool {
        self.thread_id.is_none()
            && (self.list_threads
                || self
                    .thread_query
                    .as_deref()
                    .is_some_and(|query| !query.trim().is_empty()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadHistoryListItem {
    pub thread_id: acp_v2::SessionId,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<acp_v2::SessionId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub folder_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadHistorySubagentOutcome {
    pub session_id: acp_v2::SessionId,
    pub label: String,
    pub status: String,
    pub output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ThreadHistoryEntry {
    UserMessage {
        message_index: usize,
        text: String,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        mentions: Vec<String>,
    },
    AssistantMessage {
        message_index: usize,
        text: String,
    },
    Thinking {
        message_index: usize,
        text: String,
    },
    ToolCall {
        message_index: usize,
        tool_use_id: String,
        tool_name: String,
        input: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        #[serde(skip_serializing_if = "std::ops::Not::not", default)]
        is_error: bool,
    },
    Compaction {
        message_index: usize,
        strategy: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        summary: Option<String>,
    },
    SubagentResult {
        message_index: usize,
        results: Vec<ThreadHistorySubagentOutcome>,
    },
}

impl ThreadHistoryEntry {
    fn message_index(&self) -> usize {
        match self {
            Self::UserMessage { message_index, .. }
            | Self::AssistantMessage { message_index, .. }
            | Self::Thinking { message_index, .. }
            | Self::ToolCall { message_index, .. }
            | Self::Compaction { message_index, .. }
            | Self::SubagentResult { message_index, .. } => *message_index,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadHistoryQueryDetail {
    pub thread_id: acp_v2::SessionId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub total_messages: usize,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub compaction_indices: Vec<usize>,
    pub scanned_range: (usize, usize),
    pub total_matching_entries: usize,
    pub returned_entries: usize,
    pub truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_message_index: Option<usize>,
    pub entries: Vec<ThreadHistoryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum QueryThreadHistoryToolOutput {
    ThreadList {
        total_threads: usize,
        matched_threads: usize,
        truncated: bool,
        threads: Vec<ThreadHistoryListItem>,
    },
    Entries(Box<ThreadHistoryQueryDetail>),
    Error {
        error: String,
    },
}

impl From<QueryThreadHistoryToolOutput> for LanguageModelToolResultContent {
    fn from(output: QueryThreadHistoryToolOutput) -> Self {
        serde_json::to_string_pretty(&output)
            .unwrap_or_else(|error| {
                format!("Failed to serialize query_thread_history output: {error}")
            })
            .into()
    }
}

pub struct QueryThreadHistoryTool {
    environment: Rc<dyn ThreadEnvironment>,
}

impl QueryThreadHistoryTool {
    pub fn new(environment: Rc<dyn ThreadEnvironment>) -> Self {
        Self { environment }
    }
}

impl AgentTool for QueryThreadHistoryTool {
    type Input = QueryThreadHistoryToolInput;
    type Output = QueryThreadHistoryToolOutput;

    const NAME: &'static str = "query_thread_history";

    fn kind() -> acp_v2::ToolKind {
        acp_v2::ToolKind::Read
    }

    fn allow_in_restricted_mode() -> bool {
        false
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(input) => {
                if input.is_list_request() {
                    if let Some(query) = input.thread_query.as_deref()
                        && !query.trim().is_empty()
                    {
                        return format!("Search saved threads (/{query}/)").into();
                    }
                    return "List saved threads".into();
                }

                let target = match input.thread_id.as_ref() {
                    Some(thread_id) => {
                        let short_id: String = thread_id.0.chars().take(8).collect();
                        format!("thread {short_id}")
                    }
                    None if input.before_latest_compaction => "pre-compaction history".to_string(),
                    None => "thread history".to_string(),
                };

                if let Some(regex) = input.search_regex.as_deref() {
                    format!("Query {target} (/{regex}/)").into()
                } else if let Some(tools) = input.tool_names.as_ref()
                    && !tools.is_empty()
                {
                    format!("Query {target} ({})", tools.join(", ")).into()
                } else {
                    format!("Query {target}").into()
                }
            }
            Err(_) => "Query thread history".into(),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let environment = self.environment.clone();
        cx.spawn(async move |cx| {
            let input =
                input
                    .recv()
                    .await
                    .map_err(|error| QueryThreadHistoryToolOutput::Error {
                        error: error.to_string(),
                    })?;

            let query_task = cx.update(|cx| environment.query_thread_history(input, cx));
            match query_task.await {
                Ok(output) => {
                    let rendered = serde_json::to_string_pretty(&output).unwrap_or_default();
                    event_stream.update_fields(|update| {
                        update.content(vec![format!("```json\n{rendered}\n```").into()])
                    });
                    Ok(output)
                }
                Err(error) => {
                    let error_text = error.to_string();
                    event_stream
                        .update_fields(|update| update.content(vec![error_text.clone().into()]));
                    Err(QueryThreadHistoryToolOutput::Error { error: error_text })
                }
            }
        })
    }

    fn replay(
        &self,
        _input: Self::Input,
        output: Self::Output,
        event_stream: ToolCallEventStream,
        _cx: &mut App,
    ) -> Result<()> {
        let rendered = match &output {
            QueryThreadHistoryToolOutput::Error { error } => error.clone(),
            _ => format!(
                "```json\n{}\n```",
                serde_json::to_string_pretty(&output).unwrap_or_default()
            ),
        };
        event_stream.update_fields(|update| update.content(vec![rendered.into()]));
        Ok(())
    }
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    let total_chars = text.chars().count();
    if total_chars <= max_chars {
        return text.to_string();
    }
    let truncated: String = text.chars().take(max_chars).collect();
    format!("{truncated}… [truncated, {total_chars} chars total]")
}

fn format_tool_result_content(content: &[LanguageModelToolResultContent]) -> String {
    let mut parts = Vec::new();
    for item in content {
        match item {
            LanguageModelToolResultContent::Text(text) => {
                if !text.is_empty() {
                    parts.push(text.as_ref());
                }
            }
            LanguageModelToolResultContent::Image(_) => {
                parts.push("<image>");
            }
        }
    }
    parts.join("\n")
}

pub fn evaluate_thread_history_list(
    threads: &[DbThreadMetadata],
    workspace_roots: &[PathBuf],
    input: &QueryThreadHistoryToolInput,
) -> Result<QueryThreadHistoryToolOutput> {
    let query_regex = input
        .thread_query
        .as_deref()
        .filter(|pattern| !pattern.trim().is_empty())
        .map(|pattern| {
            RegexBuilder::new(pattern)
                .case_insensitive(!input.case_sensitive)
                .build()
                .map_err(|error| anyhow!("Invalid `thread_query` regex {pattern:?}: {error}"))
        })
        .transpose()?;

    let limit = input
        .limit
        .unwrap_or(DEFAULT_QUERY_LIMIT)
        .clamp(1, MAX_QUERY_LIMIT);

    let mut matched = Vec::new();
    let mut visible_total = 0usize;

    for meta in threads {
        if !input.include_subagents && meta.parent_session_id.is_some() {
            continue;
        }
        visible_total += 1;

        if input.workspace_only && !workspace_roots.is_empty() {
            let overlaps_workspace = meta.folder_paths.paths().iter().any(|folder_path| {
                let path: &Path = folder_path.as_ref();
                workspace_roots.iter().any(|root| {
                    path == root.as_path() || path.starts_with(root) || root.starts_with(path)
                })
            });
            if !overlaps_workspace {
                continue;
            }
        }

        let folder_strings: Vec<String> = meta
            .folder_paths
            .paths()
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect();

        if let Some(regex) = query_regex.as_ref() {
            let title_matches = regex.is_match(meta.title.as_ref());
            let folder_matches = folder_strings.iter().any(|path| regex.is_match(path));
            let id_matches = regex.is_match(meta.id.0.as_ref());
            if !title_matches && !folder_matches && !id_matches {
                continue;
            }
        }

        matched.push(ThreadHistoryListItem {
            thread_id: meta.id.clone(),
            title: meta.title.to_string(),
            parent_session_id: meta.parent_session_id.clone(),
            created_at: meta.created_at.map(|dt| dt.to_rfc3339()),
            updated_at: meta.updated_at.to_rfc3339(),
            folder_paths: folder_strings,
        });
    }

    let matched_threads = matched.len();
    let truncated = matched_threads > limit;
    matched.truncate(limit);

    Ok(QueryThreadHistoryToolOutput::ThreadList {
        total_threads: visible_total,
        matched_threads,
        truncated,
        threads: matched,
    })
}

pub fn evaluate_thread_history_messages(
    thread_id: acp_v2::SessionId,
    title: Option<String>,
    messages: &[Arc<Message>],
    input: &QueryThreadHistoryToolInput,
) -> Result<QueryThreadHistoryToolOutput> {
    let search_regex = input
        .search_regex
        .as_deref()
        .filter(|pattern| !pattern.is_empty())
        .map(|pattern| {
            RegexBuilder::new(pattern)
                .case_insensitive(!input.case_sensitive)
                .build()
                .map_err(|error| anyhow!("Invalid `search_regex` {pattern:?}: {error}"))
        })
        .transpose()?;

    let tool_name_filters: Option<Vec<String>> = input.tool_names.as_ref().and_then(|names| {
        let filtered: Vec<String> = names
            .iter()
            .map(|name| name.trim().to_ascii_lowercase())
            .filter(|name| !name.is_empty())
            .collect();
        (!filtered.is_empty()).then_some(filtered)
    });

    let allow_kind = |kind: ThreadHistoryEntryKindFilter| -> bool {
        if let Some(kinds) = input.entry_kinds.as_ref() {
            return kinds.contains(&kind);
        }
        if tool_name_filters.is_some() {
            return kind == ThreadHistoryEntryKindFilter::ToolCall;
        }
        true
    };

    let compaction_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            matches!(&**message, Message::Compaction(_)).then_some(index)
        })
        .collect();

    let total_messages = messages.len();
    let mut scan_end = input.to_index.unwrap_or(total_messages).min(total_messages);
    if input.before_latest_compaction
        && let Some(&latest_compaction_ix) = compaction_indices.last()
    {
        scan_end = scan_end.min(latest_compaction_ix);
    }
    let scan_start = input.from_index.unwrap_or(0).min(scan_end);

    let limit = input
        .limit
        .unwrap_or(DEFAULT_QUERY_LIMIT)
        .clamp(1, MAX_QUERY_LIMIT);
    let max_content_chars = input
        .max_content_chars
        .unwrap_or(DEFAULT_MAX_CONTENT_CHARS)
        .clamp(64, MAX_ALLOWED_CONTENT_CHARS);
    let tail = input.tail.unwrap_or(
        input.from_index.is_none() && input.to_index.is_none() && input.search_regex.is_none(),
    );

    let mut matching_entries = Vec::new();

    for (message_index, message) in messages.iter().enumerate().take(scan_end).skip(scan_start) {
        match &**message {
            Message::User(user_message) => {
                if user_message.content.is_empty()
                    || !allow_kind(ThreadHistoryEntryKindFilter::UserMessage)
                {
                    continue;
                }
                let mut text_parts = Vec::new();
                let mut mentions = Vec::new();
                let mut searchable_mentions = Vec::new();
                for chunk in &*user_message.content {
                    match chunk {
                        UserMessageContent::Text(text) => {
                            if !text.is_empty() {
                                text_parts.push(text.as_str());
                            }
                        }
                        UserMessageContent::Mention { uri, content } => {
                            let label = uri.as_link().to_string();
                            mentions.push(label.clone());
                            searchable_mentions.push(label);
                            if !content.is_empty() {
                                searchable_mentions.push(content.to_string());
                            }
                        }
                        UserMessageContent::Image(_) => {
                            text_parts.push("<image>");
                        }
                    }
                }
                let combined_text = text_parts.join("\n");
                if let Some(regex) = search_regex.as_ref() {
                    let matched = regex.is_match(&combined_text)
                        || searchable_mentions
                            .iter()
                            .any(|mention| regex.is_match(mention));
                    if !matched {
                        continue;
                    }
                }
                matching_entries.push(ThreadHistoryEntry::UserMessage {
                    message_index,
                    text: truncate_text(&combined_text, max_content_chars),
                    mentions,
                });
            }
            Message::Agent(agent_message) => {
                let mut text_parts = Vec::new();
                let mut thinking_parts = Vec::new();
                let mut tool_uses = Vec::new();

                for chunk in &agent_message.content {
                    match chunk {
                        AgentMessageContent::Text(text) => {
                            if !text.is_empty() {
                                text_parts.push(text.as_str());
                            }
                        }
                        AgentMessageContent::Thinking { text, .. } => {
                            if !text.is_empty() {
                                thinking_parts.push(text.as_str());
                            }
                        }
                        AgentMessageContent::RedactedThinking(_) => {}
                        AgentMessageContent::ToolUse(tool_use) => {
                            tool_uses.push(tool_use);
                        }
                    }
                }

                if !thinking_parts.is_empty() && allow_kind(ThreadHistoryEntryKindFilter::Thinking)
                {
                    let combined_thinking = thinking_parts.join("\n\n");
                    if search_regex
                        .as_ref()
                        .is_none_or(|regex| regex.is_match(&combined_thinking))
                    {
                        matching_entries.push(ThreadHistoryEntry::Thinking {
                            message_index,
                            text: truncate_text(&combined_thinking, max_content_chars),
                        });
                    }
                }

                if !text_parts.is_empty()
                    && allow_kind(ThreadHistoryEntryKindFilter::AssistantMessage)
                {
                    let combined_text = text_parts.join("\n\n");
                    if search_regex
                        .as_ref()
                        .is_none_or(|regex| regex.is_match(&combined_text))
                    {
                        matching_entries.push(ThreadHistoryEntry::AssistantMessage {
                            message_index,
                            text: truncate_text(&combined_text, max_content_chars),
                        });
                    }
                }

                if allow_kind(ThreadHistoryEntryKindFilter::ToolCall) {
                    for tool_use in tool_uses {
                        if let Some(allowed_names) = tool_name_filters.as_ref() {
                            let lower_name = tool_use.name.to_ascii_lowercase();
                            if !allowed_names.iter().any(|name| name == &lower_name) {
                                continue;
                            }
                        }

                        let raw_input = if !tool_use.raw_input.is_empty() {
                            tool_use.raw_input.clone()
                        } else {
                            tool_use.input.to_display_json().to_string()
                        };
                        let tool_result = agent_message.tool_results.get(&tool_use.id);
                        let raw_output =
                            tool_result.map(|result| format_tool_result_content(&result.content));
                        let is_error = tool_result.is_some_and(|result| result.is_error);

                        if let Some(regex) = search_regex.as_ref() {
                            let matched = regex.is_match(tool_use.name.as_ref())
                                || regex.is_match(&raw_input)
                                || raw_output
                                    .as_deref()
                                    .is_some_and(|output| regex.is_match(output));
                            if !matched {
                                continue;
                            }
                        }

                        matching_entries.push(ThreadHistoryEntry::ToolCall {
                            message_index,
                            tool_use_id: tool_use.id.to_string(),
                            tool_name: tool_use.name.to_string(),
                            input: truncate_text(&raw_input, max_content_chars),
                            output: raw_output
                                .map(|output| truncate_text(&output, max_content_chars)),
                            is_error,
                        });
                    }
                }
            }
            Message::Compaction(info) => {
                if !allow_kind(ThreadHistoryEntryKindFilter::Compaction) {
                    continue;
                }
                let (strategy, summary) = match info {
                    CompactionInfo::Summary(summary) => {
                        ("summary".to_string(), Some(summary.to_string()))
                    }
                    CompactionInfo::ProviderNative { provider, .. } => {
                        (format!("provider_native:{}", provider.0), None)
                    }
                };
                if let Some(regex) = search_regex.as_ref() {
                    let matched = regex.is_match(&strategy)
                        || summary.as_deref().is_some_and(|text| regex.is_match(text));
                    if !matched {
                        continue;
                    }
                }
                matching_entries.push(ThreadHistoryEntry::Compaction {
                    message_index,
                    strategy,
                    summary: summary.map(|text| truncate_text(&text, max_content_chars)),
                });
            }
            Message::SubagentResult { results, .. } => {
                if results.is_empty() || !allow_kind(ThreadHistoryEntryKindFilter::SubagentResult) {
                    continue;
                }
                if let Some(regex) = search_regex.as_ref() {
                    let matched = results.iter().any(|result| {
                        regex.is_match(result.label.as_ref())
                            || regex.is_match(&result.output)
                            || regex.is_match(result.session_id.0.as_ref())
                    });
                    if !matched {
                        continue;
                    }
                }
                let formatted_results = results
                    .iter()
                    .map(|result| ThreadHistorySubagentOutcome {
                        session_id: result.session_id.clone(),
                        label: result.label.to_string(),
                        status: format!("{:?}", result.status).to_ascii_lowercase(),
                        output: truncate_text(&result.output, max_content_chars),
                    })
                    .collect();
                matching_entries.push(ThreadHistoryEntry::SubagentResult {
                    message_index,
                    results: formatted_results,
                });
            }
            Message::Resume => {}
        }
    }

    let total_matching_entries = matching_entries.len();
    let truncated = total_matching_entries > limit;
    let next_message_index;
    let entries = if truncated {
        if tail {
            next_message_index = None;
            matching_entries.split_off(total_matching_entries - limit)
        } else {
            let last_returned_ix = matching_entries[limit - 1].message_index();
            next_message_index = matching_entries[limit..]
                .iter()
                .find(|entry| entry.message_index() > last_returned_ix)
                .map(ThreadHistoryEntry::message_index);
            matching_entries.truncate(limit);
            matching_entries
        }
    } else {
        next_message_index = None;
        matching_entries
    };

    Ok(QueryThreadHistoryToolOutput::Entries(Box::new(
        ThreadHistoryQueryDetail {
            thread_id,
            title,
            total_messages,
            compaction_indices,
            scanned_range: (scan_start, scan_end),
            total_matching_entries,
            returned_entries: entries.len(),
            truncated,
            next_message_index,
            entries,
        },
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentMessage, UserMessage};
    use acp_thread::ClientUserMessageId;
    use chrono::Utc;
    use collections::IndexMap;
    use language_model::{
        LanguageModelToolResult, LanguageModelToolUse, LanguageModelToolUseId,
        LanguageModelToolUseInput,
    };
    use util::path_list::PathList;

    fn sample_messages() -> Vec<Arc<Message>> {
        let mut tool_results = IndexMap::default();
        let tool_id = LanguageModelToolUseId::from("tool-1");
        tool_results.insert(
            tool_id.clone(),
            LanguageModelToolResult {
                tool_use_id: tool_id.clone(),
                tool_name: Arc::from("terminal"),
                is_error: true,
                content: vec![LanguageModelToolResultContent::Text(Arc::from(
                    "error[E0308]: mismatched types in crates/agent/src/thread.rs:42",
                ))],
                output: None,
            },
        );

        vec![
            Arc::new(Message::User(UserMessage {
                id: ClientUserMessageId::new(),
                content: Arc::from([UserMessageContent::Text(
                    "Fix the compaction bug in thread.rs".to_string(),
                )]),
            })),
            Arc::new(Message::Agent(AgentMessage {
                content: vec![
                    AgentMessageContent::Thinking {
                        text: "Checking compiler diagnostics first".to_string(),
                        signature: None,
                    },
                    AgentMessageContent::ToolUse(LanguageModelToolUse {
                        id: tool_id,
                        name: Arc::from("terminal"),
                        raw_input: r#"{"command":"cargo test -p agent"}"#.to_string(),
                        input: LanguageModelToolUseInput::Json(
                            serde_json::json!({"command": "cargo test -p agent"}),
                        ),
                        is_input_complete: true,
                        thought_signature: None,
                    }),
                    AgentMessageContent::Text("Found the E0308 error.".to_string()),
                ],
                tool_results,
                reasoning_details: None,
            })),
            Arc::new(Message::Compaction(CompactionInfo::Summary(
                "Summary: fixed compaction".into(),
            ))),
            Arc::new(Message::User(UserMessage {
                id: ClientUserMessageId::new(),
                content: Arc::from([UserMessageContent::Text(
                    "What was the exact rustc error?".to_string(),
                )]),
            })),
        ]
    }

    #[test]
    fn test_before_latest_compaction_and_regex_recall() {
        let messages = sample_messages();
        let input = QueryThreadHistoryToolInput {
            before_latest_compaction: true,
            search_regex: Some("E0308".to_string()),
            tool_names: Some(vec!["terminal".to_string()]),
            ..Default::default()
        };

        let output = evaluate_thread_history_messages(
            acp_v2::SessionId::new("thread-1"),
            Some("Compaction work".to_string()),
            &messages,
            &input,
        )
        .expect("query succeeds");

        let QueryThreadHistoryToolOutput::Entries(detail) = output else {
            panic!("expected entries output");
        };
        assert_eq!(detail.total_messages, 4);
        assert_eq!(detail.compaction_indices, vec![2]);
        assert_eq!(detail.scanned_range, (0, 2));
        assert_eq!(detail.returned_entries, 1);
        match &detail.entries[0] {
            ThreadHistoryEntry::ToolCall {
                message_index,
                tool_name,
                output,
                is_error,
                ..
            } => {
                assert_eq!(*message_index, 1);
                assert_eq!(tool_name, "terminal");
                assert!(*is_error);
                assert!(
                    output
                        .as_deref()
                        .is_some_and(|text| text.contains("error[E0308]"))
                );
            }
            other => panic!("expected tool call entry, got {other:?}"),
        }
    }

    #[test]
    fn test_list_threads_filters_subagents_workspace_and_query() {
        let now = Utc::now();
        let threads = vec![
            DbThreadMetadata {
                id: acp_v2::SessionId::new("t-main"),
                parent_session_id: None,
                title: "Refactor compaction pointer".into(),
                updated_at: now,
                created_at: Some(now),
                folder_paths: PathList::new(&[PathBuf::from("/work/zed")]),
            },
            DbThreadMetadata {
                id: acp_v2::SessionId::new("t-sub"),
                parent_session_id: Some(acp_v2::SessionId::new("t-main")),
                title: "Subagent compaction check".into(),
                updated_at: now,
                created_at: Some(now),
                folder_paths: PathList::new(&[PathBuf::from("/work/zed")]),
            },
            DbThreadMetadata {
                id: acp_v2::SessionId::new("t-other"),
                parent_session_id: None,
                title: "Compaction notes".into(),
                updated_at: now,
                created_at: Some(now),
                folder_paths: PathList::new(&[PathBuf::from("/work/other")]),
            },
        ];

        let output = evaluate_thread_history_list(
            &threads,
            &[PathBuf::from("/work/zed")],
            &QueryThreadHistoryToolInput {
                list_threads: true,
                workspace_only: true,
                thread_query: Some("compaction".to_string()),
                ..Default::default()
            },
        )
        .expect("list succeeds");

        let QueryThreadHistoryToolOutput::ThreadList {
            total_threads,
            matched_threads,
            threads,
            ..
        } = output
        else {
            panic!("expected thread list");
        };
        assert_eq!(total_threads, 2);
        assert_eq!(matched_threads, 1);
        assert_eq!(threads[0].thread_id, acp_v2::SessionId::new("t-main"));
    }
}
