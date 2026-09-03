use std::collections::BTreeMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use acp_thread::{
    AcpThread, AgentThreadEntry, AssistantMessageChunk, SessionMessageSide, ThreadStatus,
    ToolCallStatus,
};
use agent_client_protocol::v2::schema as acp_v2;
use anyhow::{Result, anyhow};
use gpui::{App, Entity, SharedString, Task};
use language::unified_diff;
use language_model::LanguageModelToolResultContent;
use project::Project;
use regex::RegexBuilder;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use util::paths::{PathMatcher, PathStyle};
use util::rel_path::RelPath;

use crate::{AgentTool, Thread, ThreadEnvironment, ToolCallEventStream, ToolInput};

const DEFAULT_QUERY_LIMIT: usize = 25;
const MAX_QUERY_LIMIT: usize = 100;
const DEFAULT_MAX_CONTENT_CHARS: usize = 2_000;
const MAX_ALLOWED_CONTENT_CHARS: usize = 16_000;
const PREVIEW_MAX_CHARS: usize = 240;

fn deserialize_session_id<'de, D>(deserializer: D) -> Result<Option<acp_v2::SessionId>, D::Error>
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
pub enum SubagentEntryKindFilter {
    UserMessage,
    AssistantMessage,
    Thinking,
    ToolCall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SubagentToolStatusFilter {
    Pending,
    WaitingForConfirmation,
    InProgress,
    Paused,
    Completed,
    Failed,
    Rejected,
    Canceled,
}

impl SubagentToolStatusFilter {
    fn matches(self, status: ToolCallStatus) -> bool {
        matches!(
            (self, status),
            (Self::Pending, ToolCallStatus::Pending)
                | (
                    Self::WaitingForConfirmation,
                    ToolCallStatus::WaitingForConfirmation
                )
                | (Self::InProgress, ToolCallStatus::InProgress)
                | (Self::Paused, ToolCallStatus::Paused)
                | (Self::Completed, ToolCallStatus::Completed)
                | (Self::Failed, ToolCallStatus::Failed)
                | (Self::Rejected, ToolCallStatus::Rejected)
                | (Self::Canceled, ToolCallStatus::Canceled)
        )
    }
}

/// Inspect and query the live or completed activity, tool calls, terminal output,
/// reasoning, and file changes of sub-agents in this thread.
///
/// ### Usage modes
/// - **Overview of all sub-agents:** Omit `session_id` to list every sub-agent spawned in
///   this thread with its status, queued steering count, model, token usage, tool-call
///   counts, active tool calls, file diff stats, and latest message preview.
/// - **Structured query on a sub-agent:** Pass `session_id` along with any combination of
///   filters (`entry_kinds`, `tool_names`, `statuses`, `path_glob`, `search_regex`,
///   `from_entry`, `to_entry`, `limit`, `tail`) and file-inspection flags
///   (`include_file_changes`, `include_diffs`).
/// - **Incremental progress tracking:** Save `next_entry_index` from a response and pass
///   it as `from_entry` on a later call to inspect only new actions since your last query.
/// - **Course-correcting a running sub-agent:** After inspecting a running sub-agent's
///   actions, diffs, or live terminal output, call `spawn_agent` with its `session_id`
///   (and `interrupt: true` if it should stop its current step immediately) to steer it.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct QuerySubagentToolInput {
    /// Session ID of a specific sub-agent to query. Omit to list a summary of all
    /// sub-agents in this thread.
    #[serde(default, deserialize_with = "deserialize_session_id")]
    pub session_id: Option<acp_v2::SessionId>,

    /// Entry categories to include when querying a specific `session_id`.
    /// Omit to include all categories; pass `[]` to fetch only summary and file changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_kinds: Option<Vec<SubagentEntryKindFilter>>,

    /// Filter tool-call entries to tools whose name matches one of these names
    /// (e.g. `["edit_file", "write_file", "terminal"]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_names: Option<Vec<String>>,

    /// Filter tool-call entries by execution status
    /// (e.g. `["in_progress", "waiting_for_confirmation", "failed"]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub statuses: Option<Vec<SubagentToolStatusFilter>>,

    /// Glob pattern matched against file paths referenced by tool calls and `file_changes`
    /// (e.g. `"**/thread*.rs"` or `"crates/agent/**"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_glob: Option<String>,

    /// Regular expression (Rust `regex` syntax) matched against message text, thinking
    /// text, tool inputs, tool outputs, and live terminal commands/output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_regex: Option<String>,

    /// Whether `search_regex` matching is case-sensitive (default: `false`).
    #[serde(default)]
    pub case_sensitive: bool,

    /// Inclusive 0-based entry index to start scanning from. Pass `next_entry_index`
    /// from a previous call to read only new entries since that check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_entry: Option<usize>,

    /// Exclusive 0-based entry index to stop scanning at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_entry: Option<usize>,

    /// Maximum number of matching entries to return (default: 25, max: 100).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,

    /// When `true` (default when `from_entry` is omitted), returns the most recent
    /// `limit` matching entries; when `false` (default when `from_entry` is set),
    /// returns the earliest `limit` matching entries from `from_entry` forward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail: Option<bool>,

    /// Include the sub-agent's file read/modification summary from its ActionLog (default: `true`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_file_changes: Option<bool>,

    /// Include unified diffs for modified/created files in `file_changes` (default: `false`).
    #[serde(default)]
    pub include_diffs: bool,

    /// Maximum characters per entry body/input/output/diff before truncation (default: 2000, max: 16000).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_content_chars: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentTokenUsageSummary {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub max_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentFileStat {
    pub path: String,
    pub lines_added: u32,
    pub lines_removed: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentChangedFile {
    pub path: String,
    pub lines_added: u32,
    pub lines_removed: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diff: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentSummary {
    pub session_id: acp_v2::SessionId,
    pub label: String,
    pub status: String,
    pub pending_steers: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_usage: Option<SubagentTokenUsageSummary>,
    pub total_entries: usize,
    pub tool_call_counts: BTreeMap<String, usize>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub active_tool_calls: Vec<String>,
    pub files_read_count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub files_changed: Vec<SubagentFileStat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest_message_preview: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentTerminalSnapshot {
    pub command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub working_dir: Option<String>,
    pub running: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signal: Option<String>,
    pub elapsed_ms: u64,
    pub output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SubagentQueriedEntry {
    UserMessage {
        entry_index: usize,
        text: String,
    },
    AssistantMessage {
        entry_index: usize,
        text: String,
    },
    Thinking {
        entry_index: usize,
        text: String,
    },
    ToolCall {
        entry_index: usize,
        tool_call_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_name: Option<String>,
        title: String,
        status: String,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        locations: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        input: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        terminals: Vec<SubagentTerminalSnapshot>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentQueryDetail {
    pub summary: SubagentSummary,
    pub total_matching_entries: usize,
    pub returned_entries: usize,
    pub next_entry_index: usize,
    pub entries: Vec<SubagentQueriedEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files_read: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_changes: Option<Vec<SubagentChangedFile>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum QuerySubagentToolOutput {
    Detail(Box<SubagentQueryDetail>),
    Overview { subagents: Vec<SubagentSummary> },
    Error { error: String },
}

impl From<QuerySubagentToolOutput> for LanguageModelToolResultContent {
    fn from(output: QuerySubagentToolOutput) -> Self {
        serde_json::to_string_pretty(&output)
            .unwrap_or_else(|error| format!("Failed to serialize query_subagent output: {error}"))
            .into()
    }
}

pub struct QuerySubagentTool {
    environment: Rc<dyn ThreadEnvironment>,
}

impl QuerySubagentTool {
    pub fn new(environment: Rc<dyn ThreadEnvironment>) -> Self {
        Self { environment }
    }
}

impl AgentTool for QuerySubagentTool {
    type Input = QuerySubagentToolInput;
    type Output = QuerySubagentToolOutput;

    const NAME: &'static str = "query_subagent";

    fn kind() -> acp_v2::ToolKind {
        acp_v2::ToolKind::Read
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(input) => match input.session_id {
                Some(session_id) => {
                    let short_id: String = session_id.0.chars().take(8).collect();
                    if let Some(regex) = input.search_regex.as_deref() {
                        format!("Query subagent {short_id} (/{regex}/)").into()
                    } else if let Some(tools) = input.tool_names.as_ref()
                        && !tools.is_empty()
                    {
                        format!("Query subagent {short_id} ({})", tools.join(", ")).into()
                    } else {
                        format!("Query subagent {short_id}").into()
                    }
                }
                None => "Query all subagents".into(),
            },
            Err(_) => "Query subagent".into(),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        cx.spawn(async move |cx| {
            let input = input
                .recv()
                .await
                .map_err(|error| QuerySubagentToolOutput::Error {
                    error: error.to_string(),
                })?;

            let result = cx.update(|cx| self.environment.query_subagents(input, cx));
            match result {
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
                    Err(QuerySubagentToolOutput::Error { error: error_text })
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
            QuerySubagentToolOutput::Error { error } => error.clone(),
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

fn status_slug(status: ToolCallStatus) -> &'static str {
    match status {
        ToolCallStatus::Pending => "pending",
        ToolCallStatus::WaitingForConfirmation => "waiting_for_confirmation",
        ToolCallStatus::InProgress => "in_progress",
        ToolCallStatus::Paused => "paused",
        ToolCallStatus::Completed => "completed",
        ToolCallStatus::Failed => "failed",
        ToolCallStatus::Rejected => "rejected",
        ToolCallStatus::Canceled => "canceled",
    }
}

fn format_project_or_abs_path(path: &Path, project: &Entity<Project>, cx: &App) -> String {
    let project_ref = project.read(cx);
    let path_style = project_ref.path_style(cx);
    if let Some(project_path) = project_ref.project_path_for_absolute_path(path, cx)
        && let Some(worktree) = project_ref.worktree_for_id(project_path.worktree_id, cx)
    {
        return worktree
            .read(cx)
            .root_name()
            .join(&project_path.path)
            .display(path_style)
            .to_string();
    }
    path.display().to_string()
}

fn extract_input_paths(raw_input: &serde_json::Value) -> Vec<String> {
    let Some(object) = raw_input.as_object() else {
        return Vec::new();
    };
    let mut paths = Vec::new();
    for key in [
        "path",
        "source_path",
        "destination_path",
        "include_pattern",
        "glob",
        "cd",
    ] {
        if let Some(value) = object.get(key).and_then(|value| value.as_str()) {
            paths.push(value.to_string());
        }
    }
    paths
}

fn summarize_subagent(
    subagent_thread: Option<&Entity<Thread>>,
    acp_thread: &Entity<AcpThread>,
    project: &Entity<Project>,
    cx: &App,
) -> SubagentSummary {
    let acp_ref = acp_thread.read(cx);
    let session_id = acp_ref.session_id().clone();
    let label = acp_ref
        .title()
        .or_else(|| subagent_thread.and_then(|thread| thread.read(cx).title()))
        .map(|title| title.to_string())
        .unwrap_or_else(|| "Subagent".to_string());

    let is_waiting_permission = acp_ref.is_waiting_for_confirmation();
    let is_active_run = subagent_thread
        .is_some_and(|thread| thread.read(cx).is_active_subagent_run())
        || acp_ref.status() == ThreadStatus::Generating;

    let status = if is_waiting_permission {
        "waiting_for_confirmation"
    } else if is_active_run {
        "running"
    } else if acp_ref.had_error() {
        "failed"
    } else {
        let last_stop = subagent_thread.and_then(|thread| {
            thread
                .read(cx)
                .subagent_outcomes()
                .last()
                .map(|outcome| outcome.stop_reason.clone())
        });
        match last_stop {
            Some(acp_v2::StopReason::Cancelled) => "canceled",
            Some(
                acp_v2::StopReason::Error(_)
                | acp_v2::StopReason::MaxTokens
                | acp_v2::StopReason::MaxTurnRequests
                | acp_v2::StopReason::Refusal,
            ) => "failed",
            Some(acp_v2::StopReason::EndTurn) => "completed",
            _ if acp_ref.entries().is_empty() => "idle",
            _ => "completed",
        }
    }
    .to_string();

    let pending_steers =
        subagent_thread.map_or(0, |thread| thread.read(cx).pending_subagent_steers_count());
    let model = subagent_thread
        .and_then(|thread| thread.read(cx).model())
        .map(|model| format!("{}/{}", model.provider_id(), model.id.0));
    let token_usage = subagent_thread
        .and_then(|thread| thread.read(cx).latest_token_usage())
        .map(|usage| SubagentTokenUsageSummary {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            max_tokens: usage.max_tokens,
        });

    let mut tool_call_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut active_tool_calls = Vec::new();
    let mut latest_message_preview = None;

    for entry in acp_ref.entries() {
        match entry {
            AgentThreadEntry::ToolCall(call) => {
                let tool_name = call
                    .tool_name
                    .as_ref()
                    .or(call.name.as_ref())
                    .map(|name| name.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                *tool_call_counts.entry(tool_name.clone()).or_insert(0) += 1;
                if matches!(
                    call.status(),
                    ToolCallStatus::Pending
                        | ToolCallStatus::WaitingForConfirmation
                        | ToolCallStatus::InProgress
                ) {
                    let title = call.label.read(cx).source().to_string();
                    active_tool_calls.push(format!(
                        "{tool_name}: {title} ({})",
                        status_slug(call.status())
                    ));
                }
            }
            AgentThreadEntry::AssistantMessage(message) => {
                if let Some(chunk) = message.chunks.last() {
                    let text = match chunk {
                        AssistantMessageChunk::Message { block, .. }
                        | AssistantMessageChunk::Thought { block, .. } => block.to_markdown(cx),
                    };
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        latest_message_preview = Some(truncate_text(trimmed, PREVIEW_MAX_CHARS));
                    }
                }
            }
            _ => {}
        }
    }

    let action_log = acp_ref.action_log().read(cx);
    let files_read_count = action_log.read_files().count();
    let mut files_changed = action_log
        .changed_buffers(cx)
        .filter_map(|(buffer, diff)| {
            let file = buffer.read(cx).file()?;
            let path = file.full_path(cx).display().to_string();
            let stats = action_log::DiffStats::single_file(diff.read(cx));
            Some(SubagentFileStat {
                path,
                lines_added: stats.lines_added,
                lines_removed: stats.lines_removed,
            })
        })
        .collect::<Vec<_>>();
    files_changed.sort_by(|left, right| left.path.cmp(&right.path));
    let _ = project;

    SubagentSummary {
        session_id,
        label,
        status,
        pending_steers,
        model,
        token_usage,
        total_entries: acp_ref.entries().len(),
        tool_call_counts,
        active_tool_calls,
        files_read_count,
        files_changed,
        latest_message_preview,
    }
}

pub(crate) fn evaluate_subagent_query(
    subagents: &[(Option<Entity<Thread>>, Entity<AcpThread>)],
    project: &Entity<Project>,
    input: QuerySubagentToolInput,
    cx: &App,
) -> Result<QuerySubagentToolOutput> {
    let Some(target_session_id) = input.session_id.as_ref() else {
        let summaries = subagents
            .iter()
            .map(|(thread, acp_thread)| {
                summarize_subagent(thread.as_ref(), acp_thread, project, cx)
            })
            .collect();
        return Ok(QuerySubagentToolOutput::Overview {
            subagents: summaries,
        });
    };

    let Some((subagent_thread, acp_thread)) = subagents
        .iter()
        .find(|(_, acp_thread)| acp_thread.read(cx).session_id() == target_session_id)
    else {
        return Err(anyhow!(
            "No subagent session found with id {target_session_id} on this thread"
        ));
    };

    let path_style: PathStyle = project.read(cx).path_style(cx);
    let path_matcher = match input.path_glob.as_deref() {
        Some(glob) if !glob.trim().is_empty() => Some(
            PathMatcher::new([glob], path_style)
                .map_err(|error| anyhow!("Invalid path_glob '{glob}': {error}"))?,
        ),
        _ => None,
    };

    let regex = match input.search_regex.as_deref() {
        Some(pattern) if !pattern.is_empty() => Some(
            RegexBuilder::new(pattern)
                .case_insensitive(!input.case_sensitive)
                .build()
                .map_err(|error| anyhow!("Invalid search_regex '{pattern}': {error}"))?,
        ),
        _ => None,
    };

    let max_chars = input
        .max_content_chars
        .unwrap_or(DEFAULT_MAX_CONTENT_CHARS)
        .clamp(64, MAX_ALLOWED_CONTENT_CHARS);
    let limit = input
        .limit
        .unwrap_or(DEFAULT_QUERY_LIMIT)
        .clamp(1, MAX_QUERY_LIMIT);
    let use_tail = input.tail.unwrap_or(input.from_entry.is_none());

    let acp_ref = acp_thread.read(cx);
    let entries = acp_ref.entries();
    let total_entries = entries.len();
    let start_ix = input.from_entry.unwrap_or(0).min(total_entries);
    let end_ix = input.to_entry.unwrap_or(total_entries).min(total_entries);

    let allows_kind = |kind: SubagentEntryKindFilter| -> bool {
        input
            .entry_kinds
            .as_ref()
            .is_none_or(|kinds| kinds.contains(&kind))
    };

    let matches_path = |candidates: &[String]| -> bool {
        let Some(matcher) = path_matcher.as_ref() else {
            return true;
        };
        candidates.iter().any(|candidate| {
            let trimmed = candidate.trim_start_matches(['/', '\\']);
            RelPath::new(Path::new(trimmed), path_style)
                .is_ok_and(|rel_path| matcher.is_match(rel_path))
        })
    };

    let matches_regex = |haystacks: &[&str]| -> bool {
        let Some(regex) = regex.as_ref() else {
            return true;
        };
        haystacks.iter().any(|text| regex.is_match(text))
    };

    let mut matched_entries = Vec::new();

    if start_ix < end_ix {
        for (entry_index, entry) in entries[start_ix..end_ix]
            .iter()
            .enumerate()
            .map(|(offset, entry)| (start_ix + offset, entry))
        {
            match entry {
                AgentThreadEntry::UserMessage(user_message) => {
                    if !allows_kind(SubagentEntryKindFilter::UserMessage)
                        || path_matcher.is_some()
                        || input.tool_names.is_some()
                        || input.statuses.is_some()
                    {
                        continue;
                    }
                    let full_text = user_message.content.to_markdown(cx);
                    if !matches_regex(&[full_text.as_str()]) {
                        continue;
                    }
                    matched_entries.push(SubagentQueriedEntry::UserMessage {
                        entry_index,
                        text: truncate_text(&full_text, max_chars),
                    });
                }
                AgentThreadEntry::SessionMessage(session_message) => {
                    if !matches!(
                        session_message.side(acp_ref.session_id()),
                        SessionMessageSide::Incoming { .. }
                    ) {
                        continue;
                    }
                    if !allows_kind(SubagentEntryKindFilter::UserMessage)
                        || path_matcher.is_some()
                        || input.tool_names.is_some()
                        || input.statuses.is_some()
                    {
                        continue;
                    }
                    let full_text = session_message.content.to_markdown(cx);
                    if !matches_regex(&[full_text.as_str()]) {
                        continue;
                    }
                    matched_entries.push(SubagentQueriedEntry::UserMessage {
                        entry_index,
                        text: truncate_text(&full_text, max_chars),
                    });
                }
                AgentThreadEntry::AssistantMessage(assistant_message) => {
                    if path_matcher.is_some()
                        || input.tool_names.is_some()
                        || input.statuses.is_some()
                    {
                        continue;
                    }
                    for chunk in &assistant_message.chunks {
                        match chunk {
                            AssistantMessageChunk::Message { block, .. } => {
                                if !allows_kind(SubagentEntryKindFilter::AssistantMessage) {
                                    continue;
                                }
                                let full_text = block.to_markdown(cx);
                                if full_text.trim().is_empty()
                                    || !matches_regex(&[full_text.as_str()])
                                {
                                    continue;
                                }
                                matched_entries.push(SubagentQueriedEntry::AssistantMessage {
                                    entry_index,
                                    text: truncate_text(&full_text, max_chars),
                                });
                            }
                            AssistantMessageChunk::Thought { block, .. } => {
                                if !allows_kind(SubagentEntryKindFilter::Thinking) {
                                    continue;
                                }
                                let full_text = block.to_markdown(cx);
                                if full_text.trim().is_empty()
                                    || !matches_regex(&[full_text.as_str()])
                                {
                                    continue;
                                }
                                matched_entries.push(SubagentQueriedEntry::Thinking {
                                    entry_index,
                                    text: truncate_text(&full_text, max_chars),
                                });
                            }
                        }
                    }
                }
                AgentThreadEntry::ToolCall(call) => {
                    if !allows_kind(SubagentEntryKindFilter::ToolCall) {
                        continue;
                    }
                    let tool_name = call
                        .tool_name
                        .as_ref()
                        .or(call.name.as_ref())
                        .map(|name| name.to_string());
                    if let Some(allowed_tools) = input.tool_names.as_ref()
                        && !allowed_tools.is_empty()
                        && !tool_name
                            .as_ref()
                            .is_some_and(|name| allowed_tools.iter().any(|allowed| allowed == name))
                    {
                        continue;
                    }
                    if let Some(allowed_statuses) = input.statuses.as_ref()
                        && !allowed_statuses.is_empty()
                        && !allowed_statuses
                            .iter()
                            .any(|filter| filter.matches(call.status()))
                    {
                        continue;
                    }

                    let mut locations = call
                        .locations
                        .iter()
                        .map(|location| format_project_or_abs_path(&location.path, project, cx))
                        .collect::<Vec<_>>();
                    let mut candidate_paths = locations.clone();
                    if let Some(raw_input) = call.raw_input.as_ref() {
                        candidate_paths.extend(extract_input_paths(raw_input));
                    }
                    if !matches_path(&candidate_paths) {
                        continue;
                    }
                    locations.sort();
                    locations.dedup();

                    let title = call.label.read(cx).source().to_string();
                    let raw_input_text = call
                        .raw_input
                        .as_ref()
                        .and_then(|value| serde_json::to_string(value).ok());
                    let content_markdown = call
                        .content()
                        .iter()
                        .map(|content| content.to_markdown(cx))
                        .collect::<Vec<_>>()
                        .join("\n\n");

                    let mut terminals = Vec::new();
                    let mut terminal_search_texts = Vec::new();
                    for terminal_entity in call.terminals() {
                        let terminal = terminal_entity.read(cx);
                        let raw_command = terminal.command().read(cx).source().to_string();
                        let command = raw_command
                            .trim_start_matches("```\n")
                            .trim_end_matches("\n```")
                            .to_string();
                        let working_dir = terminal
                            .working_dir()
                            .as_ref()
                            .map(|dir| format_project_or_abs_path(dir, project, cx));
                        let (running, exit_code, signal, full_output) = match terminal.output() {
                            Some(finished) => (
                                false,
                                finished.exit_status.exit_code,
                                finished.exit_status.signal.clone(),
                                finished.content.clone(),
                            ),
                            None => (true, None, None, terminal.inner().read(cx).get_content()),
                        };
                        let elapsed_ms = terminal.started_at().elapsed().as_millis() as u64;
                        terminal_search_texts.push(command.clone());
                        terminal_search_texts.push(full_output.clone());
                        terminals.push(SubagentTerminalSnapshot {
                            command,
                            working_dir,
                            running,
                            exit_code,
                            signal,
                            elapsed_ms,
                            output: truncate_text(&full_output, max_chars),
                        });
                    }

                    let mut regex_haystacks: Vec<&str> = vec![title.as_str()];
                    if let Some(name) = tool_name.as_deref() {
                        regex_haystacks.push(name);
                    }
                    if let Some(input_str) = raw_input_text.as_deref() {
                        regex_haystacks.push(input_str);
                    }
                    if !content_markdown.is_empty() {
                        regex_haystacks.push(content_markdown.as_str());
                    }
                    for text in &terminal_search_texts {
                        regex_haystacks.push(text.as_str());
                    }
                    for path in &candidate_paths {
                        regex_haystacks.push(path.as_str());
                    }
                    if !matches_regex(&regex_haystacks) {
                        continue;
                    }

                    let output = if content_markdown.trim().is_empty() || !terminals.is_empty() {
                        None
                    } else {
                        Some(truncate_text(&content_markdown, max_chars))
                    };

                    matched_entries.push(SubagentQueriedEntry::ToolCall {
                        entry_index,
                        tool_call_id: call.id.0.to_string(),
                        tool_name,
                        title,
                        status: status_slug(call.status()).to_string(),
                        locations,
                        input: raw_input_text.map(|text| truncate_text(&text, max_chars)),
                        output,
                        terminals,
                    });
                }
                AgentThreadEntry::Elicitation(_) | AgentThreadEntry::ContextCompaction(_) => {}
            }
        }
    }

    let total_matching_entries = matched_entries.len();
    let sliced_entries = if total_matching_entries <= limit {
        matched_entries
    } else if use_tail {
        matched_entries.split_off(total_matching_entries - limit)
    } else {
        matched_entries.truncate(limit);
        matched_entries
    };

    let include_files = input.include_file_changes.unwrap_or(true) || input.include_diffs;
    let (files_read, file_changes) = if include_files {
        let action_log = acp_ref.action_log().read(cx);
        let mut read_paths = action_log
            .read_files()
            .map(|path| format_project_or_abs_path(path, project, cx))
            .filter(|path| matches_path(std::slice::from_ref(path)))
            .collect::<Vec<_>>();
        read_paths.sort();
        read_paths.dedup();

        let mut changes = action_log
            .changed_buffers(cx)
            .filter_map(|(buffer, diff_entity)| {
                let buffer_ref = buffer.read(cx);
                let file = buffer_ref.file()?;
                let path = file.full_path(cx).display().to_string();
                if !matches_path(std::slice::from_ref(&path)) {
                    return None;
                }
                let diff_ref = diff_entity.read(cx);
                let stats = action_log::DiffStats::single_file(diff_ref);
                let diff = if input.include_diffs {
                    let old_text = diff_ref.base_text_string(cx).unwrap_or_default();
                    let new_text = buffer_ref.text();
                    let unified = unified_diff(&old_text, &new_text);
                    if let Some(regex) = regex.as_ref()
                        && !regex.is_match(&unified)
                        && !regex.is_match(&path)
                    {
                        return None;
                    }
                    Some(truncate_text(&unified, max_chars))
                } else {
                    None
                };
                Some(SubagentChangedFile {
                    path,
                    lines_added: stats.lines_added,
                    lines_removed: stats.lines_removed,
                    diff,
                })
            })
            .collect::<Vec<_>>();
        changes.sort_by(|left, right| left.path.cmp(&right.path));
        (Some(read_paths), Some(changes))
    } else {
        (None, None)
    };

    let summary = summarize_subagent(subagent_thread.as_ref(), acp_thread, project, cx);

    Ok(QuerySubagentToolOutput::Detail(Box::new(
        SubagentQueryDetail {
            summary,
            total_matching_entries,
            returned_entries: sliced_entries.len(),
            next_entry_index: total_entries,
            entries: sliced_entries,
            files_read,
            file_changes,
        },
    )))
}
