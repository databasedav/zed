use std::rc::Rc;

use acp_thread::AcpThread;
use agent_client_protocol::v2::schema as acp_v2;
use anyhow::Result;
use gpui::{App, Entity, Global, Task};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeSubagentModelInfo {
    pub id: String,
    pub name: String,
    pub provider_id: String,
    pub provider_name: String,
    pub is_default: bool,
    pub supports_thinking: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supported_efforts: Vec<String>,
    #[serde(default)]
    pub supports_fast_mode: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_speed: Option<String>,
}

pub struct NativeSubagentSpawnRequest {
    pub parent_acp_thread: Entity<AcpThread>,
    pub tool_call_id: String,
    pub label: String,
    pub message: String,
    pub session_id: Option<acp_v2::SessionId>,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub speed: Option<String>,
    pub interrupt: bool,
    pub on_finished: Box<dyn FnOnce(NativeSubagentFinishedEvent) + 'static>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeSubagentSpawnResponse {
    pub session_id: acp_v2::SessionId,
    pub status: String,
    pub model_id: String,
    pub model_name: String,
    pub thinking_effort: Option<String>,
    pub speed: Option<String>,
    pub message_start_index: usize,
    pub interrupt: bool,
    pub pending_steers: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeSubagentFinishedEvent {
    pub parent_session_id: acp_v2::SessionId,
    pub session_id: acp_v2::SessionId,
    pub tool_call_id: String,
    pub label: String,
    pub status: String,
    pub output: String,
    pub message_end_index: Option<usize>,
    pub model_name: String,
    pub thinking_effort: Option<String>,
    pub speed: Option<String>,
    pub canceled_by_parent_stop: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeSubagentQueryRequest {
    #[serde(default)]
    pub session_id: Option<acp_v2::SessionId>,
    #[serde(default)]
    pub entry_kinds: Option<Vec<String>>,
    #[serde(default)]
    pub tool_names: Option<Vec<String>>,
    #[serde(default)]
    pub statuses: Option<Vec<String>>,
    #[serde(default)]
    pub path_glob: Option<String>,
    #[serde(default)]
    pub search_regex: Option<String>,
    #[serde(default)]
    pub case_sensitive: bool,
    #[serde(default)]
    pub from_entry: Option<usize>,
    #[serde(default)]
    pub to_entry: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub tail: Option<bool>,
    #[serde(default)]
    pub include_file_changes: bool,
    #[serde(default)]
    pub include_diffs: bool,
    #[serde(default)]
    pub max_content_chars: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeThreadHistoryQueryRequest {
    #[serde(default)]
    pub thread_id: Option<acp_v2::SessionId>,
    #[serde(default)]
    pub list_threads: bool,
    #[serde(default)]
    pub thread_query: Option<String>,
    #[serde(default)]
    pub workspace_only: bool,
    #[serde(default)]
    pub include_subagents: bool,
    #[serde(default)]
    pub before_latest_compaction: bool,
    #[serde(default)]
    pub entry_kinds: Option<Vec<String>>,
    #[serde(default)]
    pub tool_names: Option<Vec<String>>,
    #[serde(default)]
    pub search_regex: Option<String>,
    #[serde(default)]
    pub case_sensitive: bool,
    #[serde(default)]
    pub from_index: Option<usize>,
    #[serde(default)]
    pub to_index: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub tail: Option<bool>,
    #[serde(default)]
    pub max_content_chars: Option<usize>,
}

/// Bridge installed by `agent_ui` so external ACP connections can delegate subagents
/// and thread-history queries to Zed's native `NativeAgent`.
pub trait NativeSubagentBridge: 'static {
    fn list_models(&self, cx: &mut App) -> Vec<NativeSubagentModelInfo>;

    fn spawn_or_follow_up(
        &self,
        request: NativeSubagentSpawnRequest,
        cx: &mut App,
    ) -> Task<Result<NativeSubagentSpawnResponse>>;

    fn query(
        &self,
        parent_acp_thread: Entity<AcpThread>,
        request: NativeSubagentQueryRequest,
        cx: &mut App,
    ) -> Task<Result<String>>;

    fn query_thread_history(
        &self,
        parent_acp_thread: Entity<AcpThread>,
        request: NativeThreadHistoryQueryRequest,
        cx: &mut App,
    ) -> Task<Result<String>>;

    fn cancel(
        &self,
        parent_acp_thread: Entity<AcpThread>,
        session_id: acp_v2::SessionId,
        canceled_by_parent_stop: bool,
        cx: &mut App,
    ) -> Task<Result<bool>>;
}

pub struct GlobalNativeSubagentBridge(pub Rc<dyn NativeSubagentBridge>);

impl Global for GlobalNativeSubagentBridge {}
