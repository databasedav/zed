use acp_thread::{SUBAGENT_SESSION_INFO_META_KEY, SubagentSessionInfo};
use agent_client_protocol::v2::schema as acp_v2;
use anyhow::Result;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use std::rc::Rc;
use std::sync::Arc;

use acp_thread::AgentModelId;

use crate::{AgentTool, ThreadEnvironment, ToolCallEventStream, ToolInput};

/// Spawn a sub-agent for a well-scoped task.
///
/// `spawn_agent` starts the sub-agent asynchronously in the background and returns immediately
/// (`status: "running"`) so the main thread is never blocked. After calling `spawn_agent`,
/// proceed immediately with any independent work you can do on the main thread. If you have no
/// further independent work until a running sub-agent finishes, end your turn normally without
/// sleeping or polling—your thread will automatically resume with a `<subagent_result>` message
/// as soon as the sub-agent completes.
///
/// ### Designing delegated subtasks
/// - An agent does not see your conversation history. Include all relevant context (file paths, requirements, constraints) in the message.
/// - Subtasks must be concrete, well-defined, and self-contained.
/// - Delegated subtasks must materially advance the main task.
/// - Do not duplicate work between your work and delegated subtasks.
/// - Do not use this tool for tasks you could accomplish directly with one or two tool calls. For example, don't ask the agent to read a single file and return the contents, you can do this yourself.
/// - When you delegate work, focus on coordinating and synthesizing results instead of duplicating the same work yourself.
/// - Avoid issuing multiple delegate calls for the same unresolved subproblem unless the new delegated task is genuinely different and necessary.
/// - Narrow the delegated ask to the concrete output you need next.
/// - For code-edit subtasks, decompose work so each delegated task has a disjoint write set.
/// - When sending a follow-up using an existing agent session_id, the agent already has the context from the previous turn. Send only a short, direct message. Do NOT repeat the original task or context.
///   - If that sub-agent is **already running**, calling `spawn_agent` with its `session_id` **steers** it mid-run: by default (`interrupt: false`) the message is injected cleanly at the sub-agent's next tool-call/reasoning boundary without losing in-flight tool work; pass `interrupt: true` to stop its current step immediately and pivot it to the new instructions. You can inspect a sub-agent's live actions, diffs, and terminal output first with `query_subagent`.
///   - If that sub-agent has **completed**, calling `spawn_agent` with its `session_id` resumes it with a new background turn.
///
/// ### Parallel delegation patterns
/// - Run multiple independent information-seeking subtasks in parallel when you have distinct questions that can be answered independently.
/// - Split implementation into disjoint codebase slices and spawn multiple agents for them in parallel when the write scopes do not overlap.
/// - When a plan has multiple independent steps, prefer delegating those steps in parallel rather than serializing them unnecessarily.
/// - Reuse the returned session_id when you want to follow up on the same delegated subproblem instead of creating a duplicate session.
///
/// ### Model selection
/// - When the user requests a particular model or asks you to choose based on cost or capability, call `list_agents_and_models` first, then pass the exact `models[].id` from the native Zed agent entry (`is_native: true`) in `model`.
/// - Omit `model` to use the user's configured subagent model, or the parent model when no subagent model is configured.
/// - Do not silently choose a different model when an explicit model is unavailable unless the user allowed fallback.
/// - A resumed session keeps its existing model, so `model` cannot be combined with `session_id`.
///
/// ### Output
/// - Launching or resuming an idle sub-agent returns `session_id` and `status: "running"` immediately.
/// - Steering a running sub-agent returns `session_id`, `status: "steered"`, `interrupt`, and `pending_steers`; the sub-agent's final `<subagent_result>` will be delivered once the steered run finishes.
/// - When the sub-agent finishes, you will receive only its final message in `<subagent_result>` along with its `session_id` for any follow-up messages.
/// - Error results may also include a `session_id` if a session was already created.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct SpawnAgentToolInput {
    /// Short label displayed in the UI while the agent runs (e.g., "Researching alternatives")
    pub label: String,
    /// The prompt for the agent. For new sessions, include full context needed for the task. For follow-ups or mid-run steering (with session_id), you can rely on the agent already having the previous message.
    pub message: String,
    /// Session ID of an existing agent session to steer (if running) or continue (if completed), instead of creating a new one. Omit to create a new agent.
    #[serde(default, deserialize_with = "deserialize_session_id")]
    pub session_id: Option<acp_v2::SessionId>,
    /// Optional model override. Pass the exact `models[].id` returned for the
    /// native Zed agent (`is_native: true`) by `list_agents_and_models`.
    /// Omit to preserve default behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Only used when `session_id` refers to a currently running sub-agent:
    /// - `false` (default): steer cleanly at the sub-agent's next tool-call/reasoning boundary.
    /// - `true`: immediately interrupt the sub-agent's current turn/tool call and pivot to `message`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub interrupt: bool,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
pub enum SpawnAgentToolOutput {
    Success {
        session_id: acp_v2::SessionId,
        output: String,
        session_info: SubagentSessionInfo,
    },
    Error {
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(default)]
        session_id: Option<acp_v2::SessionId>,
        error: String,
        session_info: Option<SubagentSessionInfo>,
    },
    Canceled {
        session_id: acp_v2::SessionId,
        canceled: bool,
        session_info: SubagentSessionInfo,
    },
    Steered {
        session_id: acp_v2::SessionId,
        steered: bool,
        interrupt: bool,
        pending_steers: usize,
        message: String,
        session_info: SubagentSessionInfo,
    },
    Running {
        session_id: acp_v2::SessionId,
        session_info: SubagentSessionInfo,
    },
}

impl From<SpawnAgentToolOutput> for LanguageModelToolResultContent {
    fn from(output: SpawnAgentToolOutput) -> Self {
        match output {
            SpawnAgentToolOutput::Running {
                session_id,
                session_info: _,
            } => serde_json::to_string(
                &serde_json::json!({ "session_id": session_id, "status": "running" }),
            )
            .unwrap_or_else(|e| format!("Failed to serialize spawn_agent output: {e}"))
            .into(),
            SpawnAgentToolOutput::Steered {
                session_id,
                steered: _,
                interrupt,
                pending_steers,
                message: _,
                session_info: _,
            } => serde_json::to_string(&serde_json::json!({
                "session_id": session_id,
                "status": "steered",
                "interrupt": interrupt,
                "pending_steers": pending_steers,
            }))
            .unwrap_or_else(|e| format!("Failed to serialize spawn_agent output: {e}"))
            .into(),
            SpawnAgentToolOutput::Success {
                session_id,
                output,
                session_info: _, // Don't show this to the model
            } => serde_json::to_string(
                &serde_json::json!({ "session_id": session_id, "output": output }),
            )
            .unwrap_or_else(|e| format!("Failed to serialize spawn_agent output: {e}"))
            .into(),
            SpawnAgentToolOutput::Error {
                session_id,
                error,
                session_info: _, // Don't show this to the model
            } => serde_json::to_string(
                &serde_json::json!({ "session_id": session_id, "error": error }),
            )
            .unwrap_or_else(|e| format!("Failed to serialize spawn_agent output: {e}"))
            .into(),
            SpawnAgentToolOutput::Canceled {
                session_id,
                canceled: _,
                session_info: _,
            } => serde_json::to_string(
                &serde_json::json!({ "session_id": session_id, "error": "User canceled" }),
            )
            .unwrap_or_else(|e| format!("Failed to serialize spawn_agent output: {e}"))
            .into(),
        }
    }
}

/// Tool that spawns an agent thread to work on a task.
pub struct SpawnAgentTool {
    environment: Rc<dyn ThreadEnvironment>,
}

impl SpawnAgentTool {
    pub fn new(environment: Rc<dyn ThreadEnvironment>) -> Self {
        Self { environment }
    }
}

impl AgentTool for SpawnAgentTool {
    type Input = SpawnAgentToolInput;
    type Output = SpawnAgentToolOutput;

    const NAME: &'static str = "spawn_agent";

    fn kind() -> acp_v2::ToolKind {
        acp_v2::ToolKind::Other
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(i) => i.label.into(),
            Err(value) => value
                .get("label")
                .and_then(|v| v.as_str())
                .map(|s| SharedString::from(s.to_owned()))
                .unwrap_or_else(|| "Spawning agent".into()),
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
                .map_err(|e| SpawnAgentToolOutput::Error {
                    session_id: None,
                    error: e.to_string(),
                    session_info: None,
                })?;

            let SpawnAgentToolInput {
                label,
                message,
                session_id,
                model,
                interrupt,
            } = input;
            let subagent_label: SharedString = label.clone().into();
            let tool_use_id = event_stream.tool_use_id().clone();
            let tool_call_id = event_stream.tool_call_id().clone();

            enum LaunchOutcome {
                Spawned(Rc<dyn crate::SubagentHandle>, SubagentSessionInfo),
                Steered {
                    session_info: SubagentSessionInfo,
                    pending_steers: usize,
                },
            }

            let launch_outcome = cx.update(|cx| {
                let outcome = match (session_id, model) {
                    (Some(_), Some(_)) => Err(anyhow::anyhow!(
                        "model cannot be changed when resuming a subagent session"
                    )),
                    (Some(session_id), None) => self
                        .environment
                        .follow_up_subagent(session_id, message.clone(), interrupt, cx)
                        .map(|follow_up| match follow_up {
                            crate::SubagentFollowUp::Resumed(subagent) => {
                                subagent.mark_started(cx);
                                let session_info = SubagentSessionInfo {
                                    session_id: subagent.id(),
                                    message_start_index: subagent.num_entries(cx),
                                    message_end_index: None,
                                    model_name: None,
                                };
                                LaunchOutcome::Spawned(subagent, session_info)
                            }
                            crate::SubagentFollowUp::Steered {
                                session_info,
                                pending_steers,
                            } => LaunchOutcome::Steered {
                                session_info,
                                pending_steers,
                            },
                        }),
                    (None, model) => self
                        .environment
                        .create_subagent(label, model.map(AgentModelId::from), cx)
                        .map(|subagent| {
                            subagent.mark_started(cx);
                            let session_info = SubagentSessionInfo {
                                session_id: subagent.id(),
                                message_start_index: subagent.num_entries(cx),
                                message_end_index: None,
                                model_name: None,
                            };
                            LaunchOutcome::Spawned(subagent, session_info)
                        }),
                };
                outcome.map_err(|err| SpawnAgentToolOutput::Error {
                    session_id: None,
                    error: err.to_string(),
                    session_info: None,
                })
            })?;

            let (subagent, session_info) = match launch_outcome {
                LaunchOutcome::Steered {
                    session_info,
                    pending_steers,
                } => {
                    let steer_summary = if interrupt {
                        format!("Interrupted and steered subagent:\n\n{message}")
                    } else {
                        format!("Steered subagent at next boundary:\n\n{message}")
                    };
                    event_stream.update_fields(|update| {
                        update
                            .content(vec![steer_summary.into()])
                            .meta(acp_v2::Meta::from_iter([(
                                SUBAGENT_SESSION_INFO_META_KEY.into(),
                                serde_json::json!(&session_info),
                            )]))
                    });
                    return Ok(SpawnAgentToolOutput::Steered {
                        session_id: session_info.session_id.clone(),
                        steered: true,
                        interrupt,
                        pending_steers,
                        message,
                        session_info,
                    });
                }
                LaunchOutcome::Spawned(subagent, session_info) => {
                    event_stream.update_fields(|update| {
                        update.meta(acp_v2::Meta::from_iter([(
                            SUBAGENT_SESSION_INFO_META_KEY.into(),
                            serde_json::json!(&session_info),
                        )]))
                    });
                    (subagent, session_info)
                }
            };

            cx.spawn({
                let mut session_info = session_info.clone();
                async move |cx| {
                    let send_result = subagent.send(message, cx).await;

                    let status = match &send_result {
                        crate::SubagentSendResult::Completed(_) => "completed",
                        crate::SubagentSendResult::Canceled => "canceled",
                        crate::SubagentSendResult::Error(_) => "error",
                    };
                    telemetry::event!(
                        "Subagent Completed",
                        subagent_session = session_info.session_id.to_string(),
                        status,
                    );

                    cx.update(|cx| {
                        session_info.message_end_index =
                            Some(subagent.num_entries(cx).saturating_sub(1));
                        subagent.complete(
                            subagent_label,
                            tool_use_id,
                            tool_call_id,
                            session_info,
                            send_result,
                            cx,
                        );
                    });
                }
            })
            .detach();

            Ok(SpawnAgentToolOutput::Running {
                session_id: session_info.session_id.clone(),
                session_info,
            })
        })
    }

    fn status_for_output(output: &Self::Output) -> Option<acp_v2::ToolCallStatus> {
        match output {
            SpawnAgentToolOutput::Running { .. } => Some(acp_v2::ToolCallStatus::InProgress),
            SpawnAgentToolOutput::Success { .. } | SpawnAgentToolOutput::Steered { .. } => {
                Some(acp_v2::ToolCallStatus::Completed)
            }
            SpawnAgentToolOutput::Error { .. } => Some(acp_v2::ToolCallStatus::Failed),
            SpawnAgentToolOutput::Canceled { .. } => Some(acp_v2::ToolCallStatus::Cancelled),
        }
    }

    fn replay_status(output: &Self::Output) -> Option<acp_v2::ToolCallStatus> {
        match output {
            SpawnAgentToolOutput::Success { .. } | SpawnAgentToolOutput::Steered { .. } => {
                Some(acp_v2::ToolCallStatus::Completed)
            }
            SpawnAgentToolOutput::Error { .. } => Some(acp_v2::ToolCallStatus::Failed),
            SpawnAgentToolOutput::Running { .. } | SpawnAgentToolOutput::Canceled { .. } => {
                Some(acp_v2::ToolCallStatus::Cancelled)
            }
        }
    }

    fn replay(
        &self,
        _input: Self::Input,
        output: Self::Output,
        event_stream: ToolCallEventStream,
        _cx: &mut App,
    ) -> Result<()> {
        let (content, session_info) = match output {
            SpawnAgentToolOutput::Success {
                output,
                session_info,
                ..
            } => (Some(output.into()), Some(session_info)),
            SpawnAgentToolOutput::Steered {
                interrupt,
                message,
                session_info,
                ..
            } => {
                let steer_summary = if interrupt {
                    format!("Interrupted and steered subagent:\n\n{message}")
                } else {
                    format!("Steered subagent at next boundary:\n\n{message}")
                };
                (Some(steer_summary.into()), Some(session_info))
            }
            SpawnAgentToolOutput::Error {
                error,
                session_info,
                ..
            } => (Some(error.into()), session_info),
            SpawnAgentToolOutput::Running { session_info, .. }
            | SpawnAgentToolOutput::Canceled { session_info, .. } => (None, Some(session_info)),
        };

        let meta = session_info.map(|session_info| {
            acp_v2::Meta::from_iter([(
                SUBAGENT_SESSION_INFO_META_KEY.into(),
                serde_json::json!(&session_info),
            )])
        });
        event_stream.update_fields(|update| {
            let update = if let Some(content) = content {
                update.content(vec![content])
            } else {
                update
            };
            match meta {
                Some(meta) => update.meta(meta),
                None => update,
            }
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn session_ids_preserve_legacy_tool_json() -> Result<(), serde_json::Error> {
        let session_id = "  session/雪:\"quoted\"\\opaque  ";
        let input_json = json!({
            "label": "label",
            "message": "message",
            "session_id": session_id,
        });
        let input: SpawnAgentToolInput = serde_json::from_value(input_json.clone())?;
        assert_eq!(
            input.session_id.as_ref().map(|id| id.0.as_ref()),
            Some(session_id)
        );
        assert_eq!(serde_json::to_value(input)?, input_json);

        let output_json = json!({
            "session_id": session_id,
            "error": "failed after session creation",
            "session_info": null,
        });
        let output: SpawnAgentToolOutput = serde_json::from_value(output_json.clone())?;
        assert_eq!(serde_json::to_value(output)?, output_json);

        let running_json = json!({
            "session_id": session_id,
            "session_info": {
                "session_id": session_id,
                "message_start_index": 0,
            },
        });
        let running: SpawnAgentToolOutput = serde_json::from_value(running_json.clone())?;
        assert!(matches!(running, SpawnAgentToolOutput::Running { .. }));
        assert_eq!(serde_json::to_value(&running)?, running_json);

        let canceled_json = json!({
            "session_id": session_id,
            "canceled": true,
            "session_info": {
                "session_id": session_id,
                "message_start_index": 0,
                "message_end_index": 1,
            },
        });
        let canceled: SpawnAgentToolOutput = serde_json::from_value(canceled_json.clone())?;
        assert!(matches!(canceled, SpawnAgentToolOutput::Canceled { .. }));
        assert_eq!(serde_json::to_value(&canceled)?, canceled_json);
        Ok(())
    }

    #[test]
    fn deserializes_blank_session_id_as_absent() {
        for session_id in [json!(null), json!(""), json!("   ")] {
            let input: SpawnAgentToolInput = serde_json::from_value(json!({
                "label": "label",
                "message": "message",
                "session_id": session_id,
            }))
            .unwrap();

            assert!(input.session_id.is_none());
        }

        let input: SpawnAgentToolInput = serde_json::from_value(json!({
            "label": "label",
            "message": "message",
        }))
        .unwrap();
        assert!(input.session_id.is_none());

        let input: SpawnAgentToolInput = serde_json::from_value(json!({
            "label": "label",
            "message": "message",
            "session_id": "existing-session",
        }))
        .unwrap();
        assert_eq!(input.session_id.unwrap().to_string(), "existing-session");
    }
}
