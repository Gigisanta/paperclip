import { hasWorkspaceRestoreFailure } from "@paperclipai/shared";
import { randomUUID } from "node:crypto";
import { claimedAdapterType, conversationRecoveryActionPredicate, getConversationOwnershipBlocker } from "./conversation-continuation.js";
import { persistActivity } from "./activity-log.js";
import { appendHeartbeatRunEvent } from "./heartbeat-run-events.js";
import { logger } from "../middleware/logger.js";
import { and, eq, inArray, isNull, not, or, sql } from "drizzle-orm";
import {
  agentWakeupRequests,
  chatActions,
  environmentLeases,
  heartbeatRuns,
  issueRecoveryActions,
  issues,
  nativeRunFinalizations,
  type Db,
} from "@paperclipai/db";
import { conflict } from "../errors.js";
import { buildExecutionContinuation } from "./execution-continuation.js";
import {
  EXECUTION_RECONCILIATION_CAUSES,
  type ExecutionReconciliation,
} from "@paperclipai/shared";
import { parseIssueExecutionState } from "./issue-execution-policy.js";
import { isSupersededConversationRun } from "./agent-conversations.js";
import { hasRequiredWorkspaceRecovery } from "./workspace-restore-recovery-state.js";
import { hasRemoteTerminationReceipt } from "./remote-execution-termination.js";
import { issueRecoveryActionService } from "./issue-recovery-actions.js";

type RecoveryAction = typeof issueRecoveryActions.$inferSelect;
type ReconciliationOwner = { kind: "chat_failed_run_retry"; actionId: string };

function reconciliationMatches(recorded: Record<string, unknown>, decision: ExecutionReconciliation) {
  return ["runId", "providerStopped", "actionOutcome", "outcomeEvidence", "workspaceRepairEvidence"]
    .every(key => recorded[key] === decision[key as keyof ExecutionReconciliation]);
}

/**
 * The caller holds the issue row lock shared by resolution and queue admission.
 * A legacy source can have old resolved no-replay rows as well as a newer intent.
 * Fold those records into that intent, retaining their evidence, not their holds.
 */
async function consolidateLegacyReconciliation(
  db: Db,
  task: typeof issues.$inferSelect,
  selected: RecoveryAction,
  decision: ExecutionReconciliation,
  actorId: string,
  deliveryOwner?: ReconciliationOwner,
): Promise<RecoveryAction> {
  const fingerprint = `legacy-execution:${decision.runId}`;
  const rows = await db.select().from(issueRecoveryActions).where(and(
    // Include malformed cross-company/source records so they fail closed rather
    // than silently leaving an unexamined sibling behind.
    or(eq(issueRecoveryActions.companyId, task.companyId), eq(issueRecoveryActions.sourceIssueId, task.id)),
    or(eq(issueRecoveryActions.fingerprint, fingerprint),
      and(eq(issueRecoveryActions.cause, "legacy_execution_requires_reconciliation"),
        or(sql`${issueRecoveryActions.evidence}->>'runId' = ${decision.runId}`,
          sql`${issueRecoveryActions.evidence}->>'sourceRunId' = ${decision.runId}`,
          sql`${issueRecoveryActions.evidence}->'executionReconciliation'->>'runId' = ${decision.runId}`))),
  )).orderBy(issueRecoveryActions.createdAt, issueRecoveryActions.id).for("update");
  if (!rows.some(row => row.id === selected.id)) throw conflict("The recovery source changed.");
  const owners = new Set<string>();
  const canonicalIds = new Set<string>();
  for (const row of rows) {
    const recorded = row.evidence.executionReconciliation as Record<string, unknown> | undefined;
    const automatic = row.evidence.automaticRecovery as Record<string, unknown> | undefined;
    const consolidation = row.evidence.executionConsolidation as Record<string, unknown> | undefined;
    const superseded = row.evidence.supersededExecutionRecovery as Record<string, unknown> | undefined;
    if (row.companyId !== task.companyId || row.sourceIssueId !== task.id ||
        row.kind !== "active_run_watchdog" || row.cause !== "legacy_execution_requires_reconciliation" ||
        row.fingerprint !== fingerprint || row.returnOwnerAgentId !== task.assigneeAgentId ||
        !row.returnOwnerAgentId || hasRequiredWorkspaceRecovery(row.evidence) ||
        ![row.evidence.runId, row.evidence.sourceRunId, automatic?.runId]
          .every(value => value === undefined || value === decision.runId) ||
        (row.evidence.runId ?? row.evidence.sourceRunId) !== decision.runId ||
        (recorded && !reconciliationMatches(recorded, decision)) ||
        (consolidation && consolidation.sourceRunId !== decision.runId) ||
        (superseded && superseded.sourceRunId !== decision.runId)) {
      throw conflict("Conflicting recovery source, owner, or reconciliation evidence; no actions were consolidated.");
    }
    if (recorded && !superseded) {
      if (!["pending", "delivered", "delegated"].includes(String(row.evidence.continuationDelivery))) {
        throw conflict("This reconciliation no longer authorizes delivery.");
      }
      const owner = row.evidence.continuationDeliveryOwner as Record<string, unknown> | null | undefined;
      if (owner && (owner.kind !== "chat_failed_run_retry" || typeof owner.actionId !== "string")) {
        throw conflict("Invalid recovery continuation owner.");
      }
      owners.add(owner ? `chat_failed_run_retry:${owner.actionId}` : "issue");
    }
    const canonicalId = consolidation?.canonicalActionId ?? superseded?.canonicalActionId;
    if (canonicalId !== undefined) {
      if (typeof canonicalId !== "string") throw conflict("Invalid recovery consolidation evidence.");
      canonicalIds.add(canonicalId);
    }
  }
  if (deliveryOwner) owners.add(`chat_failed_run_retry:${deliveryOwner.actionId}`);
  if (owners.size > 1 || canonicalIds.size > 1) throw conflict("Conflicting recovery continuation owners.");
  const canonicalId = [...canonicalIds][0];
  const canonical = canonicalId ? rows.find(row => row.id === canonicalId)
    : rows.find(row => row.evidence.executionReconciliation && !row.evidence.supersededExecutionRecovery) ?? selected;
  if (!canonical || canonical.evidence.supersededExecutionRecovery) throw conflict("The canonical recovery action is unavailable.");
  const successors = await db.select().from(heartbeatRuns).where(and(
    or(eq(heartbeatRuns.retryOfRunId, decision.runId), and(
      sql`${heartbeatRuns.contextSnapshot}->>'previousRunId' = ${decision.runId}`,
      sql`${heartbeatRuns.contextSnapshot}->>'source' = 'execution.reconciled'`)),
  )).for("update");
  if (successors.length > 1 || successors.some(run => run.id === decision.runId ||
      run.companyId !== task.companyId || run.agentId !== task.assigneeAgentId ||
      (run.nativeIssueId ?? run.contextSnapshot?.issueId) !== task.id ||
      (run.nativeIssueId != null && run.nativeIssueId !== task.id) ||
      (run.contextSnapshot?.issueId !== undefined && run.contextSnapshot.issueId !== task.id) ||
      (run.retryOfRunId !== null && run.retryOfRunId !== decision.runId))) {
    throw conflict("Conflicting successors already exist for this recovery source; inspect them before continuing.");
  }
  const successor = successors[0];
  if ((canonical.evidence.continuationRunId !== undefined &&
      canonical.evidence.continuationRunId !== successor?.id) ||
      (canonical.evidence.continuationDelivery === "delivered" && !successor)) {
    throw conflict("The recorded recovery successor does not match the admitted source-run continuation.");
  }
  if (!successor && ["done", "cancelled"].includes(task.status)) throw conflict("The recovery task is no longer actionable.");
  const source = await validateExecutionReconciliation({ db, companyId: task.companyId, issueId: task.id,
    agentId: task.assigneeAgentId, sourceRunId: decision.runId, decision });
  if ((source.nativeIssueId != null && source.nativeIssueId !== task.id) ||
      (source.contextSnapshot?.issueId !== undefined && source.contextSnapshot.issueId !== task.id)) {
    throw conflict("The recovery source has conflicting task scope.");
  }
  const key = `execution-reconciliation:${task.companyId}:${task.id}:${decision.runId}`;
  const actionIds = rows.map(row => row.id);
  const receiptKeys = [key, ...actionIds.map(id => `execution-reconciliation:${id}`)];
  const wakes = await db.select().from(agentWakeupRequests).where(or(
    inArray(agentWakeupRequests.idempotencyKey, receiptKeys),
    sql`${agentWakeupRequests.payload}->'executionWait'->>'requestedIdempotencyKey' in (${sql.join(receiptKeys.map(value => sql`${value}`), sql`, `)})`,
  )).orderBy(agentWakeupRequests.requestedAt, agentWakeupRequests.id).for("update");
  for (const wake of wakes) {
    const context = wake.payload?._paperclipWakeContext as Record<string, unknown> | undefined;
    if (wake.companyId !== task.companyId || wake.agentId !== task.assigneeAgentId || wake.payload?.issueId !== task.id ||
        wake.source !== "automation" || wake.triggerDetail !== "system" ||
        wake.requestedByActorType !== "system" || wake.requestedByActorId !== "execution-recovery" ||
        (wake.payload?.recoveryActionId !== undefined && !actionIds.includes(String(wake.payload.recoveryActionId))) ||
        (context?.previousRunId !== undefined && context.previousRunId !== decision.runId) ||
        (context?.issueId !== undefined && context.issueId !== task.id) ||
        (context?.recoveryActionId !== undefined && !actionIds.includes(String(context.recoveryActionId))) ||
        (wake.runId && wake.runId !== successor?.id)) {
      throw conflict("Conflicting recovery wake receipt; no new continuation was authorized.");
    }
  }
  const wake = wakes.find(row => row.runId) ?? wakes.find(row => row.idempotencyKey === key)
    ?? wakes.find(row => row.status !== "skipped") ?? wakes[0];
  const now = new Date();
  if (successor && successor.retryOfRunId === null) {
    await db.update(heartbeatRuns).set({ retryOfRunId: decision.runId, updatedAt: now })
      .where(eq(heartbeatRuns.id, successor.id));
  }
  const prior = canonical.evidence.executionConsolidation as Record<string, unknown> | undefined;
  const consolidation = {
    version: 1, sourceRunId: decision.runId, canonicalActionId: canonical.id, actionIds,
    idempotencyKey: key, wakeRequestId: wake?.id ?? prior?.wakeRequestId ?? null,
    actorId: prior?.actorId ?? actorId, recordedAt: prior?.recordedAt ?? now.toISOString(),
  };
  for (const receipt of wakes) {
    const receiptEvidence = receipt.payload?.executionConsolidation as Record<string, unknown> | undefined;
    await db.update(agentWakeupRequests).set({
      ...(receipt.id === wake?.id ? { idempotencyKey: key,
        ...(successor ? { runId: successor.id, status: receipt.runId ? receipt.status : "coalesced" } : {}) }
        : { status: "skipped", reason: "execution_reconciliation_superseded", finishedAt: now }),
      payload: { ...receipt.payload, recoveryActionId: canonical.id,
        executionConsolidation: { ...receiptEvidence, canonicalActionId: canonical.id, sourceRunId: decision.runId,
          canonicalWakeRequestId: wake!.id, originalIdempotencyKey: receiptEvidence?.originalIdempotencyKey ?? receipt.idempotencyKey,
          originalStatus: receiptEvidence?.originalStatus ?? receipt.status,
          originalReason: receiptEvidence?.originalReason ?? receipt.reason,
          originalExecutionWait: receiptEvidence?.originalExecutionWait ?? receipt.payload?.executionWait } },
      updatedAt: now,
    }).where(eq(agentWakeupRequests.id, receipt.id));
  }
  for (const sibling of rows) {
    if (sibling.id === canonical.id) continue;
    if (["active", "escalated"].includes(sibling.status)) {
      await issueRecoveryActionService(db).resolveActiveForIssue({ companyId: task.companyId, sourceIssueId: task.id,
        actionId: sibling.id, status: "resolved", outcome: "cancelled",
        resolutionNote: `Consolidated into recovery action ${canonical.id} for source run ${decision.runId}.` });
    }
    await db.update(issueRecoveryActions).set({
      evidence: { ...sibling.evidence, automaticRecovery: undefined, continuationDelivery: "superseded",
        supersededExecutionRecovery: sibling.evidence.supersededExecutionRecovery ?? {
          canonicalActionId: canonical.id, sourceRunId: decision.runId, actorId, recordedAt: now.toISOString(),
          automaticRecovery: sibling.evidence.automaticRecovery,
          continuationDelivery: sibling.evidence.continuationDelivery,
        } },
      wakePolicy: null, monitorPolicy: null, updatedAt: now,
    }).where(eq(issueRecoveryActions.id, sibling.id));
  }
  return { ...canonical, evidence: { ...canonical.evidence, executionConsolidation: consolidation,
    ...(canonical.evidence.automaticRecovery ? { reconciledAutomaticRecovery: canonical.evidence.automaticRecovery } : {}),
    continuationDelivery: successor ? "delivered" : canonical.evidence.continuationDelivery ?? "pending",
    ...(successor ? { continuationRunId: successor.id } : {}) } };
}

/** An operator records observed outcomes; this is not permission to blindly retry. */
export async function validateExecutionReconciliation(input: {
  db: Db;
  companyId: string;
  issueId: string;
  agentId: string | null;
  sourceRunId: unknown;
  decision: ExecutionReconciliation | undefined;
  /** Board-only source repair: never authorize a continuation of this run. */
  workspaceRepairOnly?: boolean;
}) {
  const { db, companyId, issueId, agentId, decision } = input;
  if (!decision || decision.runId !== input.sourceRunId || (!agentId && !input.workspaceRepairOnly)) {
    throw conflict(
      "Reconcile the recorded execution and its action outcomes before continuing this task.",
    );
  }
  const [run] = await db
    .select()
    .from(heartbeatRuns)
    .where(
      and(
        eq(heartbeatRuns.companyId, companyId),
        eq(heartbeatRuns.id, decision.runId),
      ),
    );
  const [task] = await db
    .select()
    .from(issues)
    .where(and(eq(issues.companyId, companyId), eq(issues.id, issueId)));
  const review =
    task?.status === "in_review"
      ? parseIssueExecutionState(task.executionState)
      : null;
  const isCurrentReviewer =
    review?.status === "pending" &&
    review.currentParticipant?.type === "agent" &&
    review.currentParticipant.agentId === run?.agentId;
  const retainedWorkspace = hasRequiredWorkspaceRecovery(run?.resultJson);
  const repairOnly = input.workspaceRepairOnly === true && retainedWorkspace;
  if (
    (input.workspaceRepairOnly && !retainedWorkspace) ||
    !run ||
    !task ||
    (!repairOnly && (task.assigneeAgentId !== agentId || (run.agentId !== agentId && !isCurrentReviewer))) ||
    (run.nativeIssueId ?? run.contextSnapshot?.issueId) !== issueId ||
    !["failed", "interrupted", "timed_out", "cancelled"].includes(run.status)
  ) {
    throw conflict(
      "The recovery source or task owner changed. Inspect the current execution before continuing.",
    );
  }
  if (hasWorkspaceRestoreFailure(run.resultJson) &&
      (!decision.workspaceRepairEvidence || decision.workspaceRepairEvidence.trim().length < 20)) {
    throw conflict("Verify safe workspace staging or repair and record workspaceRepairEvidence before continuing this run.");
  }
  let checkLocalProcesses = true;
  if (retainedWorkspace) {
    const leases = await db.select().from(environmentLeases).where(and(
      eq(environmentLeases.companyId, companyId), eq(environmentLeases.heartbeatRunId, run.id),
    ));
    const recovery = run.resultJson!.workspaceRestoreRecovery as { leaseIds?: unknown };
    const retainedIds = recovery.leaseIds;
    // The terminal transaction records the exact source allocations. A release
    // timestamp is written before stop dispatch, so only matching provider
    // acknowledgements can establish that repair will not race active writes.
    if (!Array.isArray(retainedIds) || !retainedIds.length ||
        !retainedIds.every(id => typeof id === "string" && leases.some(lease =>
          lease.id === id && hasRemoteTerminationReceipt(lease))) ||
        leases.some(lease => lease.provider === "local"
          ? !lease.releasedAt || lease.status === "pending_cleanup" || lease.cleanupStatus === "failed"
          : !hasRemoteTerminationReceipt(lease))) {
      throw conflict("The retained workspace environment has not confirmed that it stopped. Wait for cleanup before recording workspace repair.");
    }
    checkLocalProcesses = leases.some(lease => lease.provider === "local");
  }
  // A retained remote source's PIDs belong to its sandbox, not this server.
  // Its exact lease receipts above replace host-local process probes.
  for (const pid of checkLocalProcesses ? [
    run.processPid,
    run.processGroupId ? -run.processGroupId : null,
  ] : []) {
    if (!pid) continue;
    try {
      process.kill(pid, 0);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ESRCH") continue;
      throw conflict(
        "The previous provider's process ownership cannot be verified.",
      );
    }
    throw conflict(
      "The previous provider is still running. Stop it before continuing.",
    );
  }
  const [coordinator] = await db
    .select()
    .from(nativeRunFinalizations)
    .where(
      and(
        eq(nativeRunFinalizations.companyId, companyId),
        eq(nativeRunFinalizations.runId, run.id),
      ),
    );
  const successorId = coordinator?.failureDetail?.successorRunId;
  const [adopted] = run.runtimeMode === "legacy" && typeof successorId === "string"
    ? await db.select({ id: heartbeatRuns.id }).from(heartbeatRuns).where(and(
        eq(heartbeatRuns.companyId, companyId), eq(heartbeatRuns.id, successorId),
        eq(heartbeatRuns.agentId, agentId!), eq(heartbeatRuns.retryOfRunId, run.id),
        sql`coalesce(${heartbeatRuns.nativeIssueId}::text, ${heartbeatRuns.contextSnapshot}->>'issueId') = ${issueId}`,
      )) : [];
  if (coordinator?.leaseOwner || (successorId && !adopted))
    throw conflict(
      "This execution still has a coordinator or a linked continuation. Inspect that run first.",
    );
  const leases = await db
    .select({ id: environmentLeases.id })
    .from(environmentLeases)
    .where(
      and(
        eq(environmentLeases.companyId, companyId),
        eq(environmentLeases.heartbeatRunId, run.id),
        isNull(environmentLeases.releasedAt),
      ),
    )
    .limit(1);
  if (leases.length)
    throw conflict(
      "The previous execution environment has not finished releasing its authority.",
    );
  if (!repairOnly) await buildExecutionContinuation({
    db,
    companyId,
    issueId,
    agentId: agentId!,
    context: { previousRunId: run.id },
    summary: null,
    exposeLowTrustRaw: false,
  });
  return run;
}

/** Durable delivery marker lives on the existing source-scoped recovery action. */
export async function markExecutionReconciliation(
  db: Db,
  action: Pick<
    typeof issueRecoveryActions.$inferSelect,
    "companyId" | "id" | "evidence" | "sourceIssueId"
  >,
  decision: ExecutionReconciliation,
  actorId: string,
  deliveryOwner?: { kind: "chat_failed_run_retry"; actionId: string },
  options?: { workspaceRepairOnly?: boolean },
) {
  return db.transaction(async tx => {
    const queryDb = tx as unknown as Db;
    const [task] = await tx.select().from(issues).where(and(
      eq(issues.companyId, action.companyId), eq(issues.id, action.sourceIssueId),
    )).for("update");
    const [stored] = await tx.select().from(issueRecoveryActions).where(and(
      eq(issueRecoveryActions.companyId, action.companyId), eq(issueRecoveryActions.sourceIssueId, action.sourceIssueId),
      eq(issueRecoveryActions.id, action.id),
    )).for("update");
    if (!task || !stored) throw conflict("The recovery source changed.");
    const legacy = !options?.workspaceRepairOnly && stored.cause === "legacy_execution_requires_reconciliation";
    const target = legacy
      ? await consolidateLegacyReconciliation(queryDb, task, stored, decision, actorId, deliveryOwner)
      : stored;
    const owner = deliveryOwner ?? (legacy ? target.evidence.continuationDeliveryOwner as ReconciliationOwner | undefined : undefined);
    if (options?.workspaceRepairOnly && (owner || !hasRequiredWorkspaceRecovery(target.evidence))) {
      throw conflict("Workspace repair must target the retained source without a continuation owner.");
    }
    if (owner) {
      const [retry] = await tx.select().from(chatActions).where(and(
        eq(chatActions.companyId, target.companyId), eq(chatActions.id, owner.actionId),
      ));
      if (owner.kind !== "chat_failed_run_retry" || !retry || retry.kind !== "failed_run_retry" ||
          !["issued", "processing", "processed"].includes(retry.status) || retry.payload.version !== 1 ||
          retry.payload.failedRunId !== decision.runId || retry.payload.issueId !== target.sourceIssueId) {
        throw conflict("The authorized chat retry owner is no longer valid.");
      }
    }
    await tx.update(nativeRunFinalizations).set({
      failureDetail: sql`coalesce(${nativeRunFinalizations.failureDetail}, '{}'::jsonb) || ${JSON.stringify({ replacementDenied: "operator_reconciled" })}::jsonb`,
    }).where(and(
      eq(nativeRunFinalizations.companyId, target.companyId), eq(nativeRunFinalizations.runId, decision.runId),
    ));
    const [recorded] = await tx.update(issueRecoveryActions).set({
      evidence: {
        ...target.evidence,
        automaticRecovery: undefined,
        executionReconciliation: legacy && target.evidence.executionReconciliation
          ? target.evidence.executionReconciliation : { ...decision, actorId, recordedAt: new Date().toISOString() },
        continuationDelivery: options?.workspaceRepairOnly ? "not_requested" : owner ? "delegated"
          : legacy ? target.evidence.continuationDelivery : "pending",
        ...(owner ? { continuationDeliveryOwner: owner } : {}),
      },
    }).where(and(
      eq(issueRecoveryActions.companyId, target.companyId), eq(issueRecoveryActions.id, target.id),
    )).returning();
    return recorded!;
  });
}

export async function deliverReconciledExecutions(
  db: Db,
  wake: ReturnType<typeof import("./heartbeat.js").heartbeatService>["wakeup"],
) {
  const pending = await db
    .select()
    .from(issueRecoveryActions)
    .where(
      and(
        eq(issueRecoveryActions.status, "resolved"),
        sql`${issueRecoveryActions.evidence}->>'continuationDelivery' = 'pending'`,
      ),
    )
    .limit(25);
  for (const action of pending) {
    try {
      const decision = action.evidence.executionReconciliation as
        ExecutionReconciliation | undefined;
      if (!decision || !action.returnOwnerAgentId) continue;
      const pendingDecision = and(
        eq(issueRecoveryActions.companyId, action.companyId),
        eq(issueRecoveryActions.id, action.id),
        eq(issueRecoveryActions.status, "resolved"),
        sql`${issueRecoveryActions.evidence}->>'continuationDelivery' = 'pending'`,
        sql`${issueRecoveryActions.evidence}->'executionReconciliation' = ${JSON.stringify(decision)}::jsonb`,
      );
      const [task] = await db
        .select()
        .from(issues)
        .where(
          and(
            eq(issues.companyId, action.companyId),
            eq(issues.id, action.sourceIssueId),
          ),
        );
      if (
        !task ||
        task.assigneeAgentId !== action.returnOwnerAgentId ||
        ["done", "cancelled"].includes(task.status)
      ) {
        await db
          .update(issueRecoveryActions)
          .set({
            evidence: sql`${issueRecoveryActions.evidence} || '{"continuationDelivery":"invalidated"}'::jsonb`,
          })
          .where(pendingDecision);
        continue;
      }
      const run = await wake(action.returnOwnerAgentId, {
        source: "automation",
        triggerDetail: "system",
        reason: "issue_recovery_action_restored",
        idempotencyKey: action.cause === "legacy_execution_requires_reconciliation"
          ? `execution-reconciliation:${action.companyId}:${task.id}:${decision.runId}`
          : `execution-reconciliation:${action.id}`,
        payload: { issueId: task.id, recoveryActionId: action.id },
        requestedByActorType: "system",
        requestedByActorId: "execution-recovery",
        contextSnapshot: {
          issueId: task.id,
          taskId: task.id,
          recoveryActionId: action.id,
          previousRunId: decision.runId,
          retryOfRunId: decision.runId,
          forceFreshSession: true,
          wakeReason: "issue_recovery_action_restored",
          source: "execution.reconciled",
        },
      });
      if (run)
        await db.transaction(async (tx) => {
          await tx
            .update(heartbeatRuns)
            .set({ retryOfRunId: decision.runId })
            .where(
              and(
                eq(heartbeatRuns.companyId, action.companyId),
                eq(heartbeatRuns.id, run.id),
                eq(heartbeatRuns.agentId, action.returnOwnerAgentId!),
                sql`${heartbeatRuns.contextSnapshot}->>'recoveryActionId' = ${action.id}`,
                sql`${heartbeatRuns.contextSnapshot}->>'previousRunId' = ${decision.runId}`,
              ),
            );
          await tx
            .update(issueRecoveryActions)
            .set({
              evidence: sql`${issueRecoveryActions.evidence} || ${JSON.stringify(
                {
                  continuationDelivery: "delivered",
                  continuationRunId: run.id,
                },
              )}::jsonb`,
            })
            .where(pendingDecision);
        });
    } catch {
      logger.warn(
        { recoveryActionId: action.id },
        "Reconciled execution continuation remains pending for retry",
      );
    }
  }
}

/**
 * Failed execution is a system responsibility, not a user questionnaire. After
 * automatic recovery is ruled out, preserve evidence and stop without replay.
 * This is NOT evidence that an external action succeeded or never happened.
 * The resolved record retains a dispatch hold until actual evidence clears it.
 */
export async function settleUnrecoverableExecutions(
  db: Db,
  now = new Date(),
  options: { failpoint?: (phase: "persisted") => void } = {},
) {
  // Fold obsolete conversation holds without waking historical work on upgrade.
  // Keep their evidence and record the policy change in the task's activity log.
  const obsoleteConversationHold = and(
    conversationRecoveryActionPredicate(),
    or(
      inArray(issueRecoveryActions.status, ["active", "escalated"]),
      sql`${issueRecoveryActions.evidence}->'automaticRecovery'->>'replay' = 'blocked'`,
    ),
  );
  await db.transaction(async tx => {
    const foldable = await tx.select().from(issueRecoveryActions).where(obsoleteConversationHold)
      .limit(25).for("update", { skipLocked: true });
    for (const candidate of foldable) {
      if (await getConversationOwnershipBlocker(tx as unknown as Db, candidate.companyId, candidate.sourceIssueId)) continue;
      const [action] = await tx.update(issueRecoveryActions).set({
        status: "resolved",
        outcome: "cancelled",
        resolvedAt: now,
        updatedAt: now,
        nextAction: "Automatic attempts stopped. Send a new message to continue the conversation.",
        resolutionNote: "Conversation continuation does not replay prior tool calls.",
        wakePolicy: null,
        monitorPolicy: null,
        evidence: sql`case when ${issueRecoveryActions.evidence} ? 'automaticRecovery'
          then jsonb_set(${issueRecoveryActions.evidence}, '{automaticRecovery,replay}', '"conversation_continuation"'::jsonb)
          else ${issueRecoveryActions.evidence} end`,
      }).where(and(obsoleteConversationHold, eq(issueRecoveryActions.id, candidate.id))).returning();
      if (!action) continue;
      await persistActivity(tx as unknown as Db, {
        companyId: action.companyId,
        actorType: "system",
        actorId: "execution-recovery",
        action: "issue.execution_recovery_settled",
        entityType: "issue",
        entityId: action.sourceIssueId,
        details: { recoveryActionId: action.id, outcome: "cancelled", continuation: "conversation" },
      });
    }
  });
  // Filter eligibility before applying the batch limit. A queue of sessions
  // awaiting replacement must not starve settled incidents behind it.
  const candidates = await db
    .select({ action: issueRecoveryActions })
    .from(issueRecoveryActions)
    .innerJoin(
      heartbeatRuns,
      and(
        eq(heartbeatRuns.companyId, issueRecoveryActions.companyId),
        sql`${heartbeatRuns.id}::text = ${issueRecoveryActions.evidence}->>'runId'`,
        sql`coalesce(${heartbeatRuns.nativeIssueId}::text, ${heartbeatRuns.contextSnapshot}->>'issueId') = ${issueRecoveryActions.sourceIssueId}::text`,
      ),
    )
    .leftJoin(
      nativeRunFinalizations,
      and(
        eq(nativeRunFinalizations.companyId, heartbeatRuns.companyId),
        eq(nativeRunFinalizations.runId, heartbeatRuns.id),
      ),
    )
    .where(
      and(
        not(conversationRecoveryActionPredicate()!),
        inArray(issueRecoveryActions.status, ["active", "escalated"]),
        eq(issueRecoveryActions.kind, "active_run_watchdog"),
        inArray(issueRecoveryActions.cause, [
          ...EXECUTION_RECONCILIATION_CAUSES,
        ]),
        inArray(heartbeatRuns.status, [
          "failed",
          "timed_out",
          "interrupted",
          "cancelled",
        ]),
        isNull(nativeRunFinalizations.leaseOwner),
        isNull(nativeRunFinalizations.resultId),
        or(
          isNull(nativeRunFinalizations.runId),
          eq(nativeRunFinalizations.phase, "terminal_failure"),
        ),
        sql`coalesce(${nativeRunFinalizations.failureDetail}->>'successorRunId', '') = ''`,
        sql`(${heartbeatRuns.runtimeMode} <> 'native' or coalesce(${nativeRunFinalizations.failureCode}, '') <> 'native_provider_terminal_failed'
        or coalesce(${nativeRunFinalizations.failureDetail}->>'replacementDenied', '') <> '')`,
      ),
    )
    .limit(25);
  for (const { action: candidate } of candidates) {
    const runId = candidate.evidence.runId;
    if (typeof runId !== "string") continue;
    try {
      await db.transaction(async (tx) => {
        await tx.execute(
          sql`select set_config('statement_timeout', '15000', true), set_config('lock_timeout', '1000', true)`,
        );
        // Same issue -> coordinator -> run ordering as replacement/finalization.
        const [task] = await tx
          .select()
          .from(issues)
          .where(
            and(
              eq(issues.companyId, candidate.companyId),
              eq(issues.id, candidate.sourceIssueId),
            ),
          )
          .for("update");
        const [coordinator] = await tx
          .select()
          .from(nativeRunFinalizations)
          .where(
            and(
              eq(nativeRunFinalizations.companyId, candidate.companyId),
              eq(nativeRunFinalizations.runId, runId),
            ),
          )
          .for("update");
        const [run] = await tx
          .select()
          .from(heartbeatRuns)
          .where(
            and(
              eq(heartbeatRuns.companyId, candidate.companyId),
              eq(heartbeatRuns.id, runId),
            ),
          )
          .for("update");
        const [action] = await tx
          .select()
          .from(issueRecoveryActions)
          .where(eq(issueRecoveryActions.id, candidate.id))
          .for("update");
        if (
          !task ||
          !run ||
          !action ||
          action.evidence.runId !== runId ||
          !EXECUTION_RECONCILIATION_CAUSES.includes(
            action.cause as (typeof EXECUTION_RECONCILIATION_CAUSES)[number],
          ) ||
          !["active", "escalated"].includes(action.status) ||
          (run.nativeIssueId ?? run.contextSnapshot?.issueId) !== task.id ||
          !["failed", "timed_out", "interrupted", "cancelled"].includes(
            run.status,
          )
        )
          return;
        // Give durable native recovery its chance; never preempt a resume,
        // replacement, result finalizer, or still-owned execution.
        if (
          coordinator?.leaseOwner ||
          coordinator?.resultId ||
          coordinator?.failureDetail?.successorRunId ||
          (coordinator && coordinator.phase !== "terminal_failure") ||
          (run.runtimeMode === "native" &&
            coordinator?.failureCode === "native_provider_terminal_failed" &&
            !coordinator.failureDetail?.replacementDenied)
        )
          return;
        const current =
          !isSupersededConversationRun(task, run) &&
          action.returnOwnerAgentId !== null &&
          task.assigneeAgentId === action.returnOwnerAgentId &&
          !["done", "cancelled"].includes(task.status) &&
          (!task.executionRunId || task.executionRunId === run.id) &&
          (!task.checkoutRunId || task.checkoutRunId === run.id);
        const note = current
          ? hasWorkspaceRestoreFailure(run.resultJson)
            ? "Workspace repair required. Verify safe staging or repair before continuing. Saved work and approval decisions remain in force."
            : "Automatic recovery stopped. Recorded work is preserved; actions with unverified outcomes will not be repeated."
          : "Recovery closed because the task's owner, execution, or status changed. No work was replayed.";
        let nativeFailureBlock = action.evidence.nativeFailureBlock;
        let nativeBootstrapFailureBlock = action.evidence.nativeBootstrapFailureBlock;
        if (current) {
          const [projected] = await tx
            .update(issues)
            .set({
              status: "blocked",
              executionRunId: null,
              checkoutRunId: null,
              updatedAt: now,
            })
            .where(eq(issues.id, task.id)).returning();
          // Only a transition owned by this failure grants a recovery receipt.
          // An already-blocked task may have a separate human/dependency hold.
          if (task.status !== "blocked" && run.runtimeMode === "native") {
            nativeFailureBlock = { runId: run.id, statusVersion: projected!.statusVersion };
          }
          if (task.status !== "blocked" && run.runtimeMode === "legacy" && !run.runtimeModeResolvedAt &&
              run.errorCode === "server_shutdown_interrupted" && claimedAdapterType(run) === "paperclip_runner") {
            nativeBootstrapFailureBlock = { runId: run.id, statusVersion: projected!.statusVersion, previousStatus: task.status };
          }
        }
        await tx
          .update(issueRecoveryActions)
          .set({
            status: "resolved",
            outcome: current ? "blocked" : "cancelled",
            resolvedAt: now,
            updatedAt: now,
            nextAction: note,
            resolutionNote: note,
            wakePolicy: null,
            monitorPolicy: null,
            evidence: {
              ...action.evidence,
              ...(nativeFailureBlock ? { nativeFailureBlock } : {}),
              ...(nativeBootstrapFailureBlock ? { nativeBootstrapFailureBlock } : {}),
              automaticRecovery: {
                policy: "preserve_without_replay_v1",
                runId: run.id,
                replay: "blocked",
                actionOutcome: "unknown",
                recordedAt: now.toISOString(),
              },
            },
          })
          .where(eq(issueRecoveryActions.id, action.id));
        await persistActivity(tx as unknown as Db, {
          companyId: run.companyId,
          actorType: "system",
          actorId: "execution-recovery",
          action: "issue.execution_recovery_settled",
          entityType: "issue",
          entityId: task.id,
          runId: run.id,
          details: {
            recoveryActionId: action.id,
            outcome: current ? "blocked" : "cancelled",
            replay: "not_authorized",
          },
        });
        await tx
          .update(heartbeatRuns)
          .set({ executionStatusDeliveryId: randomUUID() })
          .where(eq(heartbeatRuns.id, run.id));
        await appendHeartbeatRunEvent(tx as unknown as Db, {
          companyId: run.companyId,
          agentId: run.agentId,
          runId: run.id,
          eventType: "lifecycle",
          stream: "system",
          level: "warn",
          message: note,
          payload: {
            recoveryActionId: action.id,
            cause: action.cause,
            automaticRecovery: "preserve_without_replay_v1",
            replay: "blocked",
          },
        });
        options.failpoint?.("persisted");
      });
    } catch (err) {
      if (options.failpoint) throw err;
      logger.warn(
        { err, recoveryActionId: candidate.id },
        "Automatic recovery disposition remains pending",
      );
    }
  }
}
