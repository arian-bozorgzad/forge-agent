"""Plans are immutable; approval commits workspace changes and their audit together."""

from uuid import uuid4

from app.models import Message, Note, Preference, Run, Task, Timer, Workflow, now
from app.schemas import NoteInput, TaskPatch
from app.tools import TOOL_MODELS, WRITE_TOOLS, execute, serialize, summarize, validate_step
from app.trajectory import fingerprint, record, transaction, transition

TOOL_VERSION = "1"


def manifest():
    return {
        name: {
            "version": TOOL_VERSION,
            "schema": schema.model_json_schema(),
            "effect": "write" if name in WRITE_TOOLS else "read",
        }
        for name, schema in TOOL_MODELS.items()
    }


def resource_conditions(db, steps):
    result = []
    for step in steps:
        model = None
        if step["tool"] == "delete_note":
            model = Note
        elif step["tool"] == "complete_task":
            model = Task
        if model:
            item = db.get(model, step["arguments"]["id"])
            if item is None:
                raise ValueError("Referenced resource no longer exists")
            result.append(
                {
                    "tool": step["tool"],
                    "id": item.id,
                    "fingerprint": fingerprint(serialize(item)),
                    "snapshot": serialize(item),
                }
            )
    return result


def propose(run_id, steps):
    with transaction() as db:
        run = db.get(Run, run_id)
        if run.status != "running":
            raise ValueError("Run is no longer planning")
        validated = [validate_step(step) for step in steps]
        reasons = []
        if not 1 <= len(validated) <= 8:
            reasons.append("Plan must have 1 to 8 steps")
        if any(
            step["tool"] not in WRITE_TOOLS or step["tool"] == "propose_workflow"
            for step in validated
        ):
            reasons.append("Plans may contain only concrete approved actions")
        record(db, run, "policy_checked", {"passed": not reasons, "reasons": reasons})
        if reasons:
            run.result = "; ".join(reasons)
            transition(db, run, "failed", "policy_rejected")
            return serialize(run)
        plan = {
            "steps": validated,
            "tool_versions": {step["tool"]: TOOL_VERSION for step in validated},
            "preconditions": resource_conditions(db, validated),
        }
        run.plan, run.plan_hash = plan, fingerprint(plan)
        record(db, run, "plan_proposed", {"plan": plan, "plan_hash": run.plan_hash})
        transition(db, run, "waiting_confirmation", "validated_proposal")
        return serialize(run)


def execute_plan(run_id, plan_hash):
    result = _execute_plan(run_id, plan_hash)
    if "_error" in result:
        raise ValueError(result["_error"])
    return result


def _execute_plan(run_id, plan_hash):
    with transaction() as db:
        run = db.get(Run, run_id)
        if run is None:
            raise ValueError("Run not found")
        if not run.plan_hash or plan_hash != run.plan_hash or fingerprint(run.plan) != plan_hash:
            record(
                db,
                run,
                "approval_recorded",
                {"decision": "rejected", "plan_hash": plan_hash, "reason": "hash_mismatch"},
            )
            return {"_error": "Approval does not match the stored plan"}
        if run.status == "completed":
            return serialize(run)
        if run.status != "waiting_confirmation":
            record(
                db,
                run,
                "approval_recorded",
                {"decision": "rejected", "plan_hash": plan_hash, "reason": "not_awaiting_approval"},
            )
            return {"_error": "Run is not awaiting approval"}
        record(db, run, "approval_recorded", {"decision": "approved", "plan_hash": plan_hash})
        outcomes = []
        current_tool = None
        try:
            # A real outer BEGIN IMMEDIATE already exists; releasing this savepoint is not a commit.
            with db.begin_nested():
                if any(v != TOOL_VERSION for v in run.plan["tool_versions"].values()):
                    raise ValueError("Tool implementation changed; create a new proposal")
                actual = resource_conditions(db, run.plan["steps"])
                if actual != run.plan["preconditions"]:
                    raise ValueError("Plan is stale; create a fresh proposal")
                for step in run.plan["steps"]:
                    current_tool = step["tool"]
                    result = execute(db, step, approved=True)
                    outcomes.append({"tool": step["tool"], "result": result, "status": "applied"})
                db.flush()
        except Exception as error:
            # Keep attempted outcomes, but explicitly mark them rolled back, never completed.
            for outcome in outcomes:
                outcome["status"] = "rolled_back"
            run.result = "No workspace changes saved. Execution failed."
            record(
                db,
                run,
                "execution_finished",
                {
                    "committed": False,
                    "outcomes": outcomes,
                    "error": type(error).__name__,
                    "failed_tool": current_tool,
                    "reason": str(error)
                    if isinstance(error, ValueError)
                    else "Database/tool failure",
                },
            )
            transition(db, run, "failed", "execution_rolled_back")
        else:
            run.result = "\n".join(summarize(item["tool"], item["result"]) for item in outcomes)
            record(db, run, "execution_finished", {"committed": True, "outcomes": outcomes})
            transition(db, run, "completed", "execution_committed")
        if run.conversation_id:
            db.add(
                Message(
                    conversation_id=run.conversation_id,
                    run_id=run.id,
                    role="assistant",
                    content=run.result,
                    sources=[],
                )
            )
        return serialize(run)


def cancel_plan(run_id):
    with transaction() as db:
        run = db.get(Run, run_id)
        if run is None:
            raise ValueError("Run not found")
        if run.status == "waiting_confirmation":
            record(
                db, run, "approval_recorded", {"decision": "rejected", "plan_hash": run.plan_hash}
            )
            run.result = "Cancelled. No proposed changes applied."
            transition(db, run, "cancelled", "user_rejected")
            if run.conversation_id:
                db.add(
                    Message(
                        conversation_id=run.conversation_id,
                        run_id=run.id,
                        role="assistant",
                        content=run.result,
                        sources=[],
                    )
                )
        return serialize(run)


def manual_write(operation, arguments, item_id=None):
    """A submitted form explicitly approves its values. These are not new model tools."""
    with transaction() as db:
        request = {"operation": operation, "arguments": arguments, "id": item_id}
        run = Run(
            id=str(uuid4()),
            kind="manual",
            label=operation,
            status="running",
            plan=[],
            trajectory_version=1,
            next_event_sequence=1,
            request_fingerprint=fingerprint(request),
            provider="manual",
            model=None,
            prompt_version=None,
            limits={},
            tool_manifest={},
        )
        db.add(run)
        db.flush()
        record(db, run, "run_started", {"request": request})
        run.plan = {"operation": operation, "arguments": arguments, "id": item_id}
        run.plan_hash = fingerprint(run.plan)
        record(
            db, run, "policy_checked", {"passed": True, "reasons": [], "source": "explicit_form"}
        )
        record(db, run, "plan_proposed", {"plan": run.plan, "plan_hash": run.plan_hash})
        transition(db, run, "waiting_confirmation", "submitted_form")
        record(
            db,
            run,
            "approval_recorded",
            {"decision": "approved", "plan_hash": run.plan_hash, "source": "explicit_form"},
        )
        result = None
        try:
            with db.begin_nested():
                if operation in {"create_note", "add_task", "start_timer"}:
                    result = execute(db, {"tool": operation, "arguments": arguments}, approved=True)
                elif operation in {"create_workflow", "edit_workflow", "delete_workflow"}:
                    if operation == "create_workflow":
                        row = Workflow(**arguments)
                        db.add(row)
                    else:
                        row = db.get(Workflow, item_id)
                        if row is None:
                            raise ValueError("Workflow not found")
                        if operation == "delete_workflow":
                            db.delete(row)
                        else:
                            row.name, row.description, row.steps = (
                                arguments["name"],
                                arguments["description"],
                                arguments["steps"],
                            )
                    db.flush()
                    result = serialize(row)
                elif operation == "preferences":
                    row = db.get(Preference, "default")
                    row.display_name, row.instructions = (
                        arguments["display_name"],
                        arguments["instructions"],
                    )
                    result = serialize(row)
                elif operation == "cancel_timer":
                    row = db.get(Timer, item_id)
                    if row is None:
                        raise ValueError("Timer not found")
                    row.cancelled = True
                    result = serialize(row)
                else:
                    model = Note if "note" in operation else Task
                    row = db.get(model, item_id)
                    if row is None:
                        raise ValueError("Item not found")
                    if operation == "edit_note":
                        parsed = NoteInput.model_validate(arguments)
                        row.title, row.content, row.updated_at = parsed.title, parsed.content, now()
                    elif operation == "edit_task":
                        row.completed = TaskPatch.model_validate(arguments).completed
                    elif operation in {"delete_note", "delete_task"}:
                        db.delete(row)
                    else:
                        raise ValueError("Unknown manual operation")
                    db.flush()
                    result = serialize(row)
        except Exception as error:
            run.result = (
                str(error) if isinstance(error, ValueError) else "No workspace changes saved."
            )
            record(
                db, run, "execution_finished", {"committed": False, "error": type(error).__name__}
            )
            transition(db, run, "failed", "manual_write_failed")
        else:
            record(db, run, "execution_finished", {"committed": True, "result": result})
            transition(db, run, "completed", "manual_write_committed")
            run.result = operation
        return serialize(run), result
