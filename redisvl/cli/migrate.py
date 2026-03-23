import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from redisvl.cli.utils import add_redis_connection_options, create_redis_url
from redisvl.migration import (
    AsyncMigrationExecutor,
    BatchMigrationExecutor,
    BatchMigrationPlanner,
    MigrationExecutor,
    MigrationPlanner,
    MigrationValidator,
)
from redisvl.migration.utils import (
    list_indexes,
    load_migration_plan,
    load_yaml,
    write_benchmark_report,
    write_migration_report,
)
from redisvl.migration.wizard import MigrationWizard
from redisvl.utils.log import get_logger

logger = get_logger("[RedisVL]")


class Migrate:
    usage = "\n".join(
        [
            "rvl migrate <command> [<args>]\n",
            "Commands:",
            "\thelper       Show migration guidance and supported capabilities",
            "\tlist         List all available indexes",
            "\tplan         Generate a migration plan for a document-preserving drop/recreate migration",
            "\twizard       Interactively build a migration plan and schema patch",
            "\tapply        Execute a reviewed drop/recreate migration plan (use --async for large migrations)",
            "\tvalidate     Validate a completed migration plan against the live index",
            "",
            "Batch Commands:",
            "\tbatch-plan   Generate a batch migration plan for multiple indexes",
            "\tbatch-apply  Execute a batch migration plan with checkpointing",
            "\tbatch-resume Resume an interrupted batch migration",
            "\tbatch-status Show status of an in-progress or completed batch migration",
            "\n",
        ]
    )

    def __init__(self):
        parser = argparse.ArgumentParser(usage=self.usage)
        parser.add_argument("command", help="Subcommand to run")

        args = parser.parse_args(sys.argv[2:3])
        # Convert dashes to underscores for method lookup (e.g., batch-plan -> batch_plan)
        command = args.command.replace("-", "_")
        if not hasattr(self, command):
            parser.print_help()
            exit(0)

        try:
            getattr(self, command)()
        except Exception as e:
            logger.error(e)
            exit(1)

    def helper(self):
        parser = argparse.ArgumentParser(
            usage="rvl migrate helper [--host <host> --port <port> | --url <redis_url>]"
        )
        parser = add_redis_connection_options(parser)
        args = parser.parse_args(sys.argv[3:])
        redis_url = create_redis_url(args)
        indexes = list_indexes(redis_url=redis_url)

        print(
            """RedisVL Index Migrator

Available indexes:"""
        )
        if indexes:
            for position, index_name in enumerate(indexes, start=1):
                print(f"  {position}. {index_name}")
        else:
            print("  (none found)")

        print(
            """
Supported changes:
  - Adding or removing non-vector fields (text, tag, numeric, geo)
  - Changing field options (sortable, separator, weight)
  - Changing vector algorithm (FLAT, HNSW, SVS_VAMANA)
  - Changing distance metric (COSINE, L2, IP)
  - Tuning algorithm parameters (M, EF_CONSTRUCTION)
  - Quantizing vectors (float32 to float16/bfloat16/int8/uint8)
  - Changing key prefix (renames all keys)
  - Renaming fields (updates all documents)
  - Renaming the index

Not yet supported:
  - Changing vector dimensions
  - Changing storage type (hash to JSON)

Commands:
  rvl migrate list                                  List all indexes
  rvl migrate wizard --index <name>                 Guided migration builder
  rvl migrate plan --index <name> --schema-patch <patch.yaml>
  rvl migrate apply --plan <plan.yaml>
  rvl migrate validate --plan <plan.yaml>"""
        )

    def list(self):
        parser = argparse.ArgumentParser(
            usage="rvl migrate list [--host <host> --port <port> | --url <redis_url>]"
        )
        parser = add_redis_connection_options(parser)
        args = parser.parse_args(sys.argv[3:])
        redis_url = create_redis_url(args)
        indexes = list_indexes(redis_url=redis_url)
        print("Available indexes:")
        for position, index_name in enumerate(indexes, start=1):
            print(f"{position}. {index_name}")

    def plan(self):
        parser = argparse.ArgumentParser(
            usage=(
                "rvl migrate plan --index <name> "
                "(--schema-patch <patch.yaml> | --target-schema <schema.yaml>)"
            )
        )
        parser.add_argument("-i", "--index", help="Source index name", required=True)
        parser.add_argument("--schema-patch", help="Path to a schema patch file")
        parser.add_argument("--target-schema", help="Path to a target schema file")
        parser.add_argument(
            "--plan-out",
            help="Path to write migration_plan.yaml",
            default="migration_plan.yaml",
        )
        parser.add_argument(
            "--key-sample-limit",
            help="Maximum number of keys to sample from the index keyspace",
            type=int,
            default=10,
        )
        parser = add_redis_connection_options(parser)

        args = parser.parse_args(sys.argv[3:])
        redis_url = create_redis_url(args)
        planner = MigrationPlanner(key_sample_limit=args.key_sample_limit)
        plan = planner.create_plan(
            args.index,
            redis_url=redis_url,
            schema_patch_path=args.schema_patch,
            target_schema_path=args.target_schema,
        )
        planner.write_plan(plan, args.plan_out)
        self._print_plan_summary(args.plan_out, plan)

    def wizard(self):
        parser = argparse.ArgumentParser(
            usage=(
                "rvl migrate wizard [--index <name>] "
                "[--patch <existing_patch.yaml>] "
                "[--plan-out <migration_plan.yaml>] [--patch-out <schema_patch.yaml>]"
            )
        )
        parser.add_argument("-i", "--index", help="Source index name", required=False)
        parser.add_argument(
            "--patch",
            help="Load an existing schema patch to continue editing",
            default=None,
        )
        parser.add_argument(
            "--plan-out",
            help="Path to write migration_plan.yaml",
            default="migration_plan.yaml",
        )
        parser.add_argument(
            "--patch-out",
            help="Path to write schema_patch.yaml (for later editing)",
            default="schema_patch.yaml",
        )
        parser.add_argument(
            "--target-schema-out",
            help="Optional path to write the merged target schema",
            default=None,
        )
        parser.add_argument(
            "--key-sample-limit",
            help="Maximum number of keys to sample from the index keyspace",
            type=int,
            default=10,
        )
        parser = add_redis_connection_options(parser)
        args = parser.parse_args(sys.argv[3:])

        redis_url = create_redis_url(args)
        wizard = MigrationWizard(
            planner=MigrationPlanner(key_sample_limit=args.key_sample_limit)
        )
        plan = wizard.run(
            index_name=args.index,
            redis_url=redis_url,
            existing_patch_path=args.patch,
            plan_out=args.plan_out,
            patch_out=args.patch_out,
            target_schema_out=args.target_schema_out,
        )
        self._print_plan_summary(args.plan_out, plan)

    def apply(self):
        parser = argparse.ArgumentParser(
            usage=(
                "rvl migrate apply --plan <migration_plan.yaml> "
                "[--async] [--report-out <migration_report.yaml>]"
            )
        )
        parser.add_argument("--plan", help="Path to migration_plan.yaml", required=True)
        parser.add_argument(
            "--async",
            dest="use_async",
            help="Use async executor (recommended for large migrations with quantization)",
            action="store_true",
        )
        parser.add_argument(
            "--report-out",
            help="Path to write migration_report.yaml",
            default="migration_report.yaml",
        )
        parser.add_argument(
            "--benchmark-out",
            help="Optional path to write benchmark_report.yaml",
            default=None,
        )
        parser.add_argument(
            "--query-check-file",
            help="Optional YAML file containing fetch_ids and keys_exist checks",
            default=None,
        )
        parser = add_redis_connection_options(parser)
        args = parser.parse_args(sys.argv[3:])

        redis_url = create_redis_url(args)
        plan = load_migration_plan(args.plan)

        if args.use_async:
            report = asyncio.run(
                self._apply_async(plan, redis_url, args.query_check_file)
            )
        else:
            report = self._apply_sync(plan, redis_url, args.query_check_file)

        write_migration_report(report, args.report_out)
        if args.benchmark_out:
            write_benchmark_report(report, args.benchmark_out)
        self._print_report_summary(args.report_out, report, args.benchmark_out)

    def _apply_sync(self, plan, redis_url: str, query_check_file: Optional[str]):
        """Execute migration synchronously."""
        executor = MigrationExecutor()

        print(f"\nApplying migration to '{plan.source.index_name}'...")

        def progress_callback(step: str, detail: Optional[str]) -> None:
            step_labels = {
                "drop": "[1/5] Drop index",
                "quantize": "[2/5] Quantize vectors",
                "create": "[3/5] Create index",
                "index": "[4/5] Re-indexing",
                "validate": "[5/5] Validate",
            }
            label = step_labels.get(step, step)
            if detail and not detail.startswith("done"):
                print(f"  {label}: {detail}    ", end="\r", flush=True)
            else:
                print(f"  {label}: {detail}    ")

        report = executor.apply(
            plan,
            redis_url=redis_url,
            query_check_file=query_check_file,
            progress_callback=progress_callback,
        )

        self._print_apply_result(report)
        return report

    async def _apply_async(self, plan, redis_url: str, query_check_file: Optional[str]):
        """Execute migration asynchronously (non-blocking for large quantization jobs)."""
        executor = AsyncMigrationExecutor()

        print(f"\nApplying migration to '{plan.source.index_name}' (async mode)...")

        def progress_callback(step: str, detail: Optional[str]) -> None:
            step_labels = {
                "drop": "[1/5] Drop index",
                "quantize": "[2/5] Quantize vectors",
                "create": "[3/5] Create index",
                "index": "[4/5] Re-indexing",
                "validate": "[5/5] Validate",
            }
            label = step_labels.get(step, step)
            if detail and not detail.startswith("done"):
                print(f"  {label}: {detail}    ", end="\r", flush=True)
            else:
                print(f"  {label}: {detail}    ")

        report = await executor.apply(
            plan,
            redis_url=redis_url,
            query_check_file=query_check_file,
            progress_callback=progress_callback,
        )

        self._print_apply_result(report)
        return report

    def _print_apply_result(self, report) -> None:
        """Print the result summary after migration apply."""
        if report.result == "succeeded":
            total_time = report.timings.total_migration_duration_seconds or 0
            downtime = report.timings.downtime_duration_seconds or 0
            print(f"\nMigration completed in {total_time}s (downtime: {downtime}s)")
        else:
            print(f"\nMigration {report.result}")
            if report.validation.errors:
                for error in report.validation.errors:
                    print(f"  ERROR: {error}")

    def validate(self):
        parser = argparse.ArgumentParser(
            usage=(
                "rvl migrate validate --plan <migration_plan.yaml> "
                "[--report-out <migration_report.yaml>]"
            )
        )
        parser.add_argument("--plan", help="Path to migration_plan.yaml", required=True)
        parser.add_argument(
            "--report-out",
            help="Path to write migration_report.yaml",
            default="migration_report.yaml",
        )
        parser.add_argument(
            "--benchmark-out",
            help="Optional path to write benchmark_report.yaml",
            default=None,
        )
        parser.add_argument(
            "--query-check-file",
            help="Optional YAML file containing fetch_ids and keys_exist checks",
            default=None,
        )
        parser = add_redis_connection_options(parser)
        args = parser.parse_args(sys.argv[3:])

        redis_url = create_redis_url(args)
        plan = load_migration_plan(args.plan)
        validator = MigrationValidator()
        validation, target_info, validation_duration = validator.validate(
            plan,
            redis_url=redis_url,
            query_check_file=args.query_check_file,
        )

        from redisvl.migration.models import (
            MigrationBenchmarkSummary,
            MigrationReport,
            MigrationTimings,
        )
        from redisvl.migration.utils import timestamp_utc

        source_size = float(
            plan.source.stats_snapshot.get("vector_index_sz_mb", 0) or 0
        )
        target_size = float(target_info.get("vector_index_sz_mb", 0) or 0)

        report = MigrationReport(
            source_index=plan.source.index_name,
            target_index=plan.merged_target_schema["index"]["name"],
            result="succeeded" if not validation.errors else "failed",
            started_at=timestamp_utc(),
            finished_at=timestamp_utc(),
            timings=MigrationTimings(validation_duration_seconds=validation_duration),
            validation=validation,
            benchmark_summary=MigrationBenchmarkSummary(
                source_index_size_mb=round(source_size, 3),
                target_index_size_mb=round(target_size, 3),
                index_size_delta_mb=round(target_size - source_size, 3),
            ),
            warnings=list(plan.warnings),
            manual_actions=(
                ["Review validation errors before proceeding."]
                if validation.errors
                else []
            ),
        )
        write_migration_report(report, args.report_out)
        if args.benchmark_out:
            write_benchmark_report(report, args.benchmark_out)
        self._print_report_summary(args.report_out, report, args.benchmark_out)

    def _print_plan_summary(self, plan_out: str, plan) -> None:
        import os

        abs_path = os.path.abspath(plan_out)
        print(f"Migration plan written to {abs_path}")
        print(f"Mode: {plan.mode}")
        print(f"Supported: {plan.diff_classification.supported}")
        if plan.warnings:
            print("Warnings:")
            for warning in plan.warnings:
                print(f"- {warning}")
        if plan.diff_classification.blocked_reasons:
            print("Blocked reasons:")
            for reason in plan.diff_classification.blocked_reasons:
                print(f"- {reason}")

        print("\nNext steps:")
        print(f"  Review the plan:     cat {plan_out}")
        print(f"  Apply the migration: rvl migrate apply --plan {plan_out}")
        print(f"  Validate the result: rvl migrate validate --plan {plan_out}")
        print(
            f"\nTo add more changes:   rvl migrate wizard --index {plan.source.index_name} --patch schema_patch.yaml"
        )
        print(
            f"To start over:         rvl migrate wizard --index {plan.source.index_name}"
        )
        print(f"To cancel:             rm {plan_out}")

    def _print_report_summary(
        self,
        report_out: str,
        report,
        benchmark_out: Optional[str],
    ) -> None:
        print(f"Migration report written to {report_out}")
        print(f"Result: {report.result}")
        print(f"Schema match: {report.validation.schema_match}")
        print(f"Doc count match: {report.validation.doc_count_match}")
        print(f"Key sample exists: {report.validation.key_sample_exists}")
        print(f"Indexing failures delta: {report.validation.indexing_failures_delta}")
        if report.validation.errors:
            print("Errors:")
            for error in report.validation.errors:
                print(f"- {error}")
        if report.manual_actions:
            print("Manual actions:")
            for action in report.manual_actions:
                print(f"- {action}")
        if benchmark_out:
            print(f"Benchmark report written to {benchmark_out}")

    # -------------------------------------------------------------------------
    # Batch migration commands
    # -------------------------------------------------------------------------

    def batch_plan(self):
        """Generate a batch migration plan for multiple indexes."""
        parser = argparse.ArgumentParser(
            usage=(
                "rvl migrate batch-plan --schema-patch <patch.yaml> "
                "(--pattern <glob> | --indexes <name1,name2> | --indexes-file <file>)"
            )
        )
        parser.add_argument(
            "--schema-patch", help="Path to shared schema patch file", required=True
        )
        parser.add_argument(
            "--pattern", help="Glob pattern to match index names (e.g., '*_idx')"
        )
        parser.add_argument("--indexes", help="Comma-separated list of index names")
        parser.add_argument(
            "--indexes-file", help="File with index names (one per line)"
        )
        parser.add_argument(
            "--failure-policy",
            help="How to handle failures: fail_fast or continue_on_error",
            choices=["fail_fast", "continue_on_error"],
            default="fail_fast",
        )
        parser.add_argument(
            "--plan-out",
            help="Path to write batch_plan.yaml",
            default="batch_plan.yaml",
        )
        parser = add_redis_connection_options(parser)
        args = parser.parse_args(sys.argv[3:])

        redis_url = create_redis_url(args)
        indexes = args.indexes.split(",") if args.indexes else None

        planner = BatchMigrationPlanner()
        batch_plan = planner.create_batch_plan(
            indexes=indexes,
            pattern=args.pattern,
            indexes_file=args.indexes_file,
            schema_patch_path=args.schema_patch,
            redis_url=redis_url,
            failure_policy=args.failure_policy,
        )

        planner.write_batch_plan(batch_plan, args.plan_out)
        self._print_batch_plan_summary(args.plan_out, batch_plan)

    def batch_apply(self):
        """Execute a batch migration plan with checkpointing."""
        parser = argparse.ArgumentParser(
            usage=(
                "rvl migrate batch-apply --plan <batch_plan.yaml> "
                "[--state <batch_state.yaml>] [--report-dir <./reports>]"
            )
        )
        parser.add_argument("--plan", help="Path to batch_plan.yaml", required=True)
        parser.add_argument(
            "--accept-data-loss",
            help="Acknowledge that quantization is lossy and cannot be reverted",
            action="store_true",
        )
        parser.add_argument(
            "--state",
            help="Path to checkpoint state file",
            default="batch_state.yaml",
        )
        parser.add_argument(
            "--report-dir",
            help="Directory for per-index migration reports",
            default="./reports",
        )
        parser = add_redis_connection_options(parser)
        args = parser.parse_args(sys.argv[3:])

        # Load batch plan
        from redisvl.migration.models import BatchPlan

        plan_data = load_yaml(args.plan)
        batch_plan = BatchPlan.model_validate(plan_data)

        # Check for quantization warning
        if batch_plan.requires_quantization and not args.accept_data_loss:
            print(
                """WARNING: This batch migration includes quantization (e.g., float32 -> float16).
         Vector data will be modified. Original precision cannot be recovered.
         To proceed, add --accept-data-loss flag.

         If you need to preserve original vectors, backup your data first:
           redis-cli BGSAVE"""
            )
            return

        redis_url = create_redis_url(args)
        executor = BatchMigrationExecutor()

        def progress_callback(
            index_name: str, position: int, total: int, status: str
        ) -> None:
            print(f"[{position}/{total}] {index_name}: {status}")

        report = executor.apply(
            batch_plan,
            batch_plan_path=args.plan,
            state_path=args.state,
            report_dir=args.report_dir,
            redis_url=redis_url,
            progress_callback=progress_callback,
        )

        self._print_batch_report_summary(report)

    def batch_resume(self):
        """Resume an interrupted batch migration."""
        parser = argparse.ArgumentParser(
            usage=(
                "rvl migrate batch-resume --state <batch_state.yaml> "
                "[--plan <batch_plan.yaml>] [--retry-failed]"
            )
        )
        parser.add_argument(
            "--state", help="Path to checkpoint state file", required=True
        )
        parser.add_argument(
            "--plan", help="Path to batch_plan.yaml (optional, uses state.plan_path)"
        )
        parser.add_argument(
            "--retry-failed",
            help="Retry previously failed indexes",
            action="store_true",
        )
        parser.add_argument(
            "--report-dir",
            help="Directory for per-index migration reports",
            default="./reports",
        )
        parser = add_redis_connection_options(parser)
        args = parser.parse_args(sys.argv[3:])

        redis_url = create_redis_url(args)
        executor = BatchMigrationExecutor()

        def progress_callback(
            index_name: str, position: int, total: int, status: str
        ) -> None:
            print(f"[{position}/{total}] {index_name}: {status}")

        report = executor.resume(
            args.state,
            batch_plan_path=args.plan,
            retry_failed=args.retry_failed,
            report_dir=args.report_dir,
            redis_url=redis_url,
            progress_callback=progress_callback,
        )

        self._print_batch_report_summary(report)

    def batch_status(self):
        """Show status of an in-progress or completed batch migration."""
        parser = argparse.ArgumentParser(
            usage="rvl migrate batch-status --state <batch_state.yaml>"
        )
        parser.add_argument(
            "--state", help="Path to checkpoint state file", required=True
        )
        args = parser.parse_args(sys.argv[3:])

        state_path = Path(args.state).resolve()
        if not state_path.exists():
            print(f"State file not found: {args.state}")
            return

        from redisvl.migration.models import BatchState

        state_data = load_yaml(args.state)
        state = BatchState.model_validate(state_data)

        print(
            f"""Batch ID: {state.batch_id}
Started at: {state.started_at}
Updated at: {state.updated_at}
Current index: {state.current_index or '(none)'}
Remaining: {len(state.remaining)}
Completed: {len(state.completed)}
  - Succeeded: {state.success_count}
  - Failed: {state.failed_count}
  - Skipped: {state.skipped_count}"""
        )

        if state.completed:
            print("\nCompleted indexes:")
            for idx in state.completed:
                if idx.status == "succeeded":
                    status_icon = "[OK]"
                elif idx.status == "skipped":
                    status_icon = "[SKIP]"
                else:
                    status_icon = "[FAIL]"
                print(f"  {status_icon} {idx.name}")
                if idx.error:
                    print(f"       Error: {idx.error}")

        if state.remaining:
            print(f"\nRemaining indexes ({len(state.remaining)}):")
            for name in state.remaining[:10]:
                print(f"  - {name}")
            if len(state.remaining) > 10:
                print(f"  ... and {len(state.remaining) - 10} more")

    def _print_batch_plan_summary(self, plan_out: str, batch_plan) -> None:
        """Print summary after generating batch plan."""
        import os

        abs_path = os.path.abspath(plan_out)
        print(f"Batch plan written to {abs_path}")
        print(f"Batch ID: {batch_plan.batch_id}")
        print(f"Mode: {batch_plan.mode}")
        print(f"Failure policy: {batch_plan.failure_policy}")
        print(f"Requires quantization: {batch_plan.requires_quantization}")
        print(f"Total indexes: {len(batch_plan.indexes)}")
        print(f"  - Applicable: {batch_plan.applicable_count}")
        print(f"  - Skipped: {batch_plan.skipped_count}")

        if batch_plan.skipped_count > 0:
            print("\nSkipped indexes:")
            for idx in batch_plan.indexes:
                if not idx.applicable:
                    print(f"  - {idx.name}: {idx.skip_reason}")

        print(
            f"""
Next steps:
  Review the plan:      cat {plan_out}
  Apply the migration:  rvl migrate batch-apply --plan {plan_out}"""
        )

        if batch_plan.requires_quantization:
            print("                       (add --accept-data-loss for quantization)")

    def _print_batch_report_summary(self, report) -> None:
        """Print summary after batch migration completes."""
        print(f"\nBatch migration {report.status}")
        print(f"Batch ID: {report.batch_id}")
        print(f"Duration: {report.summary.total_duration_seconds}s")
        print(f"Total: {report.summary.total_indexes}")
        print(f"  - Succeeded: {report.summary.successful}")
        print(f"  - Failed: {report.summary.failed}")
        print(f"  - Skipped: {report.summary.skipped}")

        if report.summary.failed > 0:
            print("\nFailed indexes:")
            for idx in report.indexes:
                if idx.status == "failed":
                    print(f"  - {idx.name}: {idx.error}")
