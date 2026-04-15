import argparse
import asyncio
import os
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
    MigrationWizard,
)
from redisvl.migration.utils import (
    detect_aof_enabled,
    estimate_disk_space,
    list_indexes,
    load_migration_plan,
    load_yaml,
    write_benchmark_report,
    write_migration_report,
)
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.utils.log import get_logger

logger = get_logger("[RedisVL]")


class Migrate:
    usage = "\n".join(
        [
            "rvl migrate <command> [<args>]\n",
            "Commands:",
            "\thelper       Show migration guidance and supported capabilities",
            "\twizard       Interactively build a migration plan and schema patch",
            "\tplan         Generate a migration plan for a document-preserving drop/recreate migration",
            "\tapply        Execute a reviewed drop/recreate migration plan (use --async for large migrations)",
            "\testimate     Estimate disk space required for a migration plan (dry-run, no mutations)",
            "\trollback     Restore original vectors from a backup directory (undo quantization)",
            "\tvalidate     Validate a completed migration plan against the live index",
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
        command = args.command.replace("-", "_")
        if not hasattr(self, command):
            print(f"Unknown subcommand: {args.command}")
            parser.print_help()
            sys.exit(1)

        try:
            getattr(self, command)()
        except Exception as e:
            logger.error(e)
            sys.exit(1)

    def helper(self):
        parser = argparse.ArgumentParser(
            usage="rvl migrate helper [--host <host> --port <port> | --url <redis_url>]"
        )
        parser = add_redis_connection_options(parser)
        args = parser.parse_args(sys.argv[3:])
        redis_url = create_redis_url(args)
        indexes = list_indexes(redis_url=redis_url)

        print("RedisVL Index Migrator\n\nAvailable indexes:")
        if indexes:
            for position, index_name in enumerate(indexes, start=1):
                print(f"  {position}. {index_name}")
        else:
            print("  (none found)")

        print(
            """\nSupported changes:
  - Adding or removing non-vector fields (text, tag, numeric, geo)
  - Changing field options (sortable, separator, weight)
  - Changing vector algorithm (FLAT, HNSW, SVS-VAMANA)
  - Changing distance metric (COSINE, L2, IP)
  - Tuning algorithm parameters (M, EF_CONSTRUCTION, EF_RUNTIME, EPSILON)
  - Quantizing vectors (float32 to float16/bfloat16/int8/uint8)
  - Changing key prefix (renames all keys)
  - Renaming fields (updates all documents)
  - Renaming the index

Not yet supported:
  - Changing vector dimensions
  - Changing storage type (hash to JSON)

Commands:
  rvl migrate wizard --index <name>                 Guided migration builder
  rvl migrate plan --index <name> --schema-patch <patch.yaml>
  rvl migrate apply --plan <plan.yaml>
  rvl migrate validate --plan <plan.yaml>

  Tip: use 'rvl index listall' to see available indexes."""
        )

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

    def apply(self):
        parser = argparse.ArgumentParser(
            usage=(
                "rvl migrate apply --plan <migration_plan.yaml> "
                "[--async] [--backup-dir <dir>] [--workers N] "
                "[--report-out <migration_report.yaml>]"
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
            "--backup-dir",
            dest="backup_dir",
            help="Directory for vector backup files. Enables crash-safe resume and rollback.",
            default=None,
        )
        parser.add_argument(
            "--batch-size",
            dest="batch_size",
            type=int,
            help="Keys per pipeline batch (default 500)",
            default=500,
        )
        parser.add_argument(
            "--workers",
            dest="num_workers",
            type=int,
            help="Number of parallel workers for quantization (default 1). "
            "Each worker gets its own Redis connection. Requires --backup-dir.",
            default=1,
        )
        parser.add_argument(
            "--keep-backup",
            dest="keep_backup",
            action="store_true",
            help="Keep backup files after successful migration (default: auto-delete).",
            default=False,
        )
        # Deprecated alias for --backup-dir (was --resume in previous versions)
        parser.add_argument(
            "--resume",
            dest="legacy_resume",
            help=argparse.SUPPRESS,  # hidden from help
            default=None,
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

        # Validate --workers
        if args.num_workers < 1:
            parser.error("--workers must be >= 1")

        # Handle deprecated --resume flag
        if args.legacy_resume is not None:
            import warnings

            # Fail fast if the value looks like a checkpoint file (old semantics)
            if os.path.isfile(args.legacy_resume) or args.legacy_resume.endswith(
                (".yaml", ".yml")
            ):
                parser.error(
                    "--resume semantics have changed: it now expects a backup "
                    "directory, not a checkpoint file. Use --backup-dir <dir> instead."
                )

            warnings.warn(
                "--resume is deprecated and will be removed in a future version. "
                "Use --backup-dir instead: the backup directory replaces "
                "checkpoint files for crash-safe resume and rollback.",
                DeprecationWarning,
                stacklevel=1,
            )
            if args.backup_dir is None:
                args.backup_dir = args.legacy_resume

        # Validate --workers > 1 requires --backup-dir
        if args.num_workers > 1 and args.backup_dir is None:
            parser.error("--workers > 1 requires --backup-dir")

        redis_url = create_redis_url(args)
        plan = load_migration_plan(args.plan)

        # Print disk space estimate for quantization migrations
        aof_enabled = False
        try:
            client = RedisConnectionFactory.get_redis_connection(redis_url=redis_url)
            try:
                aof_enabled = detect_aof_enabled(client)
            finally:
                client.close()
        except Exception as exc:
            logger.debug("Could not detect AOF for CLI preflight estimate: %s", exc)

        disk_estimate = estimate_disk_space(plan, aof_enabled=aof_enabled)
        if disk_estimate.has_quantization:
            print(f"\n{disk_estimate.summary()}\n")

        if args.use_async:
            report = asyncio.run(
                self._apply_async(
                    plan,
                    redis_url,
                    args.query_check_file,
                    backup_dir=args.backup_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    keep_backup=args.keep_backup,
                )
            )
        else:
            report = self._apply_sync(
                plan,
                redis_url,
                args.query_check_file,
                backup_dir=args.backup_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                keep_backup=args.keep_backup,
            )

        write_migration_report(report, args.report_out)
        if args.benchmark_out:
            write_benchmark_report(report, args.benchmark_out)
        self._print_report_summary(args.report_out, report, args.benchmark_out)

    def estimate(self):
        """Estimate disk space required for a migration plan (dry-run)."""
        parser = argparse.ArgumentParser(
            usage="rvl migrate estimate --plan <migration_plan.yaml>"
        )
        parser.add_argument("--plan", help="Path to migration_plan.yaml", required=True)
        parser.add_argument(
            "--aof-enabled",
            action="store_true",
            help="Include AOF growth in the disk space estimate",
        )
        args = parser.parse_args(sys.argv[3:])

        plan = load_migration_plan(args.plan)
        disk_estimate = estimate_disk_space(plan, aof_enabled=args.aof_enabled)
        print(disk_estimate.summary())

    # Phases that indicate a safe/complete backup for rollback
    _SAFE_ROLLBACK_PHASES = frozenset({"ready", "active", "completed"})

    def rollback(self):
        """Restore original vectors from a backup directory (undo quantization)."""
        parser = argparse.ArgumentParser(
            usage=(
                "rvl migrate rollback --backup-dir <dir> "
                "[--index <name>] [--yes] [--force] [--url <redis_url>]"
            )
        )
        parser.add_argument(
            "--backup-dir",
            dest="backup_dir",
            help="Directory containing vector backup files from a prior migration",
            required=True,
        )
        parser.add_argument(
            "--index",
            dest="index_name",
            help="Only restore backups for this index name (filters by backup header)",
            default=None,
        )
        parser.add_argument(
            "--yes",
            "-y",
            dest="yes",
            action="store_true",
            help="Skip confirmation prompt for multi-index rollback",
            default=False,
        )
        parser.add_argument(
            "--force",
            dest="force",
            action="store_true",
            help="Proceed even if backup phase indicates incomplete dump",
            default=False,
        )
        parser = add_redis_connection_options(parser)
        args = parser.parse_args(sys.argv[3:])

        redis_url = create_redis_url(args)

        from redisvl.migration.backup import VectorBackup
        from redisvl.redis.connection import RedisConnectionFactory

        # Find backup files in the directory
        backup_dir = args.backup_dir
        if not os.path.isdir(backup_dir):
            print(f"Error: backup directory not found: {backup_dir}")
            sys.exit(1)

        # Look for .header files to find backups
        header_files = sorted(Path(backup_dir).glob("*.header"))
        if not header_files:
            print(f"Error: no backup files found in {backup_dir}")
            sys.exit(1)

        # Derive backup base paths (strip .header suffix)
        backup_paths = [str(h.with_suffix("")) for h in header_files]

        # Load, filter, and validate backups
        backups_to_restore = []
        for bp in backup_paths:
            backup = VectorBackup.load(bp)
            if backup is None:
                print(f"  Skipping {bp}: could not load backup")
                continue
            if args.index_name and backup.header.index_name != args.index_name:
                print(
                    f"  Skipping {os.path.basename(bp)}: "
                    f"index '{backup.header.index_name}' != '{args.index_name}'"
                )
                continue
            # Gate on backup phase — refuse incomplete backups unless --force
            if backup.header.phase not in self._SAFE_ROLLBACK_PHASES:
                if args.force:
                    print(
                        f"  Warning: {os.path.basename(bp)} has phase "
                        f"'{backup.header.phase}' (incomplete dump) — "
                        f"proceeding due to --force"
                    )
                else:
                    print(
                        f"  Skipping {os.path.basename(bp)}: backup phase "
                        f"'{backup.header.phase}' indicates incomplete dump. "
                        f"Use --force to restore from partial backups."
                    )
                    continue
            backups_to_restore.append((bp, backup))

        if not backups_to_restore:
            print("Error: no matching backup files found")
            sys.exit(1)

        # Require --index or --yes when multiple distinct indexes detected
        distinct_indexes = {b.header.index_name for _, b in backups_to_restore}
        if len(distinct_indexes) > 1 and not args.index_name and not args.yes:
            print(
                f"Error: found backups for {len(distinct_indexes)} distinct indexes: "
                f"{', '.join(sorted(distinct_indexes))}. "
                f"Use --index to filter or --yes to restore all."
            )
            sys.exit(1)

        client = RedisConnectionFactory.get_redis_connection(redis_url=redis_url)
        total_restored = 0
        try:
            for bp, backup in backups_to_restore:
                print(
                    f"Restoring from: {os.path.basename(bp)} "
                    f"(index={backup.header.index_name}, "
                    f"phase={backup.header.phase}, "
                    f"batches={backup.header.dump_completed_batches})"
                )

                batch_count = 0
                for keys, originals in backup.iter_batches():
                    pipe = client.pipeline(transaction=False)
                    batch_restored = 0
                    for key in keys:
                        if key in originals:
                            for field_name, original_bytes in originals[key].items():
                                pipe.hset(key, field_name, original_bytes)
                            batch_restored += 1
                    pipe.execute()
                    batch_count += 1
                    total_restored += batch_restored
                    if batch_count % 10 == 0:
                        print(
                            f"  Restored {total_restored:,} vectors "
                            f"({batch_count}/{backup.header.dump_completed_batches} batches)"
                        )

                print(
                    f"  Done: {batch_count} batches restored from {os.path.basename(bp)}"
                )
        finally:
            client.close()

        print(
            f"\nRollback complete: {total_restored:,} vectors restored to original values"
        )
        print(
            "Note: You may need to recreate the original index schema "
            "(FT.CREATE) if the index was changed during migration."
        )

    @staticmethod
    def _make_progress_callback():
        """Create a progress callback for migration apply."""
        step_labels = {
            "enumerate": "[1/8] Enumerate keys",
            "bgsave": "[2/8] BGSAVE snapshot",
            "field_rename": "[3/8] Rename fields",
            "drop": "[4/8] Drop index",
            "key_rename": "[5/8] Rename keys",
            "quantize": "[6/8] Quantize vectors",
            "create": "[7/8] Create index",
            "index": "[8/8] Re-indexing",
            "validate": "Validate",
        }

        def progress_callback(step: str, detail: Optional[str]) -> None:
            label = step_labels.get(step, step)
            if detail and not detail.startswith("done"):
                print(f"  {label}: {detail}    ", end="\r", flush=True)
            else:
                print(f"  {label}: {detail}    ")

        return progress_callback

    def _apply_sync(
        self,
        plan,
        redis_url: str,
        query_check_file: Optional[str],
        backup_dir: Optional[str] = None,
        batch_size: int = 500,
        num_workers: int = 1,
        keep_backup: bool = False,
    ):
        """Execute migration synchronously."""
        executor = MigrationExecutor()

        print(f"\nApplying migration to '{plan.source.index_name}'...")

        report = executor.apply(
            plan,
            redis_url=redis_url,
            query_check_file=query_check_file,
            progress_callback=self._make_progress_callback(),
            backup_dir=backup_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            keep_backup=keep_backup,
        )

        self._print_apply_result(report)
        return report

    async def _apply_async(
        self,
        plan,
        redis_url: str,
        query_check_file: Optional[str],
        backup_dir: Optional[str] = None,
        batch_size: int = 500,
        num_workers: int = 1,
        keep_backup: bool = False,
    ):
        """Execute migration asynchronously (non-blocking for large quantization jobs)."""
        executor = AsyncMigrationExecutor()

        print(f"\nApplying migration to '{plan.source.index_name}' (async mode)...")

        report = await executor.apply(
            plan,
            redis_url=redis_url,
            query_check_file=query_check_file,
            progress_callback=self._make_progress_callback(),
            backup_dir=backup_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            keep_backup=keep_backup,
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

        from redisvl.migration.utils import timestamp_utc

        started_at = timestamp_utc()
        validation, target_info, validation_duration = validator.validate(
            plan,
            redis_url=redis_url,
            query_check_file=args.query_check_file,
        )
        finished_at = timestamp_utc()

        from redisvl.migration.models import (
            MigrationBenchmarkSummary,
            MigrationReport,
            MigrationTimings,
        )

        source_size = float(
            plan.source.stats_snapshot.get("vector_index_sz_mb", 0) or 0
        )
        target_size = float(target_info.get("vector_index_sz_mb", 0) or 0)

        report = MigrationReport(
            source_index=plan.source.index_name,
            target_index=plan.merged_target_schema["index"]["name"],
            result="succeeded" if not validation.errors else "failed",
            started_at=started_at,
            finished_at=finished_at,
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
        print(
            f"""Migration plan written to {abs_path}
Mode: {plan.mode}
Supported: {plan.diff_classification.supported}"""
        )
        if plan.warnings:
            print("Warnings:")
            for warning in plan.warnings:
                print(f"- {warning}")
        if plan.diff_classification.blocked_reasons:
            print("Blocked reasons:")
            for reason in plan.diff_classification.blocked_reasons:
                print(f"- {reason}")

        print(
            f"""\nNext steps:
  Review the plan:     cat {plan_out}
  Apply the migration: rvl migrate apply --plan {plan_out}
  Validate the result: rvl migrate validate --plan {plan_out}
  To cancel:           rm {plan_out}"""
        )

    def _print_report_summary(
        self,
        report_out: str,
        report,
        benchmark_out: Optional[str],
    ) -> None:
        print(
            f"""Migration report written to {report_out}
Result: {report.result}
Schema match: {report.validation.schema_match}
Doc count match: {report.validation.doc_count_match}
Key sample exists: {report.validation.key_sample_exists}
Indexing failures delta: {report.validation.indexing_failures_delta}"""
        )
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
        indexes = (
            [idx.strip() for idx in args.indexes.split(",") if idx.strip()]
            if args.indexes
            else None
        )

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

        from redisvl.migration.models import BatchPlan

        plan_data = load_yaml(args.plan)
        batch_plan = BatchPlan.model_validate(plan_data)

        if batch_plan.requires_quantization and not args.accept_data_loss:
            print(
                """WARNING: This batch migration includes quantization (e.g., float32 -> float16).
         Vector data will be modified. Original precision cannot be recovered.
         To proceed, add --accept-data-loss flag.

         If you need to preserve original vectors, backup your data first:
           redis-cli BGSAVE"""
            )
            sys.exit(1)

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
            "--accept-data-loss",
            help="Acknowledge vector quantization data loss",
            action="store_true",
        )
        parser.add_argument(
            "--report-dir",
            help="Directory for per-index migration reports",
            default="./reports",
        )
        parser = add_redis_connection_options(parser)
        args = parser.parse_args(sys.argv[3:])

        # Load the batch plan to check for quantization safety gate
        executor = BatchMigrationExecutor()
        state = executor._load_state(args.state)
        plan_path = args.plan or state.plan_path or None
        if plan_path:
            batch_plan = executor._load_batch_plan(plan_path)
            if batch_plan.requires_quantization and not args.accept_data_loss:
                print(
                    """WARNING: This batch migration includes quantization (e.g., float32 -> float16).
         Vector data will be modified. Original precision cannot be recovered.
         To proceed, add --accept-data-loss flag.

         If you need to preserve original vectors, backup your data first:
           redis-cli BGSAVE"""
                )
                sys.exit(1)

        redis_url = create_redis_url(args)

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
            sys.exit(1)

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
                if idx.status == "success":
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
        print(
            f"""Batch plan written to {abs_path}
Batch ID: {batch_plan.batch_id}
Mode: {batch_plan.mode}
Failure policy: {batch_plan.failure_policy}
Requires quantization: {batch_plan.requires_quantization}
Total indexes: {len(batch_plan.indexes)}
  - Applicable: {batch_plan.applicable_count}
  - Skipped: {batch_plan.skipped_count}"""
        )

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
        print(
            f"""
Batch migration {report.status}
Batch ID: {report.batch_id}
Duration: {report.summary.total_duration_seconds}s
Total: {report.summary.total_indexes}
  - Succeeded: {report.summary.successful}
  - Failed: {report.summary.failed}
  - Skipped: {report.summary.skipped}"""
        )

        if report.summary.failed > 0:
            print("\nFailed indexes:")
            for idx in report.indexes:
                if idx.status == "failed":
                    print(f"  - {idx.name}: {idx.error}")
