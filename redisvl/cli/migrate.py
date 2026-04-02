import argparse
import sys
from typing import Optional

from redisvl.cli.utils import add_redis_connection_options, create_redis_url
from redisvl.migration import MigrationExecutor, MigrationPlanner, MigrationValidator
from redisvl.migration.utils import (
    detect_aof_enabled,
    estimate_disk_space,
    list_indexes,
    load_migration_plan,
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
            "\tlist         List all available indexes",
            "\tplan         Generate a migration plan for a document-preserving drop/recreate migration",
            "\tapply        Execute a reviewed drop/recreate migration plan",
            "\testimate     Estimate disk space required for a migration plan (dry-run, no mutations)",
            "\tvalidate     Validate a completed migration plan against the live index",
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
  rvl migrate list                                  List all indexes
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

    def apply(self):
        parser = argparse.ArgumentParser(
            usage=(
                "rvl migrate apply --plan <migration_plan.yaml> "
                "[--resume <checkpoint.yaml>] "
                "[--report-out <migration_report.yaml>]"
            )
        )
        parser.add_argument("--plan", help="Path to migration_plan.yaml", required=True)
        parser.add_argument(
            "--resume",
            dest="checkpoint_path",
            help="Path to quantization checkpoint file for crash-safe resume",
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

        report = self._apply_sync(
            plan, redis_url, args.query_check_file, args.checkpoint_path
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
        checkpoint_path: Optional[str] = None,
    ):
        """Execute migration synchronously."""
        executor = MigrationExecutor()

        print(f"\nApplying migration to '{plan.source.index_name}'...")

        report = executor.apply(
            plan,
            redis_url=redis_url,
            query_check_file=query_check_file,
            progress_callback=self._make_progress_callback(),
            checkpoint_path=checkpoint_path,
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
