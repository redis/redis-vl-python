import argparse
import sys
from argparse import Namespace
from typing import Optional

from redisvl.cli.utils import add_redis_connection_options, create_redis_url
from redisvl.migration import MigrationExecutor, MigrationPlanner, MigrationValidator
from redisvl.migration.utils import (
    list_indexes,
    load_migration_plan,
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
            "\thelper      Show migration guidance and supported capabilities",
            "\tlist        List all available indexes",
            "\tplan        Generate a migration plan for a document-preserving drop/recreate migration",
            "\twizard      Interactively build a migration plan and schema patch",
            "\tapply       Execute a reviewed drop/recreate migration plan",
            "\tvalidate    Validate a completed migration plan against the live index",
            "\n",
        ]
    )

    def __init__(self):
        parser = argparse.ArgumentParser(usage=self.usage)
        parser.add_argument("command", help="Subcommand to run")

        args = parser.parse_args(sys.argv[2:3])
        if not hasattr(self, args.command):
            parser.print_help()
            exit(0)

        try:
            getattr(self, args.command)()
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
  - Quantizing vectors (float32 to float16)

Not yet supported:
  - Changing vector dimensions
  - Changing key prefix or separator
  - Changing storage type (hash to JSON)
  - Renaming fields

Commands:
  rvl migrate list                                  List all indexes
  rvl migrate wizard --index <name>                 Guided migration builder
  rvl migrate plan --index <name> --schema-patch <patch.yaml>
  rvl migrate apply --plan <plan.yaml> --allow-downtime
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
                "rvl migrate apply --plan <migration_plan.yaml> --allow-downtime "
                "[--report-out <migration_report.yaml>]"
            )
        )
        parser.add_argument("--plan", help="Path to migration_plan.yaml", required=True)
        parser.add_argument(
            "--allow-downtime",
            help="Explicitly acknowledge downtime for drop_recreate",
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

        if not args.allow_downtime:
            raise ValueError(
                "apply requires --allow-downtime for drop_recreate migrations"
            )

        redis_url = create_redis_url(args)
        plan = load_migration_plan(args.plan)
        executor = MigrationExecutor()

        print(f"\nApplying migration to '{plan.source.index_name}'...")

        def progress_callback(step: str, detail: str) -> None:
            step_labels = {
                "drop": "[1/5] Drop index",
                "quantize": "[2/5] Quantize vectors",
                "create": "[3/5] Create index",
                "index": "[4/5] Re-indexing",
                "validate": "[5/5] Validate",
            }
            label = step_labels.get(step, step)
            # Use carriage return to update in place for progress
            if detail and not detail.startswith("done"):
                print(f"  {label}: {detail}    ", end="\r", flush=True)
            else:
                print(f"  {label}: {detail}    ")

        report = executor.apply(
            plan,
            redis_url=redis_url,
            query_check_file=args.query_check_file,
            progress_callback=progress_callback,
        )

        # Print completion summary
        if report.result == "succeeded":
            total_time = report.timings.total_migration_duration_seconds or 0
            downtime = report.timings.downtime_duration_seconds or 0
            print(f"\nMigration completed in {total_time}s (downtime: {downtime}s)")
        else:
            print(f"\nMigration {report.result}")
            # Show errors immediately for visibility
            if report.validation.errors:
                for error in report.validation.errors:
                    print(f"  ERROR: {error}")

        write_migration_report(report, args.report_out)
        if args.benchmark_out:
            write_benchmark_report(report, args.benchmark_out)
        self._print_report_summary(args.report_out, report, args.benchmark_out)

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
        print(
            f"  Apply the migration: rvl migrate apply --plan {plan_out} --allow-downtime"
        )
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
