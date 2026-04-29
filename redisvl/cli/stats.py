import argparse
import sys
from argparse import Namespace

import yaml
from pydantic import ValidationError

from redisvl.cli.utils import add_index_parsing_options, create_redis_url
from redisvl.exceptions import RedisSearchError
from redisvl.index import SearchIndex
from redisvl.schema.schema import IndexSchema
from redisvl.utils.log import get_logger

logger = get_logger("[RedisVL]")

# Exceptions commonly raised when loading or validating a schema path (-s).
SCHEMA_INPUT_ERRORS = (
    FileNotFoundError,
    ValueError,
    yaml.YAMLError,
    ValidationError,
)


def exit_schema_input_error(args: Namespace, exc: BaseException) -> None:
    if not args.schema:
        raise exc
    print(str(exc), file=sys.stderr)
    sys.exit(2)


def exit_redis_search_error(
    args: Namespace, index: SearchIndex | None, exc: RedisSearchError
) -> None:
    name = (
        index.schema.index.name
        if index is not None
        else (args.index or args.schema or "unknown")
    )
    print(
        f"Redis search operation failed for index {name!r}. {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

STATS_KEYS = [
    "num_docs",
    "num_terms",
    "max_doc_id",
    "num_records",
    "percent_indexed",
    "hash_indexing_failures",
    "number_of_uses",
    "bytes_per_record_avg",
    "doc_table_size_mb",
    "inverted_sz_mb",
    "key_table_size_mb",
    "offset_bits_per_record_avg",
    "offset_vectors_sz_mb",
    "offsets_per_term_avg",
    "records_per_doc_avg",
    "sortable_values_size_mb",
    "total_indexing_time",
    "total_inverted_index_blocks",
    "vector_index_sz_mb",
]


class Stats:
    usage = "\n".join(
        [
            "rvl stats [<args>]\n",
        ]
    )

    def __init__(self):
        parser = argparse.ArgumentParser(usage=self.usage)
        parser = add_index_parsing_options(parser)
        args = parser.parse_args(sys.argv[2:])

        try:
            self.stats(args)
        except Exception as e:
            logger.error(e, exc_info=True)
            print(str(e), file=sys.stderr)
            sys.exit(1)

    def stats(self, args: Namespace):
        """Obtain stats about an index.

        Usage:
            rvl stats -i <index_name> | -s <schema_path>
        """
        index = self._connect_to_index(args)
        try:
            _display_stats(index.info())
        except RedisSearchError as e:
            exit_redis_search_error(args, index, e)

    def _connect_to_index(self, args: Namespace) -> SearchIndex:
        redis_url = create_redis_url(args)

        if args.index:
            try:
                schema = IndexSchema.from_dict({"index": {"name": args.index}})
                return SearchIndex(schema=schema, redis_url=redis_url)
            except SCHEMA_INPUT_ERRORS as e:
                exit_schema_input_error(args, e)

        if args.schema:
            try:
                return SearchIndex.from_yaml(args.schema, redis_url=redis_url)
            except SCHEMA_INPUT_ERRORS as e:
                exit_schema_input_error(args, e)

        print("Index name or schema must be provided", file=sys.stderr)
        sys.exit(2)


def _display_stats(index_info):
    # Extracting the statistics
    stats_data = [(key, str(index_info.get(key))) for key in STATS_KEYS]

    # Display the statistics in tabular format
    print("\nStatistics:")
    max_key_length = max(len(key) for key, _ in stats_data)
    horizontal_line = "─" * (max_key_length + 2)
    print(f"╭{horizontal_line}┬────────────╮")  # top row
    print("│ Stat Key                    │ Value      │")  # header row
    print(f"├{horizontal_line}┼────────────┤")  # separator row
    for key, value in stats_data:
        print(f"│ {key:<27} │ {value[0:10]:<10} │")  # data rows
    print(f"╰{horizontal_line}┴────────────╯")  # bottom row
