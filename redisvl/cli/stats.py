import argparse
import sys
from argparse import Namespace

from redisvl.cli.utils import add_index_parsing_options, create_redis_url
from redisvl.index import SearchIndex
from redisvl.schema.schema import IndexSchema
from redisvl.utils.log import get_logger
from redisvl.utils.utils import lazy_import

logger = get_logger("[RedisVL]")

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

        parser.add_argument(
            "-f", "--format", help="Output format", type=str, default="rounded_outline"
        )
        parser = add_index_parsing_options(parser)
        args = parser.parse_args(sys.argv[2:])
        try:
            self.stats(args)
        except Exception as e:
            logger.error(e)
            exit(0)

    def stats(self, args: Namespace):
        """Obtain stats about an index.

        Usage:
            rvl stats -i <index_name> | -s <schema_path>
        """
        index = self._connect_to_index(args)
        _display_stats(index.info(), output_format=args.format)

    def _connect_to_index(self, args: Namespace) -> SearchIndex:
        # connect to redis
        try:
            redis_url = create_redis_url(args)
        except ValueError:
            logger.error(
                "Must set REDIS_ADDRESS environment variable or provide host and port"
            )
            exit(0)

        if args.index:
            schema = IndexSchema.from_dict({"index": {"name": args.index}})
            index = SearchIndex(schema=schema, redis_url=redis_url)
        elif args.schema:
            index = SearchIndex.from_yaml(args.schema, redis_url=redis_url)
        else:
            logger.error("Index name or schema must be provided")
            exit(0)

        return index


def _display_stats(index_info, output_format="rounded_outline"):
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
