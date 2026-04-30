import argparse
import sys
from argparse import Namespace

import yaml
from pydantic import ValidationError

from redisvl.cli.utils import add_index_parsing_options, create_redis_url
from redisvl.exceptions import RedisSearchError
from redisvl.index import SearchIndex
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.redis.utils import convert_bytes, make_dict
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


class Index:
    description = (
        "Create, inspect, list, and delete Redis search indexes.\n\n"
        "Use `-i/--index` to target an existing Redis index name or "
        "`-s/--schema` to load a schema YAML file. Shared Redis connection "
        "options apply to these data-plane commands."
    )
    epilog = "\n".join(
        [
            "Examples:",
            "  rvl index create -s schema.yaml",
            "  rvl index info -i user_index",
            "  rvl index listall --url redis://localhost:6379",
        ]
    )

    def __init__(self):
        parser = self._build_parser()

        args = parser.parse_args(sys.argv[2:])

        if not hasattr(self, args.command):
            print(f"Unknown command: {args.command}\n", file=sys.stderr)
            parser.print_help(sys.stderr)
            sys.exit(2)

        try:
            args.handler(args)
        except Exception as e:
            logger.error(e, exc_info=True)
            print(str(e), file=sys.stderr)
            sys.exit(1)

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="rvl index",
            description=self.description,
            epilog=self.epilog,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        shared_options = argparse.ArgumentParser(add_help=False)
        add_index_parsing_options(shared_options)

        subparsers = parser.add_subparsers(dest="command", title="Commands")

        create_parser = subparsers.add_parser(
            "create",
            parents=[shared_options],
            help="Create a new index from a schema file",
            description="Create a new Redis search index from a schema YAML file.",
            epilog="\n".join(
                [
                    "Examples:",
                    "  rvl index create -s schema.yaml",
                    "  rvl index create -s schema.yaml --url redis://localhost:6379",
                ]
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        create_parser.set_defaults(handler=self.create)

        info_parser = subparsers.add_parser(
            "info",
            parents=[shared_options],
            help="Show details about an index",
            description="Display schema and storage details for an index.",
            epilog="\n".join(
                [
                    "Examples:",
                    "  rvl index info -i user_index",
                    "  rvl index info -s schema.yaml",
                ]
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        info_parser.set_defaults(handler=self.info)

        listall_parser = subparsers.add_parser(
            "listall",
            parents=[shared_options],
            help="List indexes available on the target Redis deployment",
            description="List all Redis search indexes available on the target Redis deployment.",
            epilog="\n".join(
                [
                    "Examples:",
                    "  rvl index listall",
                    "  rvl index listall --host localhost --port 6379",
                ]
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        listall_parser.set_defaults(handler=self.listall)

        delete_parser = subparsers.add_parser(
            "delete",
            parents=[shared_options],
            help="Delete an index but leave its data in Redis",
            description="Delete an existing Redis search index without dropping indexed data.",
            epilog="\n".join(
                [
                    "Examples:",
                    "  rvl index delete -i user_index",
                    "  rvl index delete -s schema.yaml",
                ]
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        delete_parser.set_defaults(handler=self.delete)

        destroy_parser = subparsers.add_parser(
            "destroy",
            parents=[shared_options],
            help="Delete an index and drop its indexed data",
            description="Delete an existing Redis search index and drop its indexed data.",
            epilog="\n".join(
                [
                    "Examples:",
                    "  rvl index destroy -i user_index",
                    "  rvl index destroy -s schema.yaml",
                ]
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        destroy_parser.set_defaults(handler=self.destroy)

        return parser

    def create(self, args: Namespace):
        """Create an index.

        Usage:
            rvl index create -s <schema_path>
        """
        if not args.schema:
            print("Schema must be provided to create an index", file=sys.stderr)
            sys.exit(2)

        redis_url = create_redis_url(args)
        try:
            index = SearchIndex.from_yaml(args.schema, redis_url=redis_url)
        except SCHEMA_INPUT_ERRORS as e:
            exit_schema_input_error(args, e)
        try:
            index.create()
        except RedisSearchError as e:
            exit_redis_search_error(args, index, e)
        print("Index created successfully")

    def info(self, args: Namespace):
        """Obtain information about an index.

        Usage:
            rvl index info -i <index_name> | -s <schema_path>
        """
        index = self._connect_to_index(args)
        try:
            _display_in_table(index.info())
        except RedisSearchError as e:
            exit_redis_search_error(args, index, e)

    def listall(self, args: Namespace):
        """List all indices.

        Usage:
            rvl index listall
        """
        redis_url = create_redis_url(args)
        conn = RedisConnectionFactory.get_redis_connection(redis_url=redis_url)
        indices = convert_bytes(conn.execute_command("FT._LIST"))
        print("Indices:")
        for i, index in enumerate(indices):
            print(str(i + 1) + ". " + index)

    def delete(self, args: Namespace, drop=False):
        """Delete an index.

        Usage:
            rvl index delete -i <index_name> | -s <schema_path>
        """
        index = self._connect_to_index(args)
        try:
            index.delete(drop=drop)
        except RedisSearchError as e:
            exit_redis_search_error(args, index, e)
        print("Index deleted successfully")

    def destroy(self, args: Namespace):
        """Delete an index and the documents within it.

        Usage:
            rvl index destroy -i <index_name> | -s <schema_path>
        """
        self.delete(args, drop=True)

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


def _display_in_table(index_info):
    print("\n")
    attributes = index_info.get("attributes", [])
    definition = make_dict(index_info.get("index_definition"))
    index_info = [
        index_info.get("index_name"),
        definition.get("key_type"),
        definition.get("prefixes"),
        index_info.get("index_options"),
        index_info.get("indexing"),
    ]

    # Display the index information in tabular format
    headers = ["Index Name", "Storage Type", "Prefixes", "Index Options", "Indexing"]
    col_width = max(len(str(info)) for info in index_info + headers) + 2

    def print_table_edge(length, col_width, start, mid, stop):
        print(f"{start}", end="")
        for _ in range(length):
            print("─" * col_width, mid, sep="", end="")
        print(f"\b{stop}")

    print("Index Information:")

    print_table_edge(len(index_info), col_width, "╭", "┬", "╮")

    # print header row
    for header in headers:
        print(f"│ {header.ljust(col_width-2)} ", end="")
    print("│")

    print_table_edge(len(index_info), col_width, "├", "┼", "┤")

    # print data row
    for info in index_info:
        print(f"| {str(info).ljust(col_width-2)} ", end="")
    print("|")

    print_table_edge(len(index_info), col_width, "╰", "┴", "╯")

    attr_values = []
    headers = [
        "Name",
        "Attribute",
        "Type",
    ]

    for attrs in attributes:
        attr = make_dict(attrs)

        values = [attr.get("identifier"), attr.get("attribute"), attr.get("type")]
        if len(attrs) > 5:
            options = make_dict(attrs)
            for k, v in options.items():
                if k not in ["identifier", "attribute", "type"]:
                    headers.append("Field Option")
                    headers.append("Option Value")
                    values.append(k)
                    values.append(v)
        attr_values.append(values)

    # Display the attributes in tabular format
    print("Index Fields:")
    headers = headers[
        : max(len(row) for row in attr_values)
    ]  # remove extra headers with no attr values
    col_widths = [max([len(str(attr)) + 2 for attr in row]) for row in attr_values]
    print_table_edge(len(headers), max(col_widths), "╭", "┬", "╮")

    # print header row
    for header in headers:
        print(f"│ {str(header).ljust(max(col_widths)-2)} ", end="")
    print("│")

    print_table_edge(len(headers), max(col_widths), "├", "┼", "┤")

    # print data rows
    num_cols = max(len(row) for row in attr_values)
    for row in attr_values:
        row.extend([""] * (num_cols - len(row)))
        for attr in row:
            print(f"│ {str(attr).ljust(max(col_widths)-2)} ", end="")
        print("│")

    print_table_edge(len(headers), max(col_widths), "╰", "┴", "╯")
