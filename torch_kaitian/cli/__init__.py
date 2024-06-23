import argparse
import os

from .init import init_kaitian
from .run import run_kaitian


class CustomHelpFormatter(argparse.HelpFormatter):
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix == "positional arguments:":
            prefix = "command:"
        return super()._format_usage(usage, actions, groups, prefix)

    def start_section(self, heading):
        if heading == "positional arguments":
            heading = "command"
        super().start_section(heading)


def main():
    parser = argparse.ArgumentParser(
        description="KaiTian Launcher",
        usage="kaitian [-h] command ...",
        formatter_class=CustomHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="", help="")

    parser_init = subparsers.add_parser("init", help="Initialize KaiTian environment")
    parser_init.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Reinitialization KaiTian environment",
    )
    parser_run = subparsers.add_parser("run", help="Run the training code")
    parser_run.add_argument(
        "-f",
        "--file",
        default=None,
        help="Your training code",
    )
    parser_run.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Enable quiet mode, less output is printed",
    )
    parser_run.add_argument(
        "-d", "--develop", action="append", help="Development mode arguments"
    )
    known_args, unknown_args = parser.parse_known_args()
    if known_args.command == "init":
        init_kaitian(known_args, unknown_args)
    elif known_args.command == "run":
        # argument check
        if known_args.develop is None:
            if known_args.file is None:
                exit(f"[KaiTian][Error] '-f FILE' argument is required.")
            else:
                file = known_args.file
                if not os.path.exists(file):
                    exit(f"[KaiTian][Error] {file} not found.")
        run_kaitian(known_args, unknown_args)
