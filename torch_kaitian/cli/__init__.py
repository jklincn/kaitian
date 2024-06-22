import argparse

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
        "FILE", help="Your training code, for example: python run.py train.py"
    )
    parser_run.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Enable quiet mode, less output is printed",
    )
    parser_run.add_argument(
        "-d",
        "--develop",
        action="store_true",
        help="Enable development mode",
    )

    known_args, unknown_args = parser.parse_known_args()
    if known_args.command == "init":
        init_kaitian(known_args, unknown_args)
    elif known_args.command == "run":
        run_kaitian(known_args, unknown_args)
