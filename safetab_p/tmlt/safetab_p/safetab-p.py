#!/usr/bin/env python3
"""Command line interface for SafeTab-P."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Tumult Labs
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import argparse
import json
import os
import sys
import tempfile

from smart_open import open  # pylint: disable=redefined-builtin

from tmlt.common.io_helpers import get_logger_stream, write_log_file
from tmlt.safetab_p.paths import ALT_INPUT_CONFIG_DIR_SAFETAB_P, setup_input_config_dir
from tmlt.safetab_p.safetab_p_analytics import run_plan_p_analytics
from tmlt.safetab_utils.input_validation import validate_input
from tmlt.safetab_utils.regions import validate_state_filter_us
from tmlt.safetab_utils.utils import (
    READER_FLAG,
    STATE_FILTER_FLAG,
    safetab_input_reader,
)


def main():
    """Parse command line arguments and run SafeTab-P."""
    parser = argparse.ArgumentParser(prog="safetab-p")
    subparsers = parser.add_subparsers(help="safetab-p sub-commands", dest="mode")

    def add_parameters_path(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "parameters_directory",
            help=(
                "The directory that contains: config.json,"
                " ethnicity-characteristic-iterations.txt,"
                " race-and-ethnicity-code-to-iterations.txt,"
                " race-and-ethnicity-codes.txt, and race-characteristic-iterations.txt."
            ),
            type=str,
        )

    def add_data_path(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "data_path",
            help=(
                "If using a CEF reader, this is the location of the reader config "
                "file. If using a CSV reader, this is a directory that contains: "
                "GRF-C.txt and person-records.txt."
            ),
            type=str,
        )

    def add_output(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            dest="output_directory",
            help=(
                "The directory where output files will be written. Any output from "
                "previous runs will be overwritten."
            ),
            type=str,
        )

    def add_log(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "-l",
            "--log",
            dest="log_filename",
            help=(
                "The file that logs will be written to. If the file already exists, "
                "it will be overwritten."
            ),
            type=str,
            default="safetab_p.log",
        )

    def add_validate_private_output(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "-vo",
            "--validate-private-output",
            dest="output_validation_flag",
            help=(
                "If this flag is set, the SafeTab-P will perform validation on its "
                "own output."
            ),
            action="store_true",
            default=False,
        )

    add_log(parser)

    parser_validate = subparsers.add_parser(
        "validate",
        help="Validate mode checks the input files, but does not execute any queries.",
    )
    for add_arg_func in [add_parameters_path, add_data_path, add_log]:
        add_arg_func(parser_validate)

    parser_execute = subparsers.add_parser(
        "execute",
        help=(
            "Execute mode validates the input files, executes queries, and writes "
            "output."
        ),
    )
    for add_arg_func in [
        add_parameters_path,
        add_data_path,
        add_output,
        add_log,
        add_validate_private_output,
    ]:
        add_arg_func(parser_execute)

    args = parser.parse_args()

    # Set up logging.
    logger, io_stream = get_logger_stream()

    if not args.mode:
        logger.error("No mode was specified. Exiting...")
        sys.exit(1)

    if args.mode == "validate":
        logger.info("Validating SafeTab-P inputs and config...")
        setup_input_config_dir()
        with tempfile.TemporaryDirectory() as updated_config_dir:
            with open(os.path.join(args.parameters_directory, "config.json"), "r") as f:
                config_json = json.load(f)
                reader = config_json[READER_FLAG]
                state_filter = []
                if config_json["run_us"] and validate_state_filter_us(
                    config_json[STATE_FILTER_FLAG]
                ):
                    state_filter += config_json[STATE_FILTER_FLAG]
                if config_json["run_pr"]:
                    state_filter += ["72"]

            okay = validate_input(
                parameters_path=args.parameters_directory,
                input_data_configs_path=ALT_INPUT_CONFIG_DIR_SAFETAB_P,
                output_path=updated_config_dir,
                program="safetab-p",
                input_reader=safetab_input_reader(
                    reader=reader,
                    data_path=args.data_path,
                    state_filter=state_filter,
                    program="safetab-p",
                ),
                state_filter=state_filter,
            )
        if not okay:
            logger.error("SafeTab-P input validation failed. Exiting...")
            sys.exit(1)

    if args.mode == "execute":
        logger.info("Running SafeTab-P in 'execute' mode...")
        run_plan_p_analytics(
            parameters_path=args.parameters_directory,
            data_path=args.data_path,
            output_path=args.output_directory,
            should_validate_private_output=args.output_validation_flag,
        )

    if args.log_filename:
        log_content = io_stream.getvalue()
        io_stream.close()
        write_log_file(args.log_filename, log_content)


if __name__ == "__main__":
    main()
