from pathlib import Path

import pytest

from kube_resource_report.main import get_parser


def test_parse_args_missing_output_dir():
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parse_args(tmpdir):
    parser = get_parser()
    args = parser.parse_args(["--include-clusters=foobar", str(tmpdir)])
    assert args.include_clusters == "foobar"
    assert args.output_dir == Path(tmpdir)
