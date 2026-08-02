"""Range-reader tests with no live Hugging Face dependency."""

from __future__ import annotations

import urllib.error
import unittest
from unittest import mock

from climbmix_mixture.bytesource import HttpRangeReader, SourceFile


class _Response:
    def __init__(self, data: bytes, content_range: str) -> None:
        self._data = data
        self.status = 206
        self.headers = {"Content-Range": content_range}

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._data


class HttpRangeReaderTest(unittest.TestCase):
    def test_clamps_request_at_eof_and_requires_exact_content_range(self) -> None:
        source = SourceFile("part_0.tokenized.jsonl", 10)
        response = _Response(b"hij", "bytes 7-9/10")
        with mock.patch("urllib.request.urlopen", return_value=response) as open_mock:
            data = HttpRangeReader(source, "repo/name", "full-revision").read_range(
                7, 100
            )
        self.assertEqual(data, b"hij")
        request = open_mock.call_args.args[0]
        self.assertEqual(request.get_header("Range"), "bytes=7-9")
        self.assertEqual(request.get_header("Accept-encoding"), "identity")

    def test_retries_a_transient_error(self) -> None:
        source = SourceFile("part_0.tokenized.jsonl", 10)
        response = _Response(b"a", "bytes 0-0/10")
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[urllib.error.URLError("temporary"), response],
        ) as open_mock, mock.patch("climbmix_mixture.bytesource.time.sleep") as sleep_mock:
            data = HttpRangeReader(source, "repo/name", "full-revision").read_range(
                0, 1
            )
        self.assertEqual(data, b"a")
        self.assertEqual(open_mock.call_count, 2)
        sleep_mock.assert_called_once()

    def test_offset_at_eof_needs_no_http_request(self) -> None:
        source = SourceFile("part_0.tokenized.jsonl", 10)
        with mock.patch("urllib.request.urlopen") as open_mock:
            data = HttpRangeReader(source, "repo/name", "full-revision").read_range(
                10, 1
            )
        self.assertEqual(data, b"")
        open_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
