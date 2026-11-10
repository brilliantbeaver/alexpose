"""Availability classification tests; all network responses are mocked."""

import json
import unittest
from unittest.mock import patch

import check_video_availability as availability


class AvailabilityTests(unittest.TestCase):
    def test_private_login_is_not_publicly_available(self):
        self.assertEqual(availability.classify_player_status({
            "status": "LOGIN_REQUIRED", "reason": "Private video",
            "messages": ["Please sign in to view this private video."]}), availability.UNAVAILABLE)

    def test_ok_is_player_status_evidence(self):
        self.assertEqual(availability.classify_player_status({"status": "OK"}), availability.PLAYER_OK)

    def test_login_age_and_bot_are_restricted_not_unavailable(self):
        for reason in ("Sign in to confirm your age", "Sign in to confirm you're not a bot"):
            with self.subTest(reason=reason):
                self.assertEqual(availability.classify_player_status({
                    "status": "LOGIN_REQUIRED", "reason": reason}), availability.RESTRICTED)

    def test_generic_unavailable_and_missing_status_stay_inconclusive(self):
        for status in ({}, {"status": "ERROR", "reason": "This video is unavailable"}):
            self.assertEqual(availability.classify_player_status(status), availability.UNKNOWN)

    def test_nested_visible_account_termination_is_explicit_evidence(self):
        status = {"status": "ERROR", "reason": "Video unavailable", "errorScreen": {
            "playerInterstitialRenderer": {"content": {"interstitialViewModel": {
                "description": {"content": "The YouTube account associated with this video has been terminated."}}}}}}
        self.assertEqual(availability.classify_player_status(status), availability.UNAVAILABLE)

    def test_tracking_data_does_not_drive_classification(self):
        status = {"status": "ERROR", "reason": "This video is unavailable",
                  "trackingParams": "bot_country_removed", "contextParams": "copyright"}
        self.assertEqual(availability.classify_player_status(status), availability.UNKNOWN)

    def test_geographic_copyright_restriction_is_not_global_unavailability(self):
        self.assertEqual(availability.classify_player_status({"status": "ERROR",
            "reason": "This copyright holder has blocked this video in your country"}), availability.RESTRICTED)

    def check_html(self, html):
        class Response:
            status = 200
            url = "https://www.youtube.com/watch?v=abcdefghijk"
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False
            def read(self, _):
                return html.encode("utf-8")
        result = {"id": "abcdefghijk", "url": Response.url, "status": availability.UNKNOWN}
        with patch.object(availability.urllib.request, "urlopen", return_value=Response()):
            return availability.check_watch_page(result, timeout=1)

    def test_parser_handles_nested_json_and_braces_in_strings(self):
        player = {"playabilityStatus": {"status": "OK"}, "videoDetails": {
            "videoId": "abcdefghijk", "title": "Title with } braces {"}}
        result = self.check_html("<script>var ytInitialPlayerResponse = " + json.dumps(player) + ";</script>")
        self.assertEqual(result["status"], availability.PLAYER_OK)
        self.assertEqual(result["watch_page_evidence"]["playability_status"], {"status": "OK"})

    def test_malformed_or_missing_player_json_stays_inconclusive(self):
        for html in ("<html>Consent required</html>", "var ytInitialPlayerResponse = {bad json}",
                     'var ytInitialPlayerResponse = {"other": 1}'):
            with self.subTest(html=html):
                result = self.check_html(html)
                self.assertEqual(result["status"], availability.UNKNOWN)
                self.assertIn("error", result["watch_page_evidence"])

    def test_mismatched_video_id_stays_inconclusive(self):
        result = self.check_html('var ytInitialPlayerResponse = {"playabilityStatus":{"status":"OK"},'
                                 '"videoDetails":{"videoId":"other_id123"}};')
        self.assertEqual(result["status"], availability.UNKNOWN)

    def test_malformed_video_details_stays_inconclusive(self):
        result = self.check_html('var ytInitialPlayerResponse = {"playabilityStatus":{"status":"OK"},'
                                 '"videoDetails":null};')
        self.assertEqual(result["status"], availability.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
