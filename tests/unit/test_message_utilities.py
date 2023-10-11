from unittest import TestCase

from harmony.message import Message

from harmony_browse_image_generator.message_utility import has_crs


class TestMessageUtility(TestCase):
    def test_message_has_crs(self):
        message = Message({"format": {"crs": "EPSG:4326"}})
        self.assertTrue(has_crs(message))

    def test_message_has_garbage_crs(self):
        message = Message({"format": {"crs": "garbage"}})
        self.assertTrue(has_crs(message))

    def test_message_has_no_crs(self):
        message = Message({})
        self.assertFalse(has_crs(message))
