import unittest
from unittest.mock import patch
import app


class CatalogTest(unittest.TestCase):
    def setUp(self):
        app.CACHE.clear()

    def tearDown(self):
        app.CACHE.clear()

    def test_scope_and_profile_exclusion(self):
        self.assertTrue(app.included('solanaclawd/new-release', 'solanaclawd'))
        self.assertTrue(app.included('ordlibrary/hauhau-qwen36-uncensored', 'ordlibrary'))
        self.assertFalse(app.included('ordlibrary/FluxWifMe', 'ordlibrary'))
        self.assertFalse(app.included('solanaclawd/README', 'solanaclawd'))

    def test_failed_refresh_retains_stale_group_without_inventing_counts(self):
        row = dict(id='ordlibrary/clawd-chart-foundation-training', kind='dataset',
                   author='ordlibrary', downloads=None, featured=True)
        app.CACHE.update(fetched_at=0, payload=dict(repositories=[row], stats_as_of='original'))

        def fetch(author, kind):
            if author == 'ordlibrary' and kind == 'datasets':
                raise TimeoutError()
            return []

        with patch.object(app, 'fetch_group', side_effect=fetch):
            result = app.catalog()
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['stats_as_of'], 'original')
        self.assertTrue(result['repositories'][0]['stale'])
        self.assertIsNone(result['repositories'][0]['downloads'])
        self.assertEqual(result['errors'], ['ordlibrary/datasets'])

    def test_cache_and_duplicate_ids(self):
        row = dict(id='solanaclawd/model', kind='model', author='solanaclawd', downloads=10, featured=False)
        with patch.object(app, 'fetch_group', return_value=[row, row]) as fetch:
            self.assertEqual(len(app.catalog()['repositories']), 1)
            app.catalog()
            self.assertEqual(fetch.call_count, 6)


if __name__ == '__main__':
    unittest.main()
