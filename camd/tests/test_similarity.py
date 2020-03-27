import unittest
import os
from camd.similarity import FunctionalSimilarity
from camd.utils.data import load_dataframe

curated_scintillators = [('CdWO4', 5488), ('CaWO4', 3153), ('Gd2SiO5', 5382),
                                 ('Gd2SO2', 30396), ('LaBr3', 6038), ('CsI', 8355)]
curated_ids = [i[1] for i in curated_scintillators]


class TestFunctionalSimilarity(unittest.TestCase):
    def test_similarity(self):
        df = load_dataframe('oqmd1.2_exp_based_entries_featurized_v2')
        fn = FunctionalSimilarity(df, curated_ids)

        result = fn.get_df_of_similar(metric='euclidean')
        self.assertEqual(result['Composition'].iloc[0],'CsI')

        result = fn.get_df_of_similar(metric='euclidean', ignore_elements=['Cs'], include_elements=['Zr'])
        self.assertEqual(result['Composition'].iloc[0],'Gd2Zr2O7')

        fn = FunctionalSimilarity(df, curated_ids, pca=5)
        result = fn.get_df_of_similar(metric='cosine')
        self.assertEqual(result['Composition'].iloc[0],'DyVO3')

        best_metric = fn.autofind_best_metric(n_splits=2)
        self.assertIn(best_metric, fn._metrics_allowed)

        fn.plot_auto_ranks()
        self.assertTrue(True)