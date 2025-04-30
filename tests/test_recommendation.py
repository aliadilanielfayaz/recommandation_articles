import unittest
import pandas as pd
import numpy as np

# Importe les fonctions à tester depuis leur emplacement correct dans src
# Assurez-vous que cette importation fonctionne lorsque vous lancez les tests depuis la racine
from src.recommendation import generate_recommendations, recommend_based_on_user_attributes

class TestRecommendationFunctions(unittest.TestCase):

    def setUp(self):
        """Configuration initiale exécutée avant chaque test."""
        # Données pour generate_recommendations
        self.user_item_data = {
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
            'article_id': ['A1', 'A2', 'A1', 'A3', 'A2', 'A4', 'A1', 'A4'],
            'prix': [10, 15, 10, 20, 15, 25, 10, 25] # Utilisé pour créer la matrice
        }
        df_matrix = pd.DataFrame(self.user_item_data)
        # Création de la matrice utilisateur-article pour les tests
        self.user_item_matrix = df_matrix.pivot_table(index='user_id', columns='article_id', values='prix', aggfunc='sum', fill_value=0)
        # Résultat attendu de la matrice:
        # article_id  A1  A2  A3  A4
        # user_id
        # 1           10  15   0   0
        # 2           10   0  20   0
        # 3            0  15   0  25
        # 4           10   0   0  25

        # Données pour recommend_based_on_user_attributes
        self.profile_data = {
            'user_id': [1, 1, 5, 5, 6, 7, 7, 8],
            'article_id': ['A1', 'A2', 'A1', 'A3', 'A2', 'A1', 'A4', 'A1'],
            'religion': ['R1', 'R1', 'R1', 'R1', 'R2', 'R1', 'R1', 'R1'],
            'pays': ['P1', 'P1', 'P1', 'P1', 'P1', 'P2', 'P2', 'P1']
        }
        # Création du DataFrame de profils/achats pour les tests
        self.df_profiles = pd.DataFrame(self.profile_data)

    # --- Tests pour generate_recommendations ---

    def test_generate_recommendations_basic(self):
        """Teste le fonctionnement de base de generate_recommendations."""
        # User 1 a acheté A1, A2.
        # User 2 (similaire) a A1, A3.
        # User 4 (similaire) a A1, A4.
        # On s'attend à ce que A3 et A4 soient recommandés à User 1.
        recos = generate_recommendations(self.user_item_matrix, user_id=1, n_recommendations=2)
        self.assertIsInstance(recos, pd.Series)
        self.assertEqual(len(recos), 2)
        # Vérifie que les articles recommandés sont ceux attendus (A3, A4)
        self.assertIn('A3', recos.index)
        self.assertIn('A4', recos.index)

    def test_generate_recommendations_n(self):
        """Teste le nombre de recommandations retournées."""
        recos_1 = generate_recommendations(self.user_item_matrix, user_id=1, n_recommendations=1)
        self.assertEqual(len(recos_1), 1)

        recos_all = generate_recommendations(self.user_item_matrix, user_id=1, n_recommendations=5) # Demande plus que dispo
        # Il y a 4 articles au total. User 1 en a 2 (A1, A2). Il reste 2 articles potentiels (A3, A4).
        self.assertEqual(len(recos_all), 2) # Doit retourner A3 et A4

    def test_generate_recommendations_user_not_found(self):
        """Teste le cas où l'utilisateur n'est pas dans la matrice."""
        recos = generate_recommendations(self.user_item_matrix, user_id=99, n_recommendations=5)
        self.assertIsInstance(recos, pd.Series)
        self.assertTrue(recos.empty)

    # --- Tests pour recommend_based_on_user_attributes ---

    def test_recommend_attributes_match(self):
        """Teste le cas où les profils correspondent et il y a des articles à recommander."""
        # User 1 (R1, P1) a A1, A2. User 5 (R1, P1) a A1, A3.
        # Doit recommander A2 à User 5.
        recos = recommend_based_on_user_attributes(self.df_profiles, user1_id=1, user2_id=5)
        self.assertCountEqual(recos, ['A2']) # Utilise assertCountEqual car l'ordre n'est pas garanti

    def test_recommend_attributes_match_no_new_items(self):
        """Teste le cas où les profils correspondent mais user2 a déjà tout acheté de user1."""
        # User 1 (R1, P1) a A1, A2. User 8 (R1, P1) a A1.
        # Doit recommander A2 à User 8.
        recos = recommend_based_on_user_attributes(self.df_profiles, user1_id=1, user2_id=8)
        self.assertCountEqual(recos, ['A2'])

        # Test inverse : recommander pour user 1 basé sur user 8 (qui n'a que A1)
        # User 1 a déjà A1, donc aucune recommandation.
        recos_inv = recommend_based_on_user_attributes(self.df_profiles, user1_id=8, user2_id=1)
        self.assertCountEqual(recos_inv, [])

    def test_recommend_attributes_religion_mismatch(self):
        """Teste le cas où la religion ne correspond pas."""
        # User 1 (R1, P1) et User 6 (R2, P1)
        recos = recommend_based_on_user_attributes(self.df_profiles, user1_id=1, user2_id=6)
        self.assertCountEqual(recos, [])

    def test_recommend_attributes_country_mismatch(self):
        """Teste le cas où le pays ne correspond pas."""
        # User 1 (R1, P1) et User 7 (R1, P2)
        recos = recommend_based_on_user_attributes(self.df_profiles, user1_id=1, user2_id=7)
        self.assertCountEqual(recos, [])

    def test_recommend_attributes_user_not_found(self):
        """Teste le cas où un des utilisateurs n'existe pas dans le DataFrame."""
        recos = recommend_based_on_user_attributes(self.df_profiles, user1_id=1, user2_id=99) # User 99 n'existe pas
        self.assertCountEqual(recos, [])
        recos2 = recommend_based_on_user_attributes(self.df_profiles, user1_id=99, user2_id=1) # User 99 n'existe pas
        self.assertCountEqual(recos2, [])

# Permet d'exécuter les tests si le fichier est lancé directement
if __name__ == '__main__':
    unittest.main()
