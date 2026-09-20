"""Recommendation service."""

from recommendation.ranker import Ranker


class RecommendationService:
    """Generates personalized recommendations for users."""

    def __init__(self):
        self.ranker = Ranker()

    def get_recommendations(self, user_id: str, limit: int = 10) -> list[dict]:
        """Get top-K recommendations for a user.

        1. Retrieve candidate items
        2. Rank candidates
        3. Return top results
        """
        candidates = self._get_candidates(user_id)
        ranked = self.ranker.rank(candidates, user_id)
        return ranked[:limit]

    def _get_candidates(self, user_id: str) -> list[dict]:
        """Retrieve candidate items for recommendation."""
        # Placeholder: would query a database or cache
        return [
            {"item_id": f"item_{i}", "score": 0.0}
            for i in range(100)
        ]
