"""Ranking engine for recommendations."""


class Ranker:
    """Ranks candidate items based on user preferences."""

    def rank(self, candidates: list[dict], user_id: str) -> list[dict]:
        """Rank candidates for a given user.

        Uses a simple scoring heuristic. In production this would
        call a trained ML model.
        """
        scored = []
        for candidate in candidates:
            score = self._compute_score(candidate, user_id)
            scored.append({**candidate, "score": score})

        return sorted(scored, key=lambda x: x["score"], reverse=True)

    def _compute_score(self, candidate: dict, user_id: str) -> float:
        """Compute relevance score for a candidate item."""
        # Placeholder scoring
        return hash(f"{candidate['item_id']}_{user_id}") % 100 / 100.0
