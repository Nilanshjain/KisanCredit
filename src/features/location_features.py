"""Location/Geospatial Features (Part of behavioral & social analysis).

Extracts features from location patterns to measure stability and mobility.
"""

from typing import Dict
import math


class LocationFeatureExtractor:
    """Extract location and geospatial features."""

    def extract_features(self, location_pattern: Dict) -> dict:
        """Extract all location-related features.

        Args:
            location_pattern: Dictionary with location data

        Returns:
            Dictionary of location features
        """
        unique_locs = location_pattern.get('unique_locations_count', 0)
        home_lat = location_pattern.get('home_latitude', 0.0)
        home_lon = location_pattern.get('home_longitude', 0.0)
        area_type = location_pattern.get('area_type', 'Unknown')
        avg_distance = location_pattern.get('average_distance_from_home', 0.0)

        # Calculate travel radius (max distance from home)
        travel_radius = location_pattern.get('max_distance_from_home', avg_distance * 2)

        return {
            # Feature 1: Number of unique locations visited
            'location_unique_count': unique_locs,

            # Feature 2: Location stability score (inverse of unique locations)
            'location_stability_score': self._stability_score(unique_locs),

            # Feature 3: Average distance from home base
            'location_avg_distance_from_home': round(avg_distance, 2),

            # Feature 4: Travel radius (km)
            'location_travel_radius': round(travel_radius, 2),

            # Feature 5: Urban vs Rural classification (encoded)
            'location_area_type_urban': 1 if area_type == 'Urban' else 0,

            # Feature 6: Mobility score (0-100, higher = more mobile)
            'location_mobility_score': self._mobility_score(unique_locs, travel_radius),

            # Feature 7: Geographic diversity
            'location_diversity_score': self._diversity_score(unique_locs, travel_radius),
        }

    def _stability_score(self, unique_locations: int) -> float:
        """Calculate location stability (higher = more stable).

        Stable users tend to stay within 5-10 locations.
        """
        if unique_locations <= 5:
            return 100.0
        elif unique_locations <= 10:
            return 70.0
        elif unique_locations <= 20:
            return 40.0
        elif unique_locations <= 50:
            return 20.0
        else:
            return 0.0

    def _mobility_score(self, unique_locations: int, travel_radius: float) -> float:
        """Calculate mobility score based on locations and travel radius.

        High mobility might indicate:
        - Business person (good)
        - Unstable lifestyle (risky)
        """
        # Normalize unique locations (0-100 scale)
        location_component = min(100, (unique_locations / 50) * 50)

        # Normalize travel radius (0-100 scale, assuming max 500km)
        radius_component = min(100, (travel_radius / 500) * 50)

        mobility = location_component + radius_component
        return round(mobility, 2)

    def _diversity_score(self, unique_locations: int, travel_radius: float) -> float:
        """Geographic diversity using location spread.

        Higher diversity = wider geographic footprint.
        """
        if unique_locations == 0:
            return 0.0

        # Diversity is combination of locations and how spread out they are
        # More locations in larger radius = higher diversity
        diversity = math.sqrt(unique_locations) * math.log(travel_radius + 1)

        # Normalize to 0-100
        normalized = min(100, diversity * 5)
        return round(normalized, 2)
