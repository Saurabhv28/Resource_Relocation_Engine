"""Tests for allocation algorithms."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models import Resource, Request, Location, TruckSize
from algorithms import greedy_allocation, hungarian_allocation, ml_allocation, haversine_km, can_fulfill


def make_resource(id, lat, lng, size=TruckSize.MEDIUM, available=True):
    caps = {TruckSize.SMALL: 500, TruckSize.MEDIUM: 1500, TruckSize.LARGE: 3000}
    return Resource(id=id, name=f"Truck {id}", location=Location(lat=lat, lng=lng),
                    size=size, capacity_kg=caps[size], available=available)


def make_request(id, lat, lng, weight=100, priority=1):
    return Request(id=id, destination=Location(lat=lat, lng=lng),
                   weight_kg=weight, priority=priority)


class TestHaversine:
    def test_same_point(self):
        assert haversine_km(12.97, 77.59, 12.97, 77.59) == 0.0

    def test_known_distance(self):
        # Bangalore to Chennai ~290 km
        dist = haversine_km(12.97, 77.59, 13.08, 80.27)
        assert 280 < dist < 310


class TestConstraints:
    def test_capacity_fulfilled(self):
        res = make_resource("t1", 12.97, 77.59, TruckSize.MEDIUM)
        req = make_request("o1", 13.0, 77.6, weight=1000)
        assert can_fulfill(res, req) is True

    def test_capacity_exceeded(self):
        res = make_resource("t1", 12.97, 77.59, TruckSize.SMALL)
        req = make_request("o1", 13.0, 77.6, weight=600)  # > 500 kg
        assert can_fulfill(res, req) is False

    def test_unavailable_resource(self):
        res = make_resource("t1", 12.97, 77.59, available=False)
        req = make_request("o1", 13.0, 77.6, weight=100)
        assert can_fulfill(res, req) is False


class TestGreedyAllocation:
    def test_basic_assignment(self):
        resources = [make_resource("t1", 12.97, 77.59)]
        requests = [make_request("o1", 12.98, 77.60)]
        result = greedy_allocation(resources, requests)
        assert len(result.assignments) == 1
        assert result.assignments[0].resource_id == "t1"
        assert result.unassigned_requests == []

    def test_no_resources(self):
        result = greedy_allocation([], [make_request("o1", 12.98, 77.60)])
        assert len(result.assignments) == 0
        assert result.unassigned_requests == ["o1"]

    def test_picks_nearest(self):
        resources = [
            make_resource("far", 13.5, 77.59),
            make_resource("near", 12.971, 77.591),
        ]
        requests = [make_request("o1", 12.97, 77.59)]
        result = greedy_allocation(resources, requests)
        assert result.assignments[0].resource_id == "near"

    def test_priority_ordering(self):
        resources = [make_resource("t1", 12.97, 77.59)]
        requests = [
            make_request("low", 12.98, 77.60, priority=1),
            make_request("high", 13.05, 77.65, priority=3),
        ]
        result = greedy_allocation(resources, requests)
        # High priority request gets assigned first (only 1 truck)
        assert result.assignments[0].request_id == "high"
        assert "low" in result.unassigned_requests


class TestHungarianAllocation:
    def test_basic_assignment(self):
        resources = [make_resource("t1", 12.97, 77.59)]
        requests = [make_request("o1", 12.98, 77.60)]
        result = hungarian_allocation(resources, requests)
        assert len(result.assignments) == 1
        assert result.unassigned_requests == []

    def test_optimal_over_greedy(self):
        """Hungarian should find globally better assignment than greedy in this scenario."""
        # Two trucks, two orders arranged so greedy sub-optimal
        resources = [
            make_resource("t1", 12.90, 77.50),  # Southwest
            make_resource("t2", 13.10, 77.70),  # Northeast
        ]
        requests = [
            make_request("o1", 12.92, 77.52, priority=3),  # Near t1
            make_request("o2", 13.08, 77.68, priority=1),  # Near t2
        ]
        greedy_result = greedy_allocation(resources, requests)
        hungarian_result = hungarian_allocation(resources, requests)
        # Hungarian should achieve <= total distance of greedy
        assert hungarian_result.total_distance_km <= greedy_result.total_distance_km + 0.1

    def test_respects_capacity_constraint(self):
        resources = [make_resource("t1", 12.97, 77.59, TruckSize.SMALL)]
        requests = [make_request("o1", 12.98, 77.60, weight=600)]  # Exceeds 500kg
        result = hungarian_allocation(resources, requests)
        assert len(result.assignments) == 0
        assert "o1" in result.unassigned_requests


class TestAlgorithmComparison:
    def test_both_produce_valid_results(self):
        resources = [make_resource(f"t{i}", 12.97 + i*0.01, 77.59) for i in range(5)]
        requests = [make_request(f"o{i}", 12.98 + i*0.01, 77.60) for i in range(5)]

        greedy_result = greedy_allocation(resources, requests)
        hungarian_result = hungarian_allocation(resources, requests)

        # Both should assign all requests (enough resources)
        assert greedy_result.assignment_rate == 100.0
        assert hungarian_result.assignment_rate == 100.0

        # No duplicate assignments
        greedy_res_ids = [a.resource_id for a in greedy_result.assignments]
        hungarian_res_ids = [a.resource_id for a in hungarian_result.assignments]
        assert len(set(greedy_res_ids)) == len(greedy_res_ids)
        assert len(set(hungarian_res_ids)) == len(hungarian_res_ids)

    def test_metrics_reported(self):
        resources = [make_resource("t1", 12.97, 77.59)]
        requests = [make_request("o1", 12.98, 77.60)]
        result = greedy_allocation(resources, requests)
        assert result.computation_time_ms >= 0
        assert result.total_distance_km > 0
        assert result.avg_distance_km > 0


class TestMLAllocation:
    def test_basic_assignment(self):
        resources = [make_resource("t1", 12.97, 77.59)]
        requests = [make_request("o1", 12.98, 77.60)]
        result = ml_allocation(resources, requests)
        assert len(result.assignments) == 1
        assert result.assignments[0].resource_id == "t1"
        assert result.assignments[0].request_id == "o1"
        assert result.unassigned_requests == []
        assert result.algorithm == "ML-Based (Learned Patterns)"

    def test_no_resources(self):
        result = ml_allocation([], [make_request("o1", 12.98, 77.60)])
        assert len(result.assignments) == 0
        assert result.unassigned_requests == ["o1"]

    def test_no_requests(self):
        result = ml_allocation([make_resource("t1", 12.97, 77.59)], [])
        assert len(result.assignments) == 0

    def test_respects_capacity_constraint(self):
        resources = [make_resource("t1", 12.97, 77.59, TruckSize.SMALL)]
        requests = [make_request("o1", 12.98, 77.60, weight=600)]  # Exceeds 500kg
        result = ml_allocation(resources, requests)
        assert len(result.assignments) == 0
        assert "o1" in result.unassigned_requests

    def test_respects_availability_constraint(self):
        resources = [make_resource("t1", 12.97, 77.59, available=False)]
        requests = [make_request("o1", 12.98, 77.60)]
        result = ml_allocation(resources, requests)
        assert len(result.assignments) == 0

    def test_multiple_assignments_no_duplicates(self):
        resources = [make_resource(f"t{i}", 12.97 + i*0.01, 77.59) for i in range(5)]
        requests = [make_request(f"o{i}", 12.98 + i*0.01, 77.60) for i in range(5)]
        result = ml_allocation(resources, requests)
        # All should be assigned
        assert result.assignment_rate == 100.0
        # No duplicate resource or request assignments
        res_ids = [a.resource_id for a in result.assignments]
        req_ids = [a.request_id for a in result.assignments]
        assert len(set(res_ids)) == len(res_ids)
        assert len(set(req_ids)) == len(req_ids)

    def test_provides_explanation(self):
        resources = [make_resource("t1", 12.97, 77.59)]
        requests = [make_request("o1", 12.98, 77.60)]
        result = ml_allocation(resources, requests)
        explanation = result.assignments[0].explanation
        assert "ML model" in explanation
        assert "confidence score" in explanation
        assert "Distance" in explanation

    def test_competitive_with_other_algorithms(self):
        """ML should produce reasonable results compared to greedy and Hungarian."""
        resources = [make_resource(f"t{i}", 12.97 + i*0.01, 77.59) for i in range(6)]
        requests = [make_request(f"o{i}", 12.98 + i*0.005, 77.60, priority=(i % 3) + 1) for i in range(6)]

        greedy_result = greedy_allocation(resources, requests)
        hungarian_result = hungarian_allocation(resources, requests)
        ml_result = ml_allocation(resources, requests)

        # ML should achieve same assignment rate as others (all feasible)
        assert ml_result.assignment_rate == 100.0
        # ML total distance should be within reasonable range (not wildly worse)
        assert ml_result.total_distance_km < greedy_result.total_distance_km * 2


class TestThreeAlgorithmComparison:
    def test_all_three_produce_valid_results(self):
        resources = [make_resource(f"t{i}", 12.97 + i*0.01, 77.59) for i in range(5)]
        requests = [make_request(f"o{i}", 12.98 + i*0.01, 77.60) for i in range(5)]

        greedy_result = greedy_allocation(resources, requests)
        hungarian_result = hungarian_allocation(resources, requests)
        ml_result = ml_allocation(resources, requests)

        for result in [greedy_result, hungarian_result, ml_result]:
            assert result.assignment_rate == 100.0
            assert result.computation_time_ms >= 0
            assert result.total_distance_km > 0

    def test_all_report_metrics(self):
        resources = [make_resource(f"t{i}", 12.97 + i*0.02, 77.59) for i in range(3)]
        requests = [make_request(f"o{i}", 12.98 + i*0.02, 77.60) for i in range(3)]

        for algo_fn in [greedy_allocation, hungarian_allocation, ml_allocation]:
            result = algo_fn(resources, requests)
            assert hasattr(result, "total_distance_km")
            assert hasattr(result, "avg_distance_km")
            assert hasattr(result, "assignment_rate")
            assert hasattr(result, "computation_time_ms")
            assert result.algorithm != ""


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
