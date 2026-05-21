"""Allocation algorithms: Greedy, Hungarian, and ML-Based."""

import time
import math
import random
from models import Resource, Request, Assignment, AllocationResult, Location, TruckSize, TRUCK_CAPACITIES
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance between two points in km using Haversine formula."""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def compute_cost(resource: Resource, request: Request) -> float:
    """Compute cost of assigning a resource to a request.
    
    Combines distance with priority weighting.
    Lower cost = better assignment.
    """
    distance = haversine_km(
        resource.location.lat, resource.location.lng,
        request.destination.lat, request.destination.lng
    )
    # Higher priority requests get lower cost (priority 3 = most urgent)
    priority_factor = 1.0 / request.priority
    return distance * priority_factor


def can_fulfill(resource: Resource, request: Request) -> bool:
    """Check hard constraints: availability and capacity."""
    return resource.available and resource.capacity_kg >= request.weight_kg


def greedy_allocation(resources: list[Resource], requests: list[Request]) -> AllocationResult:
    """Greedy algorithm: process requests by priority, assign nearest valid resource."""
    start = time.time()

    # Sort requests by priority (highest first)
    sorted_requests = sorted(requests, key=lambda r: r.priority, reverse=True)
    available = {r.id: r for r in resources if r.available}
    assignments = []
    unassigned = []

    for req in sorted_requests:
        best_resource_id = None
        best_cost = float("inf")

        for res_id, res in available.items():
            if not can_fulfill(res, req):
                continue
            cost = compute_cost(res, req)
            if cost < best_cost:
                best_cost = cost
                best_resource_id = res_id

        if best_resource_id is not None:
            res = available.pop(best_resource_id)
            distance = haversine_km(
                res.location.lat, res.location.lng,
                req.destination.lat, req.destination.lng
            )
            assignments.append(Assignment(
                resource_id=res.id,
                request_id=req.id,
                distance_km=round(distance, 2),
                explanation=f"Nearest available truck ({res.name}, {res.size.value}) "
                            f"at {distance:.1f} km away. Priority: {req.priority}."
            ))
        else:
            unassigned.append(req.id)

    elapsed_ms = (time.time() - start) * 1000
    total_dist = sum(a.distance_km for a in assignments)
    avg_dist = total_dist / len(assignments) if assignments else 0

    return AllocationResult(
        algorithm="Greedy (Nearest-First)",
        assignments=assignments,
        unassigned_requests=unassigned,
        total_distance_km=round(total_dist, 2),
        avg_distance_km=round(avg_dist, 2),
        assignment_rate=round(len(assignments) / len(requests) * 100, 1) if requests else 0,
        computation_time_ms=round(elapsed_ms, 2)
    )


def hungarian_allocation(resources: list[Resource], requests: list[Request]) -> AllocationResult:
    """Hungarian (Kuhn-Munkres) algorithm: optimal batch assignment minimizing total cost."""
    start = time.time()

    available_resources = [r for r in resources if r.available]
    n_res = len(available_resources)
    n_req = len(requests)

    if n_res == 0 or n_req == 0:
        elapsed_ms = (time.time() - start) * 1000
        return AllocationResult(
            algorithm="Hungarian (Optimal Batch)",
            assignments=[],
            unassigned_requests=[r.id for r in requests],
            total_distance_km=0,
            avg_distance_km=0,
            assignment_rate=0,
            computation_time_ms=round(elapsed_ms, 2)
        )

    # Build cost matrix (rows=resources, cols=requests)
    INF = 1e9  # Large cost for infeasible assignments
    cost_matrix = np.full((n_res, n_req), INF)

    for i, res in enumerate(available_resources):
        for j, req in enumerate(requests):
            if can_fulfill(res, req):
                cost_matrix[i, j] = compute_cost(res, req)

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = []
    assigned_requests = set()

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] >= INF:
            continue  # Skip infeasible pairs
        res = available_resources[i]
        req = requests[j]
        distance = haversine_km(
            res.location.lat, res.location.lng,
            req.destination.lat, req.destination.lng
        )
        assignments.append(Assignment(
            resource_id=res.id,
            request_id=req.id,
            distance_km=round(distance, 2),
            explanation=f"Optimally assigned {res.name} ({res.size.value}) "
                        f"at {distance:.1f} km. Global minimum cost solution."
        ))
        assigned_requests.add(req.id)

    unassigned = [req.id for req in requests if req.id not in assigned_requests]

    elapsed_ms = (time.time() - start) * 1000
    total_dist = sum(a.distance_km for a in assignments)
    avg_dist = total_dist / len(assignments) if assignments else 0

    return AllocationResult(
        algorithm="Hungarian (Optimal Batch)",
        assignments=assignments,
        unassigned_requests=unassigned,
        total_distance_km=round(total_dist, 2),
        avg_distance_km=round(avg_dist, 2),
        assignment_rate=round(len(assignments) / len(requests) * 100, 1) if requests else 0,
        computation_time_ms=round(elapsed_ms, 2)
    )


# ─── ML-Based Allocation ────────────────────────────────────────────────────────

SIZE_ENCODING = {TruckSize.SMALL: 0, TruckSize.MEDIUM: 1, TruckSize.LARGE: 2}


def _extract_features(resource: Resource, request: Request) -> list[float]:
    """Extract features for a resource-request pair."""
    distance = haversine_km(
        resource.location.lat, resource.location.lng,
        request.destination.lat, request.destination.lng
    )
    capacity_utilization = request.weight_kg / resource.capacity_kg
    size_code = SIZE_ENCODING[resource.size]
    surplus_capacity = resource.capacity_kg - request.weight_kg
    return [
        distance,
        request.priority,
        capacity_utilization,
        size_code,
        surplus_capacity,
        distance * (1.0 / request.priority),  # priority-weighted distance
    ]


def _generate_training_data(n_scenarios: int = 50, seed: int = 123):
    """Generate training data by running Hungarian on random scenarios."""
    rng = random.Random(seed)
    center_lat, center_lng = 12.9716, 77.5946
    sizes = list(TruckSize)

    X_train = []
    y_train = []

    for _ in range(n_scenarios):
        n_res = rng.randint(4, 12)
        n_req = rng.randint(4, 12)

        resources = []
        for i in range(n_res):
            size = rng.choice(sizes)
            resources.append(Resource(
                id=f"t{i}", name=f"T{i}",
                location=Location(
                    lat=round(center_lat + rng.uniform(-0.15, 0.15), 4),
                    lng=round(center_lng + rng.uniform(-0.15, 0.15), 4),
                ),
                size=size, capacity_kg=TRUCK_CAPACITIES[size], available=True,
            ))

        requests = []
        for i in range(n_req):
            priority = rng.choices([1, 2, 3], weights=[5, 3, 2])[0]
            weight = rng.uniform(50, 2500)
            requests.append(Request(
                id=f"o{i}",
                destination=Location(
                    lat=round(center_lat + rng.uniform(-0.15, 0.15), 4),
                    lng=round(center_lng + rng.uniform(-0.15, 0.15), 4),
                ),
                weight_kg=round(weight, 1), priority=priority,
            ))

        # Get optimal assignments from Hungarian
        result = hungarian_allocation(resources, requests)
        assigned_pairs = {(a.resource_id, a.request_id) for a in result.assignments}

        # Build labeled data: 1.0 for optimal pairs, 0.0 for non-assigned feasible pairs
        res_map = {r.id: r for r in resources}
        req_map = {r.id: r for r in requests}

        for res in resources:
            for req in requests:
                if not can_fulfill(res, req):
                    continue
                features = _extract_features(res, req)
                label = 1.0 if (res.id, req.id) in assigned_pairs else 0.0
                X_train.append(features)
                y_train.append(label)

    return np.array(X_train), np.array(y_train)


def _get_trained_model():
    """Train and cache the ML model (singleton)."""
    if not hasattr(_get_trained_model, "_model"):
        X, y = _generate_training_data()
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        model.fit(X, y)
        _get_trained_model._model = model
    return _get_trained_model._model


def ml_allocation(resources: list[Resource], requests: list[Request]) -> AllocationResult:
    """ML-Based algorithm: uses a trained model to score and assign resource-request pairs."""
    start = time.time()

    model = _get_trained_model()
    available_resources = [r for r in resources if r.available]

    if not available_resources or not requests:
        elapsed_ms = (time.time() - start) * 1000
        return AllocationResult(
            algorithm="ML-Based (Learned Patterns)",
            assignments=[],
            unassigned_requests=[r.id for r in requests],
            total_distance_km=0,
            avg_distance_km=0,
            assignment_rate=0,
            computation_time_ms=round(elapsed_ms, 2)
        )

    # Score all feasible pairs
    candidates = []
    for res in available_resources:
        for req in requests:
            if can_fulfill(res, req):
                features = _extract_features(res, req)
                candidates.append((res, req, features))

    if not candidates:
        elapsed_ms = (time.time() - start) * 1000
        return AllocationResult(
            algorithm="ML-Based (Learned Patterns)",
            assignments=[],
            unassigned_requests=[r.id for r in requests],
            total_distance_km=0,
            avg_distance_km=0,
            assignment_rate=0,
            computation_time_ms=round(elapsed_ms, 2)
        )

    # Predict scores for all candidates
    X = np.array([c[2] for c in candidates])
    scores = model.predict(X)

    # Sort by predicted score (highest = best match)
    scored_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # Greedily assign based on model scores (each resource/request used at most once)
    assigned_resources = set()
    assigned_requests = set()
    assignments = []

    for (res, req, features), score in scored_candidates:
        if res.id in assigned_resources or req.id in assigned_requests:
            continue
        distance = haversine_km(
            res.location.lat, res.location.lng,
            req.destination.lat, req.destination.lng
        )
        assignments.append(Assignment(
            resource_id=res.id,
            request_id=req.id,
            distance_km=round(distance, 2),
            explanation=(
                f"ML model assigned {res.name} ({res.size.value}) "
                f"with confidence score {score:.3f}. "
                f"Distance: {distance:.1f} km, Priority: {req.priority}, "
                f"Capacity utilization: {req.weight_kg/res.capacity_kg:.0%}."
            )
        ))
        assigned_resources.add(res.id)
        assigned_requests.add(req.id)

    unassigned = [req.id for req in requests if req.id not in assigned_requests]

    elapsed_ms = (time.time() - start) * 1000
    total_dist = sum(a.distance_km for a in assignments)
    avg_dist = total_dist / len(assignments) if assignments else 0

    return AllocationResult(
        algorithm="ML-Based (Learned Patterns)",
        assignments=assignments,
        unassigned_requests=unassigned,
        total_distance_km=round(total_dist, 2),
        avg_distance_km=round(avg_dist, 2),
        assignment_rate=round(len(assignments) / len(requests) * 100, 1) if requests else 0,
        computation_time_ms=round(elapsed_ms, 2)
    )
