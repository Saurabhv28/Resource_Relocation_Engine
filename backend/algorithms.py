"""Allocation algorithms: Greedy and Hungarian."""

import time
import math
from models import Resource, Request, Assignment, AllocationResult
from scipy.optimize import linear_sum_assignment
import numpy as np


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
