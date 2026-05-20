"""Data models for the Delivery Fleet Resource Allocation Engine."""

from pydantic import BaseModel
from typing import Optional
from enum import Enum


class Location(BaseModel):
    lat: float
    lng: float


class TruckSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


# Capacity in kg for each truck size
TRUCK_CAPACITIES = {
    TruckSize.SMALL: 500,
    TruckSize.MEDIUM: 1500,
    TruckSize.LARGE: 3000,
}


class Resource(BaseModel):
    """A delivery truck with location, capacity, and availability."""
    id: str
    name: str
    location: Location
    size: TruckSize
    capacity_kg: float
    available: bool = True


class Request(BaseModel):
    """A delivery order with location, weight, and priority."""
    id: str
    destination: Location
    weight_kg: float
    priority: int = 1  # 1 (low) to 3 (high)


class Assignment(BaseModel):
    """A resource-request pairing with explanation."""
    resource_id: str
    request_id: str
    distance_km: float
    explanation: str


class AllocationResult(BaseModel):
    """Result from running an allocation algorithm."""
    algorithm: str
    assignments: list[Assignment]
    unassigned_requests: list[str]
    total_distance_km: float
    avg_distance_km: float
    assignment_rate: float  # percentage of requests fulfilled
    computation_time_ms: float
