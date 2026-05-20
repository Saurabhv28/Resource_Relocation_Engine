"""FastAPI backend for the Resource Allocation Engine."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models import Resource, Request, Location, TruckSize, TRUCK_CAPACITIES
from algorithms import greedy_allocation, hungarian_allocation
import random

app = FastAPI(title="Resource Allocation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_sample_data(n_resources: int = 8, n_requests: int = 10, seed: int = 42):
    """Generate sample delivery fleet data around a city (Bangalore, India)."""
    rng = random.Random(seed)
    center_lat, center_lng = 12.9716, 77.5946  # Bangalore

    sizes = list(TruckSize)
    resources = []
    for i in range(n_resources):
        size = rng.choice(sizes)
        resources.append(Resource(
            id=f"truck-{i+1}",
            name=f"Truck {i+1}",
            location=Location(
                lat=round(center_lat + rng.uniform(-0.1, 0.1), 4),
                lng=round(center_lng + rng.uniform(-0.1, 0.1), 4),
            ),
            size=size,
            capacity_kg=TRUCK_CAPACITIES[size],
            available=True,
        ))

    requests = []
    for i in range(n_requests):
        priority = rng.choices([1, 2, 3], weights=[5, 3, 2])[0]
        weight = rng.uniform(50, 2500)
        requests.append(Request(
            id=f"order-{i+1}",
            destination=Location(
                lat=round(center_lat + rng.uniform(-0.12, 0.12), 4),
                lng=round(center_lng + rng.uniform(-0.12, 0.12), 4),
            ),
            weight_kg=round(weight, 1),
            priority=priority,
        ))

    return resources, requests


class AllocateRequest(BaseModel):
    resources: list[Resource]
    requests: list[Request]


class AllocateResponse(BaseModel):
    greedy: dict
    hungarian: dict


@app.get("/api/sample-data")
def get_sample_data(n_resources: int = 8, n_requests: int = 10, seed: int = 42):
    """Generate sample resources and requests."""
    resources, requests = generate_sample_data(n_resources, n_requests, seed)
    return {"resources": resources, "requests": requests}


@app.post("/api/allocate")
def allocate(data: AllocateRequest):
    """Run both algorithms and return comparison results."""
    greedy_result = greedy_allocation(data.resources, data.requests)
    hungarian_result = hungarian_allocation(data.resources, data.requests)
    return {
        "greedy": greedy_result.model_dump(),
        "hungarian": hungarian_result.model_dump(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
