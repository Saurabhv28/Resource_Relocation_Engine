# Resource Allocation Engine — Delivery Fleet

A Resource Allocation Engine that optimally assigns delivery trucks to incoming orders, comparing **Greedy**, **Hungarian**, and **ML-Based** algorithms with a visual web interface.

## Domain

**Delivery Fleet**: Trucks (resources) at various locations with different cargo capacities are assigned to delivery orders (requests) with specific destinations, weights, and priorities.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Leaflet, OpenStreetMap (free, no API key) |
| Backend | FastAPI (Python) |
| Other Python libs | SciPy, NumPy, Pydantic, scikit-learn |
| Testing | pytest |

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI server
│   ├── models.py            # Data models (Resource, Request, Assignment)
│   ├── algorithms.py        # Greedy + Hungarian + ML-Based algorithms
│   ├── test_algorithms.py   # Test suite (22 tests)
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── package.json         # Node dependencies (run npm install)
│   ├── index.html           # Standalone version (no build needed)
│   ├── public/index.html    # React public HTML
│   └── src/
│       ├── index.js
│       ├── App.js
│       ├── App.css
│       └── components/
│           ├── MapView.js
│           └── MetricsPanel.js
├── .gitignore
└── README.md
```

## Setup & Run

### Prerequisites
- Python 3.10+
- Node.js 16+

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Server starts at `http://localhost:8000`.

### Frontend

**Option A — No build (standalone HTML):**
Open `frontend/index.html` directly in a browser. It uses CDN-loaded Leaflet and talks to the backend API.

**Option B — React dev server (requires Node.js):**
```bash
cd frontend
npm install
npm start
```

App opens at `http://localhost:3000`.

### Run Tests

```bash
cd backend
pytest test_algorithms.py -v
```

## How It Works

### Data Model

| Entity | Key Fields |
|--------|-----------|
| **Resource** (Truck) | id, location, size (small/medium/large), capacity_kg, available |
| **Request** (Order) | id, destination, weight_kg, priority (1-3) |
| **Assignment** | resource_id, request_id, distance_km, explanation |

### Constraints

- **Hard**: Truck must be available; truck capacity ≥ order weight
- **Soft**: Minimize travel distance; prioritize urgent orders

### Algorithms

#### 1. Greedy (Nearest-First)
- Sorts requests by priority (highest first)
- For each request, assigns the nearest truck that satisfies hard constraints
- O(n×m) time complexity
- Fast but locally optimal — may miss globally better pairings

#### 2. Hungarian (Kuhn-Munkres)
- Builds a cost matrix (distance × priority factor) for all feasible truck-order pairs
- Solves the assignment problem optimally using `scipy.optimize.linear_sum_assignment`
- O(n³) time complexity
- Finds the globally optimal set of assignments that minimizes total cost

#### 3. ML-Based (Learned Patterns)
- Uses a **Gradient Boosting Regressor** (scikit-learn) trained on synthetic scenarios solved by the Hungarian algorithm
- Extracts features per resource-request pair: distance, priority, capacity utilization, truck size, surplus capacity, and priority-weighted distance
- Predicts a "match score" for every feasible (truck, order) pair and greedily assigns in descending score order
- Training data is generated on first call (50 random scenarios) and the model is cached as a singleton
- Adapts to learned cost patterns without relying on a hand-crafted cost formula at inference time
- Balances between Greedy speed and Hungarian optimality

## Algorithm Comparison Analysis

### Key Findings

| Metric | Greedy | Hungarian | ML-Based |
|--------|--------|-----------|----------|
| Speed | Fastest (O(n×m)) | Slowest (O(n³)) | Medium (training once + O(n×m) inference) |
| Total distance | Higher | Lowest (optimal) | Near-optimal |
| Implementation | Simple loop | Uses scipy | Uses scikit-learn GBR |
| Approach | Heuristic | Exact optimization | Learned heuristic |

### When each wins:

- **Greedy wins** when: requests arrive one-by-one in real-time (online setting), or when the problem is small enough that the difference is negligible.
- **Hungarian wins** when: you can batch requests and optimize globally. The savings grow with problem size and when resources are scarce relative to requests.
- **ML-Based wins** when: you want near-optimal quality with the flexibility to incorporate complex patterns (e.g., historical demand, driver preferences) that are hard to encode in a cost matrix. Also useful when the cost structure may change over time — retrain the model rather than re-engineer the algorithm.

### Insight

With 8 trucks and 10 orders, Hungarian typically saves **5–15%** total distance over Greedy. The ML-Based approach learns to approximate Hungarian's decisions and generally achieves total distance within **5%** of optimal while being more extensible to richer feature sets.
