# Algorithm

Pocketeer detects pockets in proteins using a simple, fast approach based on geometry.

## How It Works

- **Pocketeer** finds empty spaces (pockets) by looking for "alpha-spheres": spheres that fit between protein atoms without touching any atoms except the ones that define them.
- It uses a mathematical method called **Delaunay tessellation** to divide atom coordinates into groups, making it easy to look for these spheres.
- Only spheres of certain sizes (set by the user) and in specific locations are considered pockets.
- Detected spheres are grouped into clusters (pockets) if they are close together.

## Parameters

- **Radius range (`r_min`, `r_max`)**: Sets the smallest and largest spheres to try; usually around 3-6 Å for most proteins.
- **Cluster size (`min_spheres`)**: Minimum number of spheres to call something a pocket.
- **Merge distance**: How close spheres need to be to be grouped into a pocket.
- **Polarity probe radius**: Determines if a sphere is inside the protein (buried) or at the surface.

## When to Use

- Use Pocketeer for fast, simple pocket detection, easy integration in Python workflows, or algorithm customization.
- For more advanced analysis or high-throughput work, tools like fpocket may be better.

## Limitations

- Does not consider chemical features of pockets.
- Estimates pocket volume simply, may miss complex cavities (although this is unlikely).
- Analyzes one structure at a time.

Pocketeer aims for simplicity and useful results, making it easy to understand and adapt. Feel free to submit an issue or PR!
