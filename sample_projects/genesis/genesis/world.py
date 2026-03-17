"""Core simulation: artificial life evolving inside SDM."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .client import HeatherClient

DIMENSION = 32


@dataclass
class Creature:
    vector: np.ndarray
    fitness: float = 0.0
    age: int = 0
    generation_born: int = 0
    id: int = 0


@dataclass
class Species:
    id: int
    name: str
    creatures: list[Creature] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.creatures)

    @property
    def avg_fitness(self) -> float:
        if not self.creatures:
            return 0.0
        return sum(c.fitness for c in self.creatures) / len(self.creatures)

    @property
    def center(self) -> np.ndarray:
        vecs = np.array([c.vector for c in self.creatures])
        c = vecs.mean(axis=0)
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        return c

    @property
    def status(self) -> str:
        f = self.avg_fitness
        if f > 0.85:
            return "thriving"
        elif f > 0.7:
            return "stable"
        elif f > 0.55:
            return "stressed"
        elif f > 0.4:
            return "declining"
        return "critical"


@dataclass
class GenerationReport:
    generation: int
    population: int
    species: list[Species]
    births: int
    deaths: int
    events: list[str]


# Species naming: Greek letters
SPECIES_NAMES = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi",
    "Rho", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
]


class World:
    def __init__(self, heather: HeatherClient):
        self.heather = heather
        self.creatures: list[Creature] = []
        self.generation = 0
        self.history: list[GenerationReport] = []
        self._next_creature_id = 0
        self._next_species_id = 0
        self._prev_species: list[Species] = []
        self._extinct_names: list[str] = []

        # Parameters
        self.initial_population = 50
        self.max_population = 120
        self.death_threshold = 0.60
        self.mutation_rate = 0.12
        self.cluster_threshold = 0.35

    def _new_id(self) -> int:
        self._next_creature_id += 1
        return self._next_creature_id

    def _new_species_name(self) -> str:
        used = {s.name for s in self._prev_species}
        for name in SPECIES_NAMES:
            if name not in used and name not in self._extinct_names:
                return name
        # Fallback: reuse extinct names
        if self._extinct_names:
            return self._extinct_names.pop(0) + "'"
        return f"Species-{self._next_species_id}"

    def seed(self):
        """Create initial population of random creatures and write to SDM."""
        vectors = []
        for _ in range(self.initial_population):
            vec = np.random.randn(DIMENSION).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            creature = Creature(
                vector=vec,
                fitness=1.0,
                age=0,
                generation_born=0,
                id=self._new_id(),
            )
            self.creatures.append(creature)
            vectors.append(vec.tolist())

        # Write all to SDM at once
        self.heather.write(vectors)

    def evaluate(self):
        """Measure fitness: how well does the EAM reconstruct each creature?"""
        for creature in self.creatures:
            reconstructed = self.heather.read(creature.vector.tolist(), strategy="fast")
            recon = np.array(reconstructed, dtype=np.float32)

            # Cosine similarity between original and reconstruction
            norm_orig = np.linalg.norm(creature.vector)
            norm_recon = np.linalg.norm(recon)
            if norm_orig > 0 and norm_recon > 0:
                creature.fitness = float(
                    np.dot(creature.vector, recon) / (norm_orig * norm_recon)
                )
            else:
                creature.fitness = 0.0

    def cull(self) -> int:
        """Remove creatures below death threshold. Returns number killed."""
        before = len(self.creatures)
        self.creatures = [c for c in self.creatures if c.fitness >= self.death_threshold]
        return before - len(self.creatures)

    def reproduce(self) -> int:
        """Select fit parents, produce offspring via blending. Returns number of births."""
        if len(self.creatures) < 2:
            return 0

        # How many offspring to produce
        target = min(self.max_population, self.initial_population)
        num_offspring = max(0, target - len(self.creatures))
        # Always produce at least a few if population is low
        if len(self.creatures) < self.initial_population // 2:
            num_offspring = max(num_offspring, self.initial_population // 3)
        # Cap offspring
        num_offspring = min(num_offspring, self.max_population - len(self.creatures))
        if num_offspring <= 0:
            return 0

        # Tournament selection
        offspring_vectors = []
        new_creatures = []
        for _ in range(num_offspring):
            # Pick parents via tournament (sample 3, take best 2)
            sample_size = min(3, len(self.creatures))
            candidates = random.sample(self.creatures, sample_size)
            candidates.sort(key=lambda c: c.fitness, reverse=True)
            parent_a = candidates[0]
            parent_b = candidates[1] if len(candidates) > 1 else candidates[0]

            # Blend with random weight
            weight = random.uniform(0.3, 0.7)
            child_vec = weight * parent_a.vector + (1 - weight) * parent_b.vector

            # Mutate
            noise = self.mutation_rate * np.random.randn(DIMENSION).astype(np.float32)
            child_vec = child_vec + noise

            # Normalize
            norm = np.linalg.norm(child_vec)
            if norm > 0:
                child_vec = child_vec / norm

            child = Creature(
                vector=child_vec,
                fitness=0.0,
                age=0,
                generation_born=self.generation,
                id=self._new_id(),
            )
            new_creatures.append(child)
            offspring_vectors.append(child_vec.tolist())

        # Write offspring to SDM
        if offspring_vectors:
            self.heather.write(offspring_vectors)

        self.creatures.extend(new_creatures)
        return len(new_creatures)

    def reinforce(self):
        """Re-write surviving creatures to SDM, strengthening their patterns."""
        if not self.creatures:
            return
        vectors = [c.vector.tolist() for c in self.creatures]
        self.heather.write(vectors)

    def cluster(self) -> list[Species]:
        """Group creatures into species by cosine similarity."""
        if not self.creatures:
            return []

        clusters: list[list[Creature]] = []
        centers: list[np.ndarray] = []

        for creature in sorted(self.creatures, key=lambda c: c.fitness, reverse=True):
            placed = False
            for i, center in enumerate(centers):
                sim = float(np.dot(creature.vector, center))
                if sim > self.cluster_threshold:
                    clusters[i].append(creature)
                    # Update center
                    vecs = np.array([c.vector for c in clusters[i]])
                    new_center = vecs.mean(axis=0)
                    norm = np.linalg.norm(new_center)
                    if norm > 0:
                        centers[i] = new_center / norm
                    placed = True
                    break
            if not placed:
                clusters.append([creature])
                centers.append(creature.vector.copy())

        # Build Species objects, trying to match previous species by center similarity
        species_list = []
        used_prev = set()

        for cluster_creatures in clusters:
            cluster_center = np.mean([c.vector for c in cluster_creatures], axis=0)
            norm = np.linalg.norm(cluster_center)
            if norm > 0:
                cluster_center = cluster_center / norm

            # Try to match to a previous species
            best_match = None
            best_sim = 0.0
            for prev in self._prev_species:
                if prev.id in used_prev:
                    continue
                sim = float(np.dot(cluster_center, prev.center))
                if sim > self.cluster_threshold and sim > best_sim:
                    best_match = prev
                    best_sim = sim

            if best_match:
                used_prev.add(best_match.id)
                sp = Species(id=best_match.id, name=best_match.name, creatures=cluster_creatures)
            else:
                self._next_species_id += 1
                sp = Species(
                    id=self._next_species_id,
                    name=self._new_species_name(),
                    creatures=cluster_creatures,
                )

            species_list.append(sp)

        # Detect extinctions
        for prev in self._prev_species:
            if prev.id not in used_prev:
                self._extinct_names.append(prev.name)

        return species_list

    def step(self) -> GenerationReport:
        """Run one generation. Returns a report."""
        self.generation += 1

        # Age all creatures
        for c in self.creatures:
            c.age += 1

        # Evaluate fitness
        self.evaluate()

        # Identify species BEFORE culling (so we can detect extinctions)
        prev_species_names = {s.name for s in self._prev_species}

        # Cull
        deaths = self.cull()

        # Reproduce
        births = self.reproduce()

        # Reinforce survivors
        self.reinforce()

        # Re-evaluate after reinforcement (so display shows post-reinforcement fitness)
        self.evaluate()

        # Cluster into species
        species = self.cluster()
        current_names = {s.name for s in species}

        # Detect events
        events = []
        for name in prev_species_names:
            if name not in current_names:
                events.append(f"EXTINCTION: {name} has gone extinct")

        new_species = current_names - prev_species_names
        for name in new_species:
            if self.generation > 1:
                events.append(f"SPECIATION: {name} has emerged")

        # Check for niche pressure
        for sp in species:
            if sp.size > self.max_population * 0.5:
                events.append(f"PRESSURE: {sp.name} approaching carrying capacity")

        self._prev_species = species

        report = GenerationReport(
            generation=self.generation,
            population=len(self.creatures),
            species=species,
            births=births,
            deaths=deaths,
            events=events,
        )
        self.history.append(report)
        return report
