"""
Drift — Watch thought emerge from memory.

The EAM IS the thinking. A stimulus triggers a chain of reconstructions,
each one drifting through the interference landscape shaped by all stored
experience. The trajectory IS the thought.

Chain formula at each step:
    query_{n+1} = α · reconstruction_n + β · stimulus + γ · running_context

Where:
    α = how much to follow the drift (reconstruction weight)
    β = how much the original stimulus anchors the chain
    γ = how much trajectory history accumulates
"""

import hashlib
from dataclasses import dataclass, field

import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HEATHER_URL = "http://localhost:6380"
DIMS = 128

app = FastAPI(title="Drift")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── word → vector (Lexicon-style hash) ──────────────────────────────

_vec_cache: dict[str, np.ndarray] = {}


def word_to_vec(word: str) -> np.ndarray:
    key = word.lower().strip()
    if key in _vec_cache:
        return _vec_cache[key]
    h = hashlib.sha256(key.encode()).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.RandomState(seed)
    vec = rng.randn(DIMS).astype(np.float64)
    vec /= np.linalg.norm(vec)
    _vec_cache[key] = vec
    return vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── state ────────────────────────────────────────────────────────────

concepts: dict[str, np.ndarray] = {}  # word → hash vector
pairs_written: int = 0


STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "need",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "up", "down", "out", "off", "than",
    "but", "and", "or", "nor", "not", "no", "so", "if", "then",
    "that", "this", "these", "those", "it", "its", "they", "them",
    "their", "we", "our", "he", "she", "his", "her", "you", "your",
    "who", "which", "what", "when", "where", "how", "why",
    "all", "each", "every", "both", "few", "more", "most", "some",
    "any", "other", "such", "too", "very", "just", "also", "about",
})


def tokenize(text: str) -> list[str]:
    """Tokenize: lowercase, strip punctuation, remove stop words."""
    import re
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOP_WORDS]


def register_concept(word: str) -> np.ndarray:
    key = word.lower().strip()
    if key not in concepts:
        concepts[key] = word_to_vec(key)
    return concepts[key]


# shaped embeddings — after consolidation, these replace raw hash vectors
shaped: dict[str, np.ndarray] = {}


async def consolidate():
    """
    Memory consolidation: read every concept back through the EAM.

    The reconstruction isn't the raw hash anymore — it's the hash SHAPED
    by interference from everything that co-occurs with it. These shaped
    vectors encode distributional statistics and have actual structure
    for chain traversal.

    This is like sleep — replay experiences to build representations.
    """
    shaped.clear()
    for word, hvec in concepts.items():
        reconstruction = np.array(await heather_read(hvec.tolist(), strategy="iterative"))
        norm = np.linalg.norm(reconstruction)
        if norm > 1e-10:
            shaped[word] = reconstruction / norm
        else:
            shaped[word] = hvec  # fallback to hash


def find_nearest(vec: np.ndarray, top_k: int = 1) -> list[tuple[str, float]]:
    """Find nearest concepts to a vector. Uses shaped embeddings if available."""
    lookup = shaped if shaped else concepts
    if not lookup:
        return []
    scores = []
    for word, cvec in lookup.items():
        sim = cosine_sim(vec, cvec)
        scores.append((word, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ── heather client ──────────────────────────────────────────────────

async def heather_write(vectors: list[list[float]]):
    async with httpx.AsyncClient() as c:
        await c.post(f"{HEATHER_URL}/write", json={"vectors": vectors}, timeout=10)


async def heather_read(query: list[float], strategy: str = "iterative") -> list[float]:
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{HEATHER_URL}/read",
            json={"query": query, "strategy": strategy},
            timeout=10,
        )
        return r.json()["result"]


async def heather_stats() -> dict:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{HEATHER_URL}/stats", timeout=10)
        return r.json()


# ── seed knowledge ──────────────────────────────────────────────────

async def seed_text(text: str) -> dict:
    """Parse text into co-occurrence pairs and write to SDM."""
    global pairs_written
    words = tokenize(text)
    if len(words) < 2:
        return {"concepts": len(concepts), "pairs_written": 0}

    # register all concepts
    for w in words:
        register_concept(w)

    # write co-occurrence pairs (window size 1 — adjacent words)
    vecs_to_write = []
    for i in range(len(words) - 1):
        v1 = word_to_vec(words[i])
        v2 = word_to_vec(words[i + 1])
        combined = v1 + v2
        norm = np.linalg.norm(combined)
        if norm > 1e-10:
            combined /= norm
            vecs_to_write.append(combined.tolist())

    # also write skip-gram pairs (window size 2) for richer interference
    for i in range(len(words) - 2):
        v1 = word_to_vec(words[i])
        v2 = word_to_vec(words[i + 2])
        combined = v1 + v2
        norm = np.linalg.norm(combined)
        if norm > 1e-10:
            combined /= norm
            vecs_to_write.append(combined.tolist())

    if vecs_to_write:
        await heather_write(vecs_to_write)
        pairs_written += len(vecs_to_write)

    return {"concepts": len(concepts), "pairs_written": len(vecs_to_write)}


# ── the chain — this is where thinking happens ─────────────────────

async def think_chain(
    stimulus: str,
    steps: int = 12,
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1,
) -> list[dict]:
    """
    Run a chain of EAM reconstructions from a stimulus.

    Key insight: follow the RESIDUAL (what's new in the reconstruction),
    not the reconstruction itself. The residual is the direction the
    interference landscape pushes you — that's the thought.

    Uses "fast" read (single-step, no Hopfield convergence) so the
    reconstruction retains more drift signal instead of collapsing
    to the nearest attractor.

    Chain formula:
      residual = reconstruction - projection_of_query_onto_reconstruction
      next_query = α · residual + β · stimulus + γ · running_context + noise
    """
    stim_vec = word_to_vec(stimulus.lower().strip())
    register_concept(stimulus.lower().strip())

    query = stim_vec.copy()
    running_context = np.zeros(DIMS)
    chain = []
    seen_sequence = []
    rng = np.random.RandomState(42)

    for step in range(steps):
        # use FAST read — single step, preserves more drift signal
        reconstruction = np.array(await heather_read(query.tolist(), strategy="fast"))

        recon_norm = np.linalg.norm(reconstruction)
        if recon_norm < 1e-10:
            break
        reconstruction = reconstruction / recon_norm

        # find nearest concept to reconstruction
        nearest = find_nearest(reconstruction, top_k=5)
        if not nearest:
            break

        top_concept, top_sim = nearest[0]

        # how far did the reconstruction drift from the query?
        drift = 1.0 - cosine_sim(query, reconstruction)
        distance_from_start = 1.0 - cosine_sim(stim_vec, reconstruction)

        chain.append({
            "step": step,
            "concept": top_concept,
            "similarity": round(top_sim, 4),
            "drift": round(drift, 4),
            "distance_from_start": round(distance_from_start, 4),
            "nearby": [
                {"concept": w, "similarity": round(s, 4)}
                for w, s in nearest[1:4]
            ],
            "converged": False,
        })

        # check convergence — same concept 3 times in a row (not 2, give it room)
        seen_sequence.append(top_concept)
        if len(seen_sequence) >= 3 and len(set(seen_sequence[-3:])) == 1:
            chain[-1]["converged"] = True
            break

        # also check oscillation (A→B→A→B)
        if len(seen_sequence) >= 4:
            if (seen_sequence[-1] == seen_sequence[-3] and
                    seen_sequence[-2] == seen_sequence[-4]):
                chain[-1]["converged"] = True
                break

        # ── compute next query: follow the RESIDUAL ──
        #
        # The residual = what the EAM added that wasn't in the query.
        # This is the direction the interference landscape pushes.
        #
        # Project query out of reconstruction to get pure residual:
        #   residual = recon - (recon · query) * query
        proj = np.dot(reconstruction, query)
        residual = reconstruction - proj * query
        res_norm = np.linalg.norm(residual)

        if res_norm < 1e-10:
            # reconstruction is parallel to query — no new information
            # add noise to escape
            residual = rng.randn(DIMS)
            residual /= np.linalg.norm(residual)
        else:
            residual = residual / res_norm

        # accumulate context (trajectory memory)
        running_context += reconstruction
        rc_norm = np.linalg.norm(running_context)
        rc = running_context / rc_norm if rc_norm > 1e-10 else np.zeros(DIMS)

        # small exploration noise to avoid getting trapped
        noise = rng.randn(DIMS) * 0.05

        # weighted combination
        next_query = (
            alpha * residual +       # follow what's new
            beta * stim_vec +         # anchor to stimulus
            gamma * rc +              # trajectory memory
            noise                     # exploration
        )
        qnorm = np.linalg.norm(next_query)
        if qnorm > 1e-10:
            query = next_query / qnorm
        else:
            break

    return chain


# ── 2D projection for landscape ─────────────────────────────────────

def compute_landscape(chain_concepts: list[str]) -> dict:
    """PCA projection of all concepts + chain trajectory to 2D."""
    lookup = shaped if shaped else concepts
    if len(lookup) < 3:
        return {"concepts": [], "trajectory": []}

    words = list(lookup.keys())
    vecs = np.array([lookup[w] for w in words])

    # add chain vectors
    chain_vecs = [lookup.get(c, word_to_vec(c)) for c in chain_concepts]

    all_vecs = np.vstack([vecs] + [v.reshape(1, -1) for v in chain_vecs]) if chain_vecs else vecs

    # simple PCA: center, SVD, project to 2D
    mean = all_vecs.mean(axis=0)
    centered = all_vecs - mean
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ Vt[:2].T  # project onto first 2 PCs
    except np.linalg.LinAlgError:
        return {"concepts": [], "trajectory": []}

    # normalize to [-1, 1] range
    max_abs = np.abs(proj).max()
    if max_abs > 1e-10:
        proj /= max_abs

    n_concepts = len(words)
    concept_points = [
        {"name": words[i], "x": round(float(proj[i, 0]), 4), "y": round(float(proj[i, 1]), 4)}
        for i in range(n_concepts)
    ]

    trajectory = [
        {"x": round(float(proj[n_concepts + i, 0]), 4), "y": round(float(proj[n_concepts + i, 1]), 4)}
        for i in range(len(chain_concepts))
    ]

    return {"concepts": concept_points, "trajectory": trajectory}


# ── routes ──────────────────────────────────────────────────────────

class SeedRequest(BaseModel):
    text: str


@app.post("/api/seed")
async def seed(req: SeedRequest):
    result = await seed_text(req.text)
    await consolidate()  # replay — build shaped embeddings
    return result


class ThinkRequest(BaseModel):
    stimulus: str
    steps: int = 12
    alpha: float = 0.7
    beta: float = 0.2
    gamma: float = 0.1


@app.post("/api/think")
async def think(req: ThinkRequest):
    chain = await think_chain(
        req.stimulus,
        steps=req.steps,
        alpha=req.alpha,
        beta=req.beta,
        gamma=req.gamma,
    )

    chain_concepts = [step["concept"] for step in chain]
    landscape = compute_landscape(chain_concepts)

    return {
        "chain": chain,
        "landscape": landscape,
        "stats": {
            "concepts": len(concepts),
            "pairs_written": pairs_written,
        },
    }


@app.post("/api/reset")
async def reset():
    global pairs_written
    concepts.clear()
    shaped.clear()
    _vec_cache.clear()
    pairs_written = 0
    return {"cleared": True}


@app.get("/api/concepts")
async def list_concepts():
    return {"concepts": sorted(concepts.keys()), "count": len(concepts)}


@app.get("/api/stats")
async def stats():
    try:
        hs = await heather_stats()
    except Exception:
        hs = {}
    return {
        "concepts": len(concepts),
        "pairs_written": pairs_written,
        "heather": hs,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "dims": DIMS}


# ── starter knowledge ───────────────────────────────────────────────

STARTER_KNOWLEDGE = """
Forest fires spread through dry connected trees creating firebreaks prevents cascading destruction
Firebreaks interrupt the chain of burning by removing fuel from the path of spreading flames
Controlled burns deliberately destroy fuel to protect the larger forest from uncontrolled wildfire
Fire transforms dense undergrowth into open space allowing dormant seeds to finally germinate

Immune systems remember past infections antibodies adapt to new threats herd immunity protects populations
White blood cells patrol searching for foreign invaders and destroying infected cells
Vaccination trains the immune system exposing it to weakened versions of dangerous pathogens
Fever raises temperature creating hostile environment for bacteria accelerating immune response
Autoimmune disorders occur when defense systems mistakenly attack healthy tissue
Allergies are immune overreaction to harmless substances inflammation damages the body it protects

Financial markets crash when panic spreads cascading selling creates bank runs trust collapses
Circuit breakers halt trading when prices fall fast giving traders time to think rationally
Bubbles form when speculation drives prices above intrinsic value feedback loops amplify greed
Liquidity flows like water seeking lowest point capital pools where returns concentrate
Market contagion spreads fear across borders one collapse triggers another like dominos falling

Computer networks fail when nodes overload cascading failures spread through connected services
Load balancers distribute traffic across redundant servers preventing single point failure
Firewalls filter incoming traffic blocking malicious requests before reaching vulnerable services
Latency increases when bottlenecks form packets queue waiting their turn like traffic congestion
Distributed systems sacrifice consistency for availability partitioning creates islands of certainty

Evolution selects organisms that adapt mutations create variation natural selection filters population
Genetic diversity increases resilience monocultures collapse under new threats
Convergent evolution produces similar solutions in unrelated species facing same pressures
Arms races escalate between predator and prey each adaptation triggers counter adaptation endlessly
Extinction removes species that cannot adapt opening ecological niches for new organisms

Cities grow through connected neighborhoods traffic jams cascade through congested networks
Infrastructure decay cascades when maintenance deferred small cracks become structural failures
Gentrification displaces communities when investment flows into neglected neighborhoods
Urban heat islands trap warmth concrete absorbs sunlight radiating heat long after sunset
Sprawl consumes surrounding farmland as population growth pushes boundaries outward endlessly

Rivers carve paths through landscape water flows downhill erosion shapes terrain over time
Watersheds collect rainfall funneling water through branching tributaries into streams
Floods deposit sediment enriching floodplains but destroying structures built too close
Erosion gradually weakens foundations undermining structures that seemed permanent

Epidemics spread through contact networks vaccination creates herd immunity breaks transmission
Quarantine isolates infected preventing transmission to healthy population members
Superspreaders amplify outbreak growth transmitting disease to many more than average
Mutations allow pathogens to evolve resistance against treatments escape immune recognition

Brain neurons fire together wire together memories form through repeated activation strengthening
Sleep consolidates memories replaying experiences strengthening important patterns pruning weak
Synaptic plasticity allows learning strengthening frequent connections weakening unused pathways
Habits form when repeated behaviors create strong pathways executing automatically unconsciously
Dreams remix fragments of memory creating bizarre combinations that occasionally spark insight
Trauma burns deep neural pathways that activate involuntarily triggered by sensory fragments

Earthquakes release pressure along fault lines aftershocks cascade through connected structures
Tectonic plates push against each building stress until sudden rupture releases stored energy
Tsunami waves propagate across ocean carrying enormous energy from underwater displacement
Volcanic eruptions release pressure from deep underground transforming landscape dramatically

Ideas spread through social networks viral concepts replicate mutate as people share adapt
Innovation combines existing ideas connecting previously unrelated concepts creates breakthroughs
Paradigm shifts occur when accumulated anomalies overwhelm existing frameworks forcing revolution
Censorship suppresses information but forbidden ideas often spread faster underground
Memes evolve through cultural selection surviving variants are catchier simpler more emotional

Ecosystems maintain balance through feedback predators control prey diversity creates resilience
Keystone species have disproportionate impact removing them triggers cascading ecosystem collapse
Invasive species disrupt balance outcompeting native organisms lacking natural predators
Decomposition recycles nutrients from dead organisms returning elements to soil feeding growth
Succession transforms barren ground through stages pioneer species prepare soil for climax forest

Ant colonies coordinate complex behavior through chemical signals without central planning
Flocking birds maintain formation through local rules following nearest neighbors creating patterns
Crystallization produces order from disorder as temperature drops below critical threshold
Feedback loops amplify changes positive feedback accelerates growth negative feedback stabilizes
Resonance occurs when frequency matches natural frequency small inputs produce enormous responses
Emergence produces complex behavior from simple rules no individual component understands the whole

Music harmony emerges when frequencies align mathematical ratios create consonance dissonance creates tension
Rhythm entrains biological systems heartbeat synchronizes breathing follows tempo movement locks
Melody carries memory songs trigger vivid recall of emotions places people long forgotten
Orchestras coordinate many instruments into coherent sound conductor shapes collective expression
Improvisation requires deep pattern knowledge breaking rules creatively demands mastering them first
Dissonance creates tension that demands resolution the ear expects harmony to return

Architecture distributes weight through structure columns transfer load foundations anchor to bedrock
Flying buttresses redirect force outward allowing thin walls to rise impossibly high
Bridges span gaps suspension cables distribute tension equally across the entire structure
Cracks propagate through material following paths of least resistance stress concentrates at tips
Gothic cathedrals channel light through colored glass transforming interior space into luminous sanctuary

Chemistry transforms substances through reactions catalysts accelerate change without being consumed
Crystallization separates pure substance from mixture as temperature drops ordered lattice excludes impurities
Combustion releases stored energy rapidly fire is oxidation reaction liberating heat and light
Fermentation transforms sugar into alcohol microorganisms digest creating byproducts we value
Catalysts lower activation energy barriers allowing reactions that would otherwise require extreme conditions
Acids dissolve structure breaking bonds that held material together corrosion eats metal slowly

Language evolves through usage words shift meaning grammar simplifies over generations
Metaphors map structure from one domain onto another understanding grows through analogy
Translation preserves meaning while transforming expression some concepts resist crossing between languages
Pidgins emerge when cultures collide simplified communication evolves into creole languages with full grammar
Etymology traces words back through time revealing ancient connections between modern concepts
Slang spreads through communities marking identity distinguishing insiders from outsiders

Warfare concentrates force at weak points flanking maneuvers bypass strong defenses attacking from behind
Siege warfare cuts supply lines starving defenders until resistance collapses from exhaustion
Guerrilla tactics use terrain advantage small mobile forces harass larger armies avoiding direct confrontation
Fortifications channel attackers into killing zones walls force predictable paths defenders exploit
Espionage gathers intelligence about enemy plans deception misleads opponents into costly mistakes
Attrition wears down opposition gradually exhausting resources until collapse becomes inevitable

Ocean currents circulate heat around globe deep water sinks cold dense surface water rises warm light
Tides respond to gravitational pull moon draws water creating predictable rhythmic cycles
Waves carry energy across vast distances water particles move in circles barely traveling forward
Coral builds reef structures slowly over centuries creating habitat that supports enormous biodiversity
Pressure increases with depth crushing organisms that venture too deep without adaptation

Cooking transforms raw ingredients through heat chemical reactions create flavor texture aroma
Fermentation cultures digest sugars producing complex flavors that develop slowly over time
Emulsification suspends oil in water creating stable mixture from normally incompatible liquids
Caramelization breaks down sugar molecules creating hundreds of new compounds rich bitter sweet complex
Seasoning balances flavor profiles salt suppresses bitterness acid brightens sweetness enhances

Gardens cultivate growth through careful tending pruning removes dead wood directing energy to new shoots
Composting transforms waste into fertile soil decomposition creates foundation for future growth
Grafting joins different plants creating hybrid that combines rootstock strength with desired fruit
Weeds compete for resources crowding out cultivated plants requiring constant vigilance to control
Permaculture designs self sustaining systems mimicking natural ecosystem patterns reducing maintenance

Mirrors reflect light creating virtual images that seem real but reverse left and right
Lenses focus light bending rays to converge at single point concentrating diffuse energy
Prisms separate white light into spectrum revealing hidden colors that were always present combined
Shadows reveal shape by showing where light cannot reach darkness maps the geometry of obstruction
Holograms store interference patterns encoding three dimensional information in two dimensional surface

Memory reconstructs rather than retrieves each recall slightly alters the original changing emphasis
Forgetting removes unused connections making remaining memories more accessible reducing noise
Nostalgia filters memory emphasizing positive details smoothing rough edges of actual experience
Deja vu triggers false recognition pattern matching fires incorrectly creating illusion of repetition
Conditioning links stimulus to response repeated pairing creates automatic association that bypasses thinking

Gravity attracts mass warping space itself heavier objects create deeper wells pulling neighbors inward
Orbits balance forward motion against gravitational pull falling continuously but never landing
Tidal forces stretch objects that approach too close differential gravity tears apart fragile structures
Black holes collapse space so completely that nothing escapes event horizon marks point of no return
Satellites maintain altitude by matching orbital speed to gravitational pull balanced between falling and flying

Weather patterns emerge from differential heating uneven warming creates pressure gradients driving wind
Hurricanes feed on warm ocean water heat energy converts to kinetic energy spinning faster gaining strength
Drought amplifies itself dry soil reflects more heat reducing rainfall creating vicious cycle
Lightning discharges accumulated electrical potential bridging gap between cloud and ground in milliseconds
Fog forms when temperature drops to dewpoint moisture condenses on particles suspended in still air
"""


@app.post("/api/load-starter")
async def load_starter():
    total = {"concepts": 0, "pairs_written": 0}
    for line in STARTER_KNOWLEDGE.strip().split("\n"):
        line = line.strip()
        if line:
            result = await seed_text(line)
            total["concepts"] = result["concepts"]
            total["pairs_written"] += result["pairs_written"]
    await consolidate()  # replay all — build shaped embeddings
    return total
