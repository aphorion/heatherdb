"""Cinema — Movie Recommendations from Emergent Memory."""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

from heather import HeatherClient
from omdb import OMDbClient
from engine import CinemaEngine
from seed_data import SEED_MOVIES as LOCAL_SEED_MOVIES

OMDB_KEY = os.getenv("OMDB_API_KEY", "")
HEATHER_URL = os.getenv("HEATHER_URL", "http://localhost:6380")

heather: HeatherClient
omdb: OMDbClient
engine: CinemaEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global heather, omdb, engine
    heather = HeatherClient(HEATHER_URL)
    omdb = OMDbClient(OMDB_KEY)
    engine = CinemaEngine(heather)
    yield
    await heather.close()
    await omdb.close()


app = FastAPI(title="Cinema", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Models ---

class WatchRequest(BaseModel):
    imdb_id: str

class CatalogAddRequest(BaseModel):
    imdb_id: str

class BlendRequest(BaseModel):
    users: list[str]

class StabilityRequest(BaseModel):
    outlier_id: str | None = None


# --- Health ---

@app.get("/api/health")
async def health():
    heather_ok = await heather.health()
    return {"status": "ok", "heather_ok": heather_ok}


# --- OMDb Proxy ---

@app.get("/api/search")
async def search_movies(q: str = Query(..., min_length=1)):
    if not OMDB_KEY:
        raise HTTPException(500, "OMDB_API_KEY not configured")
    try:
        results = await omdb.search(q)
    except ValueError as e:
        raise HTTPException(401, str(e))
    return {"movies": results}


@app.get("/api/movie/{imdb_id}")
async def get_movie(imdb_id: str):
    if not OMDB_KEY:
        raise HTTPException(500, "OMDB_API_KEY not configured")
    try:
        movie = await omdb.get_movie(imdb_id)
    except ValueError as e:
        raise HTTPException(401, str(e))
    if not movie:
        raise HTTPException(404, "Movie not found")
    return movie


# --- Catalog ---

@app.post("/api/catalog/add")
async def add_to_catalog(req: CatalogAddRequest):
    # Check if already in catalog
    if req.imdb_id in engine.catalog:
        return {"status": "exists", "imdb_id": req.imdb_id}
    # Fetch from OMDb
    try:
        movie = await omdb.get_movie(req.imdb_id)
    except ValueError as e:
        raise HTTPException(401, str(e))
    if not movie:
        raise HTTPException(404, "Movie not found on OMDb")
    embedding = engine.add_to_catalog(movie)
    return {"status": "added", "imdb_id": req.imdb_id}


@app.get("/api/catalog")
async def get_catalog():
    return {"movies": engine.get_catalog()}


# --- Users ---

@app.get("/api/users")
async def list_users():
    return {"users": engine.list_users()}


@app.post("/api/users/{name}/watch")
async def watch_movie(name: str, req: WatchRequest):
    ok = await engine.watch_movie(name, req.imdb_id)
    if not ok:
        raise HTTPException(404, "Movie not in catalog")
    return {"status": "watched", "user": name, "imdb_id": req.imdb_id}


@app.get("/api/users/{name}/history")
async def get_history(name: str):
    return {"movies": engine.get_history(name)}


@app.get("/api/users/{name}/fingerprint")
async def get_fingerprint(name: str):
    fp = engine.get_fingerprint(name)
    if not fp:
        return {"fingerprint": None, "genre_affinities": {}, "movies_watched": 0}
    return fp


@app.get("/api/users/{name}/recommend")
async def recommend(name: str, n: int = Query(default=10, ge=1, le=50), threshold: float = Query(default=0.0)):
    recs, fidelity = engine.recommend(name, n=n, threshold=threshold)
    return {"recommendations": recs, "fidelity": fidelity}


@app.get("/api/users/{name}/recommend/{imdb_id}/explain")
async def explain_recommendation(name: str, imdb_id: str):
    result = await engine.explain_recommendation(name, imdb_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/api/users/{name}/evolution")
async def taste_evolution(name: str):
    return engine.get_taste_evolution(name)


@app.delete("/api/users/{name}")
async def delete_user(name: str):
    ok = engine.delete_user(name)
    if not ok:
        raise HTTPException(404, "User not found")
    return {"status": "deleted", "user": name}


# --- Blend ---

@app.post("/api/blend")
async def blend(req: BlendRequest):
    if len(req.users) != 2:
        raise HTTPException(400, "Exactly 2 users required")
    result = engine.blend_fingerprints(req.users[0], req.users[1])
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


# --- Demos ---

@app.post("/api/demo/stability")
async def demo_stability(req: StabilityRequest = StabilityRequest()):
    result = await engine.demo_stability(req.outlier_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


# --- Seed ---

# Curated list of iconic, well-known movies with great posters (~200)
SEED_MOVIES = [
    # Sci-Fi
    "tt0133093",  # The Matrix
    "tt0816692",  # Interstellar
    "tt0083658",  # Blade Runner
    "tt1375666",  # Inception
    "tt0076759",  # Star Wars: A New Hope
    "tt2015381",  # Guardians of the Galaxy
    "tt0088763",  # Back to the Future
    "tt0078748",  # Alien
    "tt0103064",  # Terminator 2
    "tt0062622",  # 2001: A Space Odyssey
    "tt0086190",  # Star Wars: Return of the Jedi
    "tt0080684",  # Star Wars: Empire Strikes Back
    "tt3659388",  # The Martian
    "tt1856101",  # Blade Runner 2049
    "tt0106062",  # Jurassic Park
    "tt2543164",  # Arrival
    "tt1392190",  # Mad Max: Fury Road
    "tt0470752",  # Ex Machina
    "tt0084787",  # The Thing
    "tt0090605",  # Aliens
    "tt0209144",  # Memento
    "tt0482571",  # The Prestige
    "tt1677720",  # Ready Player One
    "tt0418279",  # Transformers
    # Action / Thriller
    "tt0468569",  # The Dark Knight
    "tt0137523",  # Fight Club
    "tt0110912",  # Pulp Fiction
    "tt0114369",  # Se7en
    "tt0167260",  # LOTR: Return of the King
    "tt0120737",  # LOTR: Fellowship
    "tt0167261",  # LOTR: Two Towers
    "tt0361748",  # Inglourious Basterds
    "tt1853728",  # Django Unchained
    "tt0848228",  # The Avengers
    "tt4154756",  # Avengers: Infinity War
    "tt4154796",  # Avengers: Endgame
    "tt0371746",  # Iron Man
    "tt1825683",  # Black Panther
    "tt0172495",  # Gladiator
    "tt0082971",  # Raiders of the Lost Ark
    "tt0097576",  # Indiana Jones: Last Crusade
    "tt0758758",  # Into the Wild
    "tt0993846",  # The Wolf of Wall Street
    "tt1130884",  # Shutter Island
    "tt0266697",  # Kill Bill: Vol. 1
    "tt0378194",  # Kill Bill: Vol. 2
    "tt0102926",  # The Silence of the Lambs
    "tt0075314",  # Taxi Driver
    "tt0117951",  # Trainspotting
    "tt0095016",  # Die Hard
    "tt0056172",  # Lawrence of Arabia
    "tt0095765",  # Cinema Paradiso
    "tt0119698",  # Princess Mononoke
    "tt0169547",  # American Beauty
    "tt0120815",  # Saving Private Ryan
    "tt0047478",  # Seven Samurai
    "tt2267998",  # Gone Girl
    "tt0180093",  # Requiem for a Dream
    "tt1345836",  # The Dark Knight Rises
    "tt0120338",  # Titanic
    "tt0054215",  # Psycho
    "tt0245429",  # Spirited Away
    "tt0076759",  # Star Wars (dupe guard handled by engine)
    "tt0364569",  # Oldboy
    "tt1049413",  # Up
    "tt0099685",  # Goodfellas
    "tt0405094",  # The Lives of Others
    "tt0086879",  # Amadeus
    "tt0057012",  # Dr. Strangelove
    "tt0050083",  # 12 Angry Men
    "tt0114814",  # The Usual Suspects
    "tt0071562",  # The Godfather Part II
    "tt0034583",  # Casablanca
    "tt0021749",  # City Lights
    "tt0043014",  # Sunset Boulevard
    "tt0053125",  # North by Northwest
    "tt0032138",  # The Wizard of Oz
    "tt0052357",  # Vertigo
    "tt0033467",  # Citizen Kane
    "tt0066921",  # A Clockwork Orange
    "tt0081398",  # Raging Bull
    "tt0075148",  # Rocky
    "tt0112573",  # Braveheart
    "tt0060196",  # The Good, the Bad and the Ugly
    "tt0253474",  # The Pianist
    "tt0091251",  # Come and See
    # Drama
    "tt0111161",  # The Shawshank Redemption
    "tt0068646",  # The Godfather
    "tt0109830",  # Forrest Gump
    "tt0108052",  # Schindler's List
    "tt0407887",  # The Departed
    "tt0038650",  # It's a Wonderful Life
    "tt0120689",  # The Green Mile
    "tt0110413",  # Léon: The Professional
    "tt0264464",  # Catch Me If You Can
    "tt0317248",  # City of God
    "tt0086250",  # Scarface
    "tt0047396",  # Rear Window
    "tt1187043",  # 3 Idiots
    "tt0198781",  # Monsters, Inc.
    "tt0435761",  # Toy Story 3
    "tt0892769",  # Dark Knight (alt)
    "tt0073486",  # One Flew Over the Cuckoo's Nest
    "tt0042876",  # Rashomon
    "tt0105236",  # Reservoir Dogs
    "tt0036775",  # Double Indemnity
    "tt0045152",  # Singin' in the Rain
    "tt0372784",  # Batman Begins
    "tt0162222",  # Cast Away
    "tt0118715",  # The Big Lebowski
    "tt0087843",  # Once Upon a Time in America
    "tt0476735",  # My Neighbor Totoro
    "tt0064116",  # Butch Cassidy and the Sundance Kid
    "tt0056592",  # To Kill a Mockingbird
    "tt0082096",  # Das Boot
    "tt0093058",  # Full Metal Jacket
    "tt0044741",  # Ikiru
    "tt0022100",  # M
    "tt0055630",  # Yojimbo
    "tt0040522",  # Bicycle Thieves
    "tt0208092",  # Snatch
    "tt0071853",  # Monty Python and the Holy Grail
    "tt0079470",  # Life of Brian
    # Horror / Thriller
    "tt0081505",  # The Shining
    "tt0070047",  # The Exorcist
    "tt7784604",  # Hereditary
    "tt1084950",  # The Orphanage
    "tt5052448",  # Get Out
    "tt6751668",  # Parasite
    "tt0175880",  # The Sixth Sense
    "tt1591095",  # Drive
    "tt0180052",  # Mulholland Drive
    "tt0119217",  # Good Will Hunting
    "tt0347149",  # Howl's Moving Castle
    "tt0382932",  # Ratatouille
    "tt0986264",  # Like Stars on Earth
    "tt0457430",  # Pan's Labyrinth
    "tt0119488",  # L.A. Confidential
    "tt0070735",  # The Sting
    "tt0055031",  # Judgment at Nuremberg
    "tt0032551",  # The Grapes of Wrath
    "tt0051201",  # 12 Angry Men (1957)
    "tt0015864",  # The Gold Rush
    "tt0017136",  # Metropolis
    "tt0027977",  # Modern Times
    "tt0040897",  # The Treasure of the Sierra Madre
    # Comedy / Romance
    "tt0118799",  # Life is Beautiful
    "tt0211915",  # Amélie
    "tt0477348",  # No Country for Old Men
    "tt0057115",  # The Great Escape
    "tt0050986",  # Wild Strawberries
    "tt0044079",  # Strangers on a Train
    "tt0361862",  # The Motorcycle Diaries
    "tt0059578",  # For a Few Dollars More
    "tt0032976",  # Rebecca
    "tt0053291",  # Some Like It Hot
    "tt0469494",  # There Will Be Blood
    "tt0058946",  # The Battle of Algiers
    "tt0338013",  # Eternal Sunshine of the Spotless Mind
    "tt0083866",  # E.T. the Extra-Terrestrial
    "tt0107290",  # Jurassic Park
    "tt0116282",  # Fargo
    "tt0325980",  # Pirates of the Caribbean
    "tt0167404",  # The Sixth Sense
    "tt0112641",  # Casino
    "tt0044706",  # High Noon
    # Animation
    "tt0910970",  # WALL·E
    "tt0114709",  # Toy Story
    "tt2380307",  # Coco
    "tt0266543",  # Finding Nemo
    "tt2948356",  # Zootopia
    "tt0126029",  # Shrek
    "tt1049413",  # Up
    "tt0892769",  # How to Train Your Dragon
    "tt0317705",  # The Incredibles
    "tt2096673",  # Inside Out
    "tt0986264",  # Taare Zameen Par
    "tt5311514",  # Your Name
    "tt0073195",  # Jaws
    # Modern Classics
    "tt1950186",  # Ford v Ferrari
    "tt7286456",  # Joker
    "tt6966692",  # Green Book
    "tt5580390",  # The Shape of Water
    "tt4633694",  # Spider-Man: Into the Spider-Verse
    "tt1160419",  # Dune
    "tt3170832",  # Room
    "tt2084970",  # The Imitation Game
    "tt2119532",  # Hacksaw Ridge
    "tt1201607",  # Harry Potter: Deathly Hallows 2
    "tt0241527",  # Harry Potter: Philosopher's Stone
    "tt0926084",  # Harry Potter: Half-Blood Prince
    "tt0304141",  # Harry Potter: Prisoner of Azkaban
    "tt0330373",  # Harry Potter: Goblet of Fire
    "tt8579674",  # 1917
    "tt1745960",  # Top Gun: Maverick
    "tt0111161",  # Shawshank (dupe guard)
    "tt10872600", # Spider-Man: No Way Home
    "tt9362722",  # Spider-Man: Across the Spider-Verse
    "tt1517268",  # Barbie
    "tt15398776", # Oppenheimer
    "tt14209916", # Everything Everywhere All at Once
    # Recent (2023-2025)
    "tt15239678", # Dune: Part Two
    "tt11866324", # Poor Things
    "tt14230458", # Anatomy of a Fall
    "tt17351924", # Saltburn
    "tt9764362",  # The Menu
    "tt13238346", # Past Lives
    "tt5537002",  # Killers of the Flower Moon
    "tt6263850",  # The Fabelmans
    "tt14039582", # Drive My Car
    "tt10366206", # John Wick: Chapter 4
    "tt15398776", # Oppenheimer (dupe guard)
    "tt21692408", # Society of the Snow
    "tt12037194", # The Whale
    "tt9603212",  # Mission: Impossible – Dead Reckoning
    "tt6710474",  # Everything Everywhere (dupe guard)
    "tt11304740", # The Holdovers
    "tt14849194", # Bottoms
    "tt21235248", # Godzilla Minus One
    "tt5090568",  # Tar
    "tt14444726", # The Banshees of Inisherin
    "tt13655120", # The Iron Claw
    "tt12789558", # Priscilla
    "tt15314262", # All Quiet on the Western Front (2022)
    "tt6856396",  # The Boy and the Heron
    "tt2906216",  # Dungeons & Dragons: Honor Among Thieves
    "tt1517268",  # Barbie (dupe guard)
    "tt14998742", # Challengers
    "tt28015403", # The Brutalist
    "tt23849204", # A Real Pain
    "tt12747748", # The Substance
    "tt28607951", # Conclave
    "tt13651794", # Nosferatu (2024)
    "tt14471268", # Anora
]


LOCAL_SEED_BY_ID = {m["imdb_id"]: m for m in LOCAL_SEED_MOVIES}

@app.post("/api/seed")
async def seed_catalog():
    """Seed the catalog with curated iconic movies. Falls back to local data if OMDb fails."""
    added = []
    skipped = []
    failed = []
    for imdb_id in SEED_MOVIES:
        if imdb_id in engine.catalog:
            skipped.append(imdb_id)
            continue
        movie = None
        # Try OMDb first
        try:
            movie = await omdb.get_movie(imdb_id)
        except Exception:
            pass
        # Fall back to local seed data
        if not movie and imdb_id in LOCAL_SEED_BY_ID:
            movie = LOCAL_SEED_BY_ID[imdb_id]
        if movie:
            engine.add_to_catalog(movie)
            added.append({"imdb_id": imdb_id, "title": movie["title"]})
        else:
            failed.append(imdb_id)
    return {"added": len(added), "skipped": len(skipped), "failed": len(failed), "movies": added}


@app.post("/api/seed/local")
async def seed_local():
    """Seed catalog from local curated data (no API key needed)."""
    added = []
    skipped = []
    for movie in LOCAL_SEED_MOVIES:
        if movie["imdb_id"] in engine.catalog:
            skipped.append(movie["imdb_id"])
            continue
        engine.add_to_catalog(movie)
        added.append({"imdb_id": movie["imdb_id"], "title": movie["title"]})
    return {"added": len(added), "skipped": len(skipped), "movies": added}


@app.post("/api/demo/coldstart")
async def demo_coldstart():
    result = await engine.demo_coldstart()
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result
