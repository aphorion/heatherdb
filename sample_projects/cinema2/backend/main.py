"""Cinema2 — Netflix-like Movie Recommendations powered by HeatherDB."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

from heather import HeatherClient
from omdb import OMDbClient
from engine import CinemaEngine
from seed_data import SEED_MOVIES as LOCAL_SEED_MOVIES
import db

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


app = FastAPI(title="Cinema2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Models ---

class WatchRequest(BaseModel):
    imdb_id: str

class BlendRequest(BaseModel):
    profiles: list[str]


# --- Health ---

@app.get("/api/health")
async def health():
    heather_ok = await heather.health()
    return {"status": "ok", "heather_ok": heather_ok}


# --- Seed ---

LOCAL_SEED_BY_ID = {m["imdb_id"]: m for m in LOCAL_SEED_MOVIES}

# 200 modern movies (2010+) to fetch from OMDb — no classics
SEED_IDS = [
    # --- 2024-2025 ---
    "tt13651794",  # Nosferatu (2024)
    "tt14471268",  # Anora
    "tt28607951",  # Conclave
    "tt12747748",  # The Substance
    "tt28015403",  # The Brutalist
    "tt23849204",  # A Real Pain
    "tt14998742",  # Challengers
    "tt15239678",  # Dune: Part Two
    "tt21235248",  # Godzilla Minus One
    "tt26446885",  # Wicked
    "tt11304740",  # The Holdovers
    "tt14849194",  # Bottoms
    "tt18259086",  # Furiosa
    "tt6263850",   # The Fabelmans
    "tt14230458",  # Anatomy of a Fall
    "tt27503384",  # A Complete Unknown
    "tt31193782",  # The Wild Robot
    "tt12263384",  # Alien: Romulus
    "tt23468450",  # Inside Out 2
    "tt15009428",  # Gladiator II
    # --- 2023 ---
    "tt15398776",  # Oppenheimer
    "tt1517268",   # Barbie
    "tt5537002",   # Killers of the Flower Moon
    "tt13238346",  # Past Lives
    "tt17351924",  # Saltburn
    "tt11866324",  # Poor Things
    "tt21692408",  # Society of the Snow
    "tt13655120",  # The Iron Claw
    "tt10366206",  # John Wick: Chapter 4
    "tt9362722",   # Spider-Man: Across the Spider-Verse
    "tt6856396",   # The Boy and the Heron
    "tt9603212",   # Mission: Impossible - Dead Reckoning
    "tt14849194",  # Bottoms
    "tt12037194",  # The Whale
    "tt5090568",   # Tár
    "tt14444726",  # The Banshees of Inisherin
    "tt9764362",   # The Menu
    "tt15314262",  # All Quiet on the Western Front (2022)
    "tt2906216",   # Dungeons & Dragons: Honor Among Thieves
    "tt12789558",  # Priscilla
    # --- 2022 ---
    "tt14209916",  # Everything Everywhere All at Once
    "tt1745960",   # Top Gun: Maverick
    "tt10872600",  # Spider-Man: No Way Home
    "tt3704428",   # Elvis
    "tt7144666",   # The Northman
    "tt9032400",   # Eternals
    "tt10954984",  # Nope
    "tt11271038",  # Aftersun
    "tt14039582",  # Drive My Car
    "tt13320662",  # The Batman
    "tt10838180",  # Marcel the Shell with Shoes On
    "tt4244994",   # The Tragedy of Macbeth
    "tt6710474",   # Everything Everywhere (alt)
    "tt8093700",   # The Woman King
    "tt10954652",  # Bones and All
    "tt12593682",  # Bullet Train
    "tt11138512",  # The Lost City
    "tt9900782",   # Bros
    "tt14444726",  # Banshees of Inisherin
    "tt11564570",  # Glass Onion
    # --- 2021 ---
    "tt1160419",   # Dune
    "tt3581920",   # The Last Duel
    "tt2382320",   # No Time to Die
    "tt10293406",  # The Power of the Dog
    "tt7740496",   # Nightmare Alley
    "tt9770150",   # Shang-Chi
    "tt11286314",  # Don't Look Up
    "tt9376612",   # Shang-Chi
    "tt6264654",   # Free Guy
    "tt3228774",   # Cruella
    "tt9032400",   # Eternals
    "tt12801262",  # Luca
    "tt4244994",   # Macbeth
    "tt1464335",   # Jungle Cruise
    "tt10648342",  # Thor: Love and Thunder
    "tt2953050",   # Encanto
    "tt10886166",  # tick, tick... BOOM!
    "tt9204128",   # CODA
    "tt7657566",   # The Suicide Squad
    "tt12361974",  # The Mitchells vs the Machines
    # --- 2020 ---
    "tt6723592",   # Tenet
    "tt3661210",   # The Trial of the Chicago 7
    "tt7131622",   # Another Round
    "tt6966692",   # Green Book
    "tt10272386",  # The Father
    "tt8368406",   # Mank
    "tt4995540",   # Minari
    "tt9419884",   # Doctor Strange: Multiverse of Madness
    "tt10199590",  # Werewolves Within
    "tt7395114",   # Sound of Metal
    "tt11555492",  # Promising Young Woman
    "tt6334354",   # Nomadland
    "tt9032400",   # Eternals
    "tt9893250",   # Soul (Pixar)
    "tt7979580",   # The Gentlemen
    "tt9620292",   # Onward
    "tt10539608",  # The Invisible Man (2020)
    "tt8946378",   # Knives Out
    # --- 2019 ---
    "tt7286456",   # Joker
    "tt6751668",   # Parasite
    "tt8579674",   # 1917
    "tt1950186",   # Ford v Ferrari
    "tt2584384",   # Jojo Rabbit
    "tt7653254",   # Marriage Story
    "tt1302006",   # The Irishman
    "tt7984734",   # The Lighthouse
    "tt4154796",   # Avengers: Endgame
    "tt6105098",   # The Two Popes
    "tt5052448",   # Get Out (2017)
    "tt7349950",   # It Chapter Two
    "tt1979376",   # Toy Story 4
    "tt4633694",   # Spider-Man: Into the Spider-Verse (2018)
    "tt1477834",   # Aquaman
    "tt6857112",   # Us
    "tt5580390",   # The Shape of Water
    "tt4925292",   # Booksmart
    "tt7131622",   # Another Round
    "tt3170832",   # Room
    # --- 2018 ---
    "tt5463162",   # Deadpool 2
    "tt1825683",   # Black Panther
    "tt5027774",   # Three Billboards Outside Ebbing, Missouri
    "tt5580390",   # The Shape of Water
    "tt4154756",   # Avengers: Infinity War
    "tt6966692",   # Green Book
    "tt3606756",   # Incredibles 2
    "tt7784604",   # Hereditary
    "tt5052448",   # Get Out
    "tt5726616",   # Call Me by Your Name
    "tt2584384",   # Jojo Rabbit
    "tt4034228",   # Manchester by the Sea
    "tt1856101",   # Blade Runner 2049
    "tt1950186",   # Ford v Ferrari
    "tt3315342",   # Logan
    "tt5013056",   # Dunkirk
    "tt1396484",   # It (2017)
    "tt4236770",   # Yellowstone
    "tt2380307",   # Coco
    "tt2096673",   # Inside Out
    # --- 2017-2015 ---
    "tt3501632",   # Thor: Ragnarok
    "tt1790809",   # Pirates: Dead Men Tell No Tales
    "tt3896198",   # Guardians of the Galaxy Vol. 2
    "tt3783958",   # La La Land
    "tt2119532",   # Hacksaw Ridge
    "tt2084970",   # The Imitation Game
    "tt1392190",   # Mad Max: Fury Road
    "tt2488496",   # Star Wars: The Force Awakens
    "tt3659388",   # The Martian
    "tt3170832",   # Room
    "tt2267998",   # Gone Girl
    "tt2802144",   # Kingsman: The Secret Service
    "tt1663202",   # The Revenant
    "tt1853728",   # Django Unchained
    "tt1291584",   # Warrior
    "tt2948356",   # Zootopia
    "tt2543164",   # Arrival
    "tt1392190",   # Mad Max: Fury Road
    "tt3783958",   # La La Land
    "tt0816692",   # Interstellar
    # --- 2014-2010 ---
    "tt1375666",   # Inception
    "tt1130884",   # Shutter Island
    "tt0993846",   # The Wolf of Wall Street
    "tt1345836",   # The Dark Knight Rises
    "tt0848228",   # The Avengers (2012)
    "tt1853728",   # Django Unchained
    "tt1631867",   # Edge of Tomorrow
    "tt1392170",   # The Hunger Games
    "tt1201607",   # Harry Potter: Deathly Hallows 2
    "tt1570728",   # Skyfall
    "tt0470752",   # Ex Machina
    "tt2562232",   # Birdman
    "tt2380307",   # Coco
    "tt1981115",   # Thor: The Dark World
    "tt2015381",   # Guardians of the Galaxy
    "tt1856101",   # Blade Runner 2049
    "tt1049413",   # Up (2009)
    "tt0371746",   # Iron Man
    "tt1270798",   # Whiplash
    "tt2278388",   # The Grand Budapest Hotel
]

@app.post("/api/seed")
async def seed_catalog():
    """Seed catalog via OMDb (200+ movies) with SQLite cache."""
    added, skipped, failed = [], [], []
    to_add = []
    seen = set()
    for imdb_id in SEED_IDS:
        if imdb_id in seen or imdb_id in engine.catalog:
            skipped.append(imdb_id)
            continue
        seen.add(imdb_id)
        movie = None
        # Check SQLite cache first
        cached = db.get_omdb_cached(imdb_id)
        if cached:
            movie = cached
        elif OMDB_KEY:
            try:
                movie = await omdb.get_movie(imdb_id)
                if movie:
                    db.set_omdb_cached(imdb_id, movie)
            except Exception:
                pass
        # Fall back to local seed data
        if not movie and imdb_id in LOCAL_SEED_BY_ID:
            movie = LOCAL_SEED_BY_ID[imdb_id]
        if movie:
            to_add.append(movie)
            added.append({"imdb_id": imdb_id, "title": movie.get("title", "")})
        else:
            failed.append(imdb_id)
    # Batch add
    if to_add:
        engine.add_to_catalog_batch(to_add)
    return {"added": len(added), "skipped": len(skipped), "failed": len(failed), "movies": added}


@app.post("/api/seed/local")
async def seed_local():
    added, skipped = [], []
    to_add = []
    for movie in LOCAL_SEED_MOVIES:
        if movie["imdb_id"] in engine.catalog:
            skipped.append(movie["imdb_id"])
            continue
        to_add.append(movie)
        added.append({"imdb_id": movie["imdb_id"], "title": movie["title"]})
    if to_add:
        engine.add_to_catalog_batch(to_add)
    return {"added": len(added), "skipped": len(skipped), "movies": added}


# --- Catalog ---

@app.get("/api/catalog")
async def get_catalog():
    return {"movies": engine.get_catalog()}


@app.get("/api/search")
async def search_movies(q: str = Query(..., min_length=1)):
    if not OMDB_KEY:
        # Search local catalog
        q_lower = q.lower()
        results = [
            {k: v for k, v in m.items() if k != "embedding"}
            for m in engine.catalog.values()
            if q_lower in m.get("title", "").lower()
        ]
        return {"movies": results}
    try:
        results = await omdb.search(q)
    except ValueError as e:
        raise HTTPException(401, str(e))
    return {"movies": results}


@app.get("/api/movie/{imdb_id}")
async def get_movie(imdb_id: str):
    # Try catalog first
    if imdb_id in engine.catalog:
        m = engine.catalog[imdb_id]
        return {k: v for k, v in m.items() if k != "embedding"}
    # Fall back to OMDb
    if OMDB_KEY:
        try:
            movie = await omdb.get_movie(imdb_id)
            if movie:
                return movie
        except ValueError as e:
            raise HTTPException(401, str(e))
    raise HTTPException(404, "Movie not found")


@app.get("/api/movie/{imdb_id}/similar")
async def similar_movies(imdb_id: str, n: int = Query(default=10, ge=1, le=20)):
    results = engine.similar_movies(imdb_id, n=n)
    if not results and imdb_id not in engine.catalog:
        raise HTTPException(404, "Movie not in catalog")
    return {"movies": results}


# --- Profiles ---

@app.get("/api/profiles")
async def list_profiles():
    return {"profiles": engine.list_profiles()}


# Blend must be before {name} routes to avoid matching "blend" as a profile name
@app.post("/api/profiles/blend")
async def blend(req: BlendRequest):
    if len(req.profiles) != 2:
        raise HTTPException(400, "Exactly 2 profiles required")
    result = engine.blend_fingerprints(req.profiles[0], req.profiles[1])
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/api/profiles/{name}")
async def create_profile(name: str):
    engine._ensure_profile(name)
    return {"status": "created", "name": name}


@app.delete("/api/profiles/{name}")
async def delete_profile(name: str):
    ok = engine.delete_profile(name)
    if not ok:
        raise HTTPException(404, "Profile not found")
    return {"status": "deleted", "name": name}


@app.post("/api/profiles/{name}/watch")
async def watch_movie(name: str, req: WatchRequest):
    engine._ensure_profile(name)
    ok = await engine.watch_movie(name, req.imdb_id)
    if not ok:
        raise HTTPException(404, "Movie not in catalog")
    return {"status": "watched", "profile": name, "imdb_id": req.imdb_id}


@app.get("/api/profiles/{name}/history")
async def get_history(name: str):
    return {"movies": engine.get_history(name)}


# --- Recommendations ---

@app.get("/api/profiles/{name}/recommendations")
async def get_recommendations(name: str, n: int = Query(default=40, ge=1, le=50)):
    return engine.tiered_recommendations(name, n=n)


@app.get("/api/profiles/{name}/because-you-watched")
async def because_you_watched(name: str, max_rows: int = Query(default=5, ge=1, le=10)):
    rows = await engine.because_you_watched(name, max_rows=max_rows)
    return {"rows": rows}


@app.get("/api/profiles/{name}/wildcards")
async def wildcards(name: str, n: int = Query(default=10, ge=1, le=20)):
    return {"movies": engine.wildcards(name, n=n)}


@app.get("/api/profiles/{name}/continue-watching")
async def continue_watching(name: str, n: int = Query(default=10, ge=1, le=20)):
    return {"movies": engine.continue_watching(name, n=n)}


@app.get("/api/profiles/{name}/movie/{imdb_id}/why")
async def why_recommended(name: str, imdb_id: str):
    result = await engine.explain_recommendation(name, imdb_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


# --- Genre Browsing ---

@app.get("/api/genres")
async def list_genres():
    return {"genres": engine.available_genres()}


@app.get("/api/genres/{genre}")
async def genre_movies(
    genre: str,
    n: int = Query(default=20, ge=1, le=50),
    profile: str | None = Query(default=None),
):
    movies = engine.movies_by_genre(genre, n=n, profile=profile)
    return {"genre": genre, "movies": movies}


# --- Taste ---

@app.get("/api/profiles/{name}/taste")
async def taste_profile(name: str):
    return engine.get_taste_profile(name)
