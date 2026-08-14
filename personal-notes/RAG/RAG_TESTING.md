# Specific RAG flows, problems and hypotesis/solutions

## Flow Example 1 - fast-paced competitive shooter

Query: `"I want a fast-paced multiplayer shooter, something competitive"`

```python
client_ = chromadb.PersistentClient(path=Path(__file__).parent / "data" / "chroma_db")
collection = client_.get_collection("steam_games")
model = SentenceTransformer("all-MiniLM-L6-v2")

user_query = "I want a fast-paced multiplayer shooter, something competitive"
chunks = retrieve(user_query, model, collection, 10)

for i, chunk in enumerate(chunks):
    similarity = f"{chunk['score']:.3}"
    print(f"{i}: {chunk['name']}, score: {similarity}")
```

Retrieved chunks (n=10):
0: Garry's Mod, score: 0.54
1: Destiny 2, score: 0.539
2: Fallout 4, score: 0.517
3: Garry's Mod, score: 0.469
4: Farlight 84, score: 0.463
5: Unturned, score: 0.462
6: Team Fortress 2, score: 0.451
7: Counter-Strike: Source, score: 0.448
8: Portal 2, score: 0.444
9: Titanfall® 2, score: 0.435

Notable: Garry's Mod (a physics sandbox with no combat focus) ranked #1.
Counter-Strike 2 and Apex Legends - the strongest actual matches in the
corpus - didn't appear in the top 10 at all.

LLM Response:
> Try Counter-Strike: Source – it's a fast-paced, competitive online
> shooter built on the Source engine with team-based action and constant
> updates. If you prefer a hero-shooter experience, Titanfall 2 also
> offers rapid-fire multiplayer with new titans and pilot abilities.
> Both fit the "fast-paced, competitive" description you're looking for.

Verdict: the LLM picked the two best options actually available in a
flawed context (retrieval failure), and reasonably ignored the clearly
irrelevant ones (Garry's Mod, Fallout 4, Portal 2). Generation did its
job well despite bad input.

## Flow Example 2 - relaxing farming/life sim

Query: `"relaxing farming and life simulation game"`, n=3:
0: The Forest, score: 0.545
1: Stardew Valley, score: 0.487
2: DayZ, score: 0.469

LLM correctly picked only Stardew Valley.

Same query, n=7 (top 3 unchanged, Stardew Valley also reappears at #6):

0: The Forest, score: 0.545
1: Stardew Valley, score: 0.487
2: DayZ, score: 0.469
3: Unturned, score: 0.457
4: 7 Days to Die, score: 0.451
5: 7 Days to Die, score: 0.44
6: Stardew Valley, score: 0.436

LLM response unchanged: "Stardew Valley."

Worst signal-to-noise ratio observed so far - 3 of top 3 slots (and 4 of
top 7) went to survival games, not life-sim games. No cross-mention or
oversized-chunk explanation here - pure vocabulary overlap (see failure
mode 3 above).

## Flow Example 3 - soulslike combat

Query: `"dark souls-like game with challenging combat and bosses"`, n=3:

0: Sekiro™: Shadows Die Twice - GOTY Edition, score: 0.546
1: The Elder Scrolls® Online, score: 0.514
2: DARK SOULS™: REMASTERED, score: 0.509

LLM correctly identified Dark Souls and Sekiro as matches, and explicitly
excluded The Elder Scrolls Online ("an MMO RPG, not characterized as a
dark souls-like").

Best signal-to-noise ratio observed so far - only 1 of 3 was off-topic,
and it was still directionally close (RPG, not a random genre). Supports
the hypothesis that distinctive vocabulary ("souls-like", "bosses")
retrieves more cleanly than generic vocabulary.

## Flow Example 4 - open world with dragons and magic

Query: `"open world game with dragons and magic"`, n=3:

0: The Elder Scrolls V: Skyrim, score: 0.512
1: The Elder Scrolls V: Skyrim, score: 0.504
2: Dragon's Dogma 2, score: 0.483

LLM correctly named both as valid open-world fantasy matches.

Not actually a case of "generic wins" - both retrieved games are genuine,
correct matches for the query (the corpus simply had two good answers).
Good example of retrieval working as intended when the query has more
than one legitimate match.