# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "219acfff-2b4c-45c2-8797-a0e3cf610899",
# META       "default_lakehouse_name": "spotify_lh",
# META       "default_lakehouse_workspace_id": "f51957ad-15f3-436a-965d-78ade7c26fc4",
# META       "known_lakehouses": [
# META         {
# META           "id": "219acfff-2b4c-45c2-8797-a0e3cf610899"
# META         }
# META       ]
# META     },
# META     "environment": {
# META       "environmentId": "9fcb0db4-9b95-4e83-9d59-e057ffecd463",
# META       "workspaceId": "8fdc525a-1dd0-4926-acd9-9fece50c1fd1"
# META     }
# META   }
# META }

# CELL ********************

## Cell 1 — Setup & Download
%pip install kagglehub -q

import os
import notebookutils
import kagglehub

KV_NAME = 'https://kv-fabricdemo-hs.vault.azure.net/'

os.environ['KAGGLE_API_TOKEN'] = notebookutils.credentials.getSecret(KV_NAME, 'kaggle-token')

# Download both datasets
print('Downloading Spotify Tracks...')
path_tracks = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
print(f'Tracks: {path_tracks}')

print('Downloading Spotify Charts...')
path_charts = kagglehub.dataset_download("dhruvildave/spotify-charts")
print(f'Charts: {path_charts}')

print('Done ✓')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Cell 1b — Load tracks from CSV
import pandas as pd
import os, notebookutils, kagglehub

KV_NAME = 'https://kv-fabricdemo-hs.vault.azure.net/'
os.environ['KAGGLE_API_TOKEN'] = notebookutils.credentials.getSecret(KV_NAME, 'kaggle-token')

path_tracks = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")

tracks = pd.read_csv(f'{path_tracks}/dataset.csv')
print(f'Tracks loaded: {tracks.shape}')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Cell 2 — Deduplicate + build clean track foundation
import numpy as np
np.random.seed(42)

# Deduplicate — same track appears under multiple genres, audio features identical
tracks_dedup = (
    tracks
    .sort_values('popularity', ascending=False)
    .drop_duplicates(subset='track_id', keep='first')
    .reset_index(drop=True)
)

# Take top 500 by popularity — all real, all recognizable
top_tracks = tracks_dedup.nlargest(500, 'popularity')[
    ['track_id','track_name','artists','album_name','popularity',
     'track_genre','danceability','energy','valence','tempo',
     'acousticness','instrumentalness','liveness','speechiness',
     'duration_ms','explicit']
].reset_index(drop=True)

print(f'Unique tracks after dedup: {len(tracks_dedup):,}')
print(f'Top 500 unique tracks ready ✓')
print()
print(top_tracks[['track_name','artists','track_genre','popularity','danceability','energy']].head(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Cell 3 — Generate full demo dataset (real tracks + generated chart layer)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

REGIONS = ['Germany','United Kingdom','France','Netherlands','Sweden',
           'Norway','Denmark','Austria','Switzerland','Spain']

REGION_WEIGHT = {
    'Germany':1.0,'United Kingdom':1.0,'France':0.9,'Netherlands':0.6,
    'Sweden':0.7,'Norway':0.5,'Denmark':0.5,'Austria':0.4,
    'Switzerland':0.5,'Spain':0.8
}

# ── TABLE 1: ARTISTS ─────────────────────────────────────────────
artist_rows = []
for _, row in top_tracks.iterrows():
    for name in row['artists'].split(';'):
        artist_rows.append({'artist_name': name.strip(),
                            'genre': row['track_genre'],
                            'popularity': row['popularity']})

artists_raw = pd.DataFrame(artist_rows)
artists = (artists_raw.groupby('artist_name')
           .agg(primary_genre=('genre', lambda x: x.mode()[0]),
                avg_popularity=('popularity','mean'),
                track_count=('popularity','count'))
           .reset_index())

n = len(artists)
artists['artist_id'] = ['A' + str(i+1).zfill(4) for i in range(n)]
artists['monthly_listeners'] = (artists['avg_popularity']
    .apply(lambda p: int(np.random.lognormal(np.log(p * 120000), 0.4)))).clip(lower=10000)
artists['followers'] = (artists['monthly_listeners'] * np.random.uniform(0.25, 0.65, n)).astype(int)
artists['is_verified'] = artists['avg_popularity'] > 80
artists['latest_release_year'] = np.random.choice(
    [2021,2022,2023,2024,2025], n, p=[0.08,0.15,0.27,0.32,0.18])
artists['latest_release_month'] = np.random.randint(1, 13, n)
artists = artists[['artist_id','artist_name','primary_genre',
                    'monthly_listeners','followers','is_verified',
                    'avg_popularity','latest_release_year','latest_release_month']]

print(f'Artists table: {len(artists)} rows')

# ── TABLE 2: TRACKS (real audio features + artist_id FK) ─────────
# Map primary artist to artist_id
artist_lookup = dict(zip(artists['artist_name'], artists['artist_id']))
top_tracks['primary_artist'] = top_tracks['artists'].str.split(';').str[0].str.strip()
top_tracks['artist_id'] = top_tracks['primary_artist'].map(artist_lookup)
tracks_clean = top_tracks[['track_id','track_name','primary_artist','artist_id',
                             'album_name','track_genre','popularity','duration_ms',
                             'explicit','danceability','energy','valence',
                             'tempo','acousticness','instrumentalness',
                             'liveness','speechiness']].copy()
print(f'Tracks table: {len(tracks_clean)} rows')

# ── TABLE 3: CHART HISTORY (generated, 52 weeks × 10 regions) ────
chart_rows = []
base_date = datetime(2025, 7, 7)  # most recent Monday
weeks = [base_date - timedelta(weeks=w) for w in range(51, -1, -1)]

for region in REGIONS:
    weight = REGION_WEIGHT[region]
    for week in weeks:
        # Sample 50 tracks per region per week, weighted by popularity
        probs = top_tracks['popularity'].values.astype(float)
        probs += np.random.uniform(0, 15, len(probs))  # add noise
        probs = probs / probs.sum()
        sampled = top_tracks.sample(50, weights=probs, replace=False)
        for rank, (_, t) in enumerate(sampled.iterrows(), 1):
            base_streams = int(weight * (51 - rank) * np.random.uniform(8000, 25000))
            trend = np.random.choice(
                ['MOVE_UP','MOVE_DOWN','SAME_POSITION','NEW_ENTRY'],
                p=[0.30, 0.30, 0.25, 0.15])
            chart_rows.append({
                'track_id':    t['track_id'],
                'region':      region,
                'week_date':   week.strftime('%Y-%m-%d'),
                'rank':        rank,
                'streams':     base_streams,
                'trend':       trend
            })

chart_history = pd.DataFrame(chart_rows)
print(f'Chart history table: {len(chart_history):,} rows')

# ── TABLE 4: STREAMING EVENTS (real-time layer, last 7 days) ─────
event_rows = []
now = datetime.now()
for i in range(2000):
    t = top_tracks.sample(1, weights=top_tracks['popularity']).iloc[0]
    region = np.random.choice(REGIONS, p=[0.18,0.18,0.14,0.08,0.08,
                                           0.06,0.06,0.05,0.06,0.11])
    ts = now - timedelta(minutes=np.random.randint(0, 10080))
    event_rows.append({
        'event_id':   f'EVT{i+1:06d}',
        'track_id':   t['track_id'],
        'region':     region,
        'timestamp':  ts.strftime('%Y-%m-%d %H:%M:%S'),
        'stream_count': np.random.randint(1000, 50000),
        'source':     np.random.choice(['mobile','desktop','smart_speaker'],
                                        p=[0.60,0.25,0.15])
    })

streaming_events = pd.DataFrame(event_rows)
print(f'Streaming events table: {len(streaming_events):,} rows')
print()
print('All tables generated ✓')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Cell 4 — Write to Lakehouse (Bronze layer)
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

tables = {
    'bronze_artists':          artists,
    'bronze_tracks':           tracks_clean,
    'bronze_chart_history':    chart_history,
    'bronze_streaming_events': streaming_events
}

for table_name, df in tables.items():
    sdf = spark.createDataFrame(df)
    sdf.write.mode('overwrite').format('delta').saveAsTable(table_name)
    print(f'✓ {table_name} saved ({len(df):,} rows)')

print('\nBronze layer complete ✓')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

## Cell 5 — Silver layer (clean + enrich)
from pyspark.sql import functions as F
from pyspark.sql.types import DateType

# ── SILVER ARTISTS ────────────────────────────────────────────────
silver_artists = spark.table('bronze_artists') \
    .withColumn('latest_release_date',
        F.to_date(F.concat_ws('-',
            F.col('latest_release_year'),
            F.lpad(F.col('latest_release_month').cast('string'), 2, '0'),
            F.lit('01')), 'yyyy-MM-dd')) \
    .withColumn('months_since_release',
        F.round(F.datediff(F.current_date(), F.col('latest_release_date')) / 30, 1)) \
    .withColumn('listener_tier',
        F.when(F.col('monthly_listeners') > 10_000_000, 'Mega')
         .when(F.col('monthly_listeners') > 1_000_000, 'Major')
         .when(F.col('monthly_listeners') > 100_000, 'Mid')
         .otherwise('Indie')) \
    .drop('latest_release_year', 'latest_release_month')

silver_artists.write.mode('overwrite').format('delta').saveAsTable('silver_artists')
print(f'✓ silver_artists ({silver_artists.count()} rows)')

# ── SILVER TRACKS ─────────────────────────────────────────────────
silver_tracks = spark.table('bronze_tracks') \
    .withColumn('duration_min', F.round(F.col('duration_ms') / 60000, 2)) \
    .withColumn('energy_level',
        F.when(F.col('energy') > 0.7, 'High')
         .when(F.col('energy') > 0.4, 'Medium')
         .otherwise('Low')) \
    .withColumn('mood',
        F.when((F.col('valence') > 0.6) & (F.col('energy') > 0.6), 'Happy')
         .when((F.col('valence') < 0.4) & (F.col('energy') > 0.6), 'Intense')
         .when((F.col('valence') > 0.6) & (F.col('energy') < 0.4), 'Peaceful')
         .otherwise('Melancholic')) \
    .drop('duration_ms')

silver_tracks.write.mode('overwrite').format('delta').saveAsTable('silver_tracks')
print(f'✓ silver_tracks ({silver_tracks.count()} rows)')

# ── SILVER CHART HISTORY ──────────────────────────────────────────
silver_chart_history = spark.table('bronze_chart_history') \
    .withColumn('week_date', F.to_date('week_date', 'yyyy-MM-dd')) \
    .withColumn('is_top10', F.col('rank') <= 10) \
    .withColumn('is_rising', F.col('trend').isin('MOVE_UP', 'NEW_ENTRY'))

silver_chart_history.write.mode('overwrite').format('delta').saveAsTable('silver_chart_history')
print(f'✓ silver_chart_history ({silver_chart_history.count()} rows)')

# ── SILVER STREAMING EVENTS ───────────────────────────────────────
silver_streaming_events = spark.table('bronze_streaming_events') \
    .withColumn('timestamp', F.to_timestamp('timestamp', 'yyyy-MM-dd HH:mm:ss')) \
    .withColumn('event_date', F.to_date('timestamp')) \
    .withColumn('hour_of_day', F.hour('timestamp'))

silver_streaming_events.write.mode('overwrite').format('delta').saveAsTable('silver_streaming_events')
print(f'✓ silver_streaming_events ({silver_streaming_events.count()} rows)')

print('\nSilver layer complete ✓')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ── Gold 1: artist_performance ──────────────────────────────────────────────
artist_perf = (
    spark.table('silver_chart_history')
    .join(spark.table('silver_tracks').select('track_id', 'artist_id'), on='track_id')
    .join(spark.table('silver_artists').select(
        'artist_id', 'artist_name', 'primary_genre',
        'listener_tier', 'months_since_release'), on='artist_id')
    .groupBy('artist_id', 'artist_name', 'primary_genre', 'listener_tier', 'months_since_release')
    .agg(
        F.sum('streams').alias('total_streams'),
        F.avg('rank').alias('avg_chart_rank'),
        F.count('*').alias('chart_appearances'),
        F.sum(F.when(F.col('is_rising'), 1).otherwise(0)).alias('weeks_rising'),
        F.sum(F.when(F.col('is_top10'), 1).otherwise(0)).alias('weeks_top10'),
    )
    .withColumn('momentum_score',
        F.round(F.col('weeks_rising') / F.col('chart_appearances') * 100, 1))
)
artist_perf.write.mode('overwrite').format('delta').saveAsTable('gold_artist_performance')
print(f'✓ gold_artist_performance ({artist_perf.count():,} rows)')

# ── Gold 2: genre_trends (last 4 week_dates) ────────────────────────────────
ch = spark.table('silver_chart_history')
last4_dates = [r.week_date for r in ch.select('week_date').distinct()
               .orderBy(F.col('week_date').desc()).limit(4).collect()]

genre_trends = (
    ch.filter(F.col('week_date').isin(last4_dates))
    .join(spark.table('silver_tracks').select('track_id', 'track_genre'), on='track_id')
    .groupBy('track_genre', 'region', 'week_date')
    .agg(
        F.avg('rank').alias('avg_rank'),
        F.sum('streams').alias('total_streams'),
        F.count('track_id').alias('charting_tracks'),
        F.sum(F.when(F.col('is_rising'), 1).otherwise(0)).alias('rising_tracks'),
    )
    .orderBy('track_genre', 'region', 'week_date')
)
genre_trends.write.mode('overwrite').format('delta').saveAsTable('gold_genre_trends')
print(f'✓ gold_genre_trends ({genre_trends.count():,} rows)')

# ── Gold 3: weekly_rankings (latest week_date) ──────────────────────────────
ch = spark.table('silver_chart_history')
latest_date = ch.agg(F.max('week_date')).collect()[0][0]

weekly_rankings = (
    ch.filter(F.col('week_date') == latest_date)
    .join(spark.table('silver_tracks').select(
        'track_id', 'track_name', 'primary_artist', 'track_genre',
        'danceability', 'energy', 'valence', 'mood', 'energy_level'), on='track_id')
    .select('region', 'rank', 'track_id', 'track_name', 'primary_artist',
            'track_genre', 'streams', 'trend', 'is_top10', 'is_rising',
            'danceability', 'energy', 'valence', 'mood', 'energy_level')
    .orderBy('region', 'rank')
)
weekly_rankings.write.mode('overwrite').format('delta').saveAsTable('gold_weekly_rankings')
print(f'✓ gold_weekly_rankings ({weekly_rankings.count():,} rows)')

# ── Gold 4: rising_tracks — demo killer query ────────────────────────────────
rising_tracks = (
    ch.filter(F.col('is_rising') == True)
    .join(spark.table('silver_tracks').select(
        'track_id', 'track_name', 'primary_artist', 'artist_id',
        'track_genre', 'danceability', 'energy', 'valence', 'mood'), on='track_id')
    .join(spark.table('silver_artists').select(
        'artist_id', 'artist_name', 'listener_tier', 'months_since_release'), on='artist_id')
    .filter(F.col('danceability') >= 0.7)
    .filter(F.col('months_since_release') >= 12)
    .groupBy('track_id', 'track_name', 'primary_artist', 'artist_name',
              'track_genre', 'danceability', 'energy', 'valence', 'mood',
              'listener_tier', 'months_since_release')
    .agg(
        F.count('region').alias('regions_charting'),
        F.avg('rank').alias('avg_rank'),
        F.sum('streams').alias('total_streams'),
        F.sum(F.when(F.col('is_top10'), 1).otherwise(0)).alias('top10_regions'),
    )
    .filter(F.col('regions_charting') >= 3)
    .orderBy(F.col('regions_charting').desc(), F.col('avg_rank').asc())
)
rising_tracks.write.mode('overwrite').format('delta').saveAsTable('gold_rising_tracks')
print(f'✓ gold_rising_tracks ({rising_tracks.count():,} rows)')

print('\nGold layer complete ✓')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

%pip install azure-kusto-data azure-kusto-ingest -q

import time, uuid, random
from datetime import datetime, timezone
import pandas as pd
import notebookutils
from azure.kusto.ingest import QueuedIngestClient, IngestionProperties
from azure.kusto.data import KustoConnectionStringBuilder

# ── Config ───────────────────────────────────────────────────────────────────
KUSTO_URI   = 'https://trd-b7df3megvv793a2c6s.z6.kusto.fabric.microsoft.com'
INGEST_URI  = 'https://ingest-trd-b7df3megvv793a2c6s.z6.kusto.fabric.microsoft.com'
DATABASE    = 'spotify_eh'
TABLE       = 'streaming_events'

# Auth via Fabric token
token_fn = lambda: notebookutils.credentials.getToken('https://kusto.kusto.windows.net')
kcsb = KustoConnectionStringBuilder.with_token_provider(INGEST_URI, token_fn)
ingest_client = QueuedIngestClient(kcsb)
ingestion_props = IngestionProperties(database=DATABASE, table=TABLE)

# Load real tracks from Gold table
tracks_df = spark.table('gold_weekly_rankings') \
    .select('track_id', 'track_name', 'primary_artist') \
    .toPandas()

REGIONS     = ['Germany','United Kingdom','France','Netherlands',
               'Sweden','Norway','Denmark','Austria','Switzerland','Spain']
PLATFORMS   = ['mobile', 'desktop', 'tablet']
EVENT_TYPES = ['stream', 'stream', 'stream', 'skip', 'add_to_playlist', 'share']

def generate_event(row):
    completed = random.random() > 0.2
    return {
        'event_id':            str(uuid.uuid4()),
        'track_id':            row.track_id,
        'track_name':          row.track_name,
        'artist_name':         row.primary_artist,
        'region':              random.choice(REGIONS),
        'event_type':          random.choice(EVENT_TYPES),
        'platform':            random.choice(PLATFORMS),
        'event_timestamp':     datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'stream_duration_sec': random.randint(30, 210) if completed else random.randint(5, 29),
        'completed':           completed,
    }

# ── Stream loop ───────────────────────────────────────────────────────────────
print('Streaming events into Eventhouse... (stop cell to pause)')
batch_num = 0
while True:
    sample = tracks_df.sample(n=20, replace=True)
    events = pd.DataFrame([generate_event(r) for r in sample.itertuples()])
    ingest_client.ingest_from_dataframe(events, ingestion_properties=ingestion_props)
    batch_num += 1
    print(f'Batch {batch_num}: 20 events pushed — {datetime.now().strftime("%H:%M:%S")}')
    time.sleep(10)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
