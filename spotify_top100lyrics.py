import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
import lyricsgenius
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import datetime
from datetime import datetime
import time

analyzer = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# ----------------------------------------
# 🔐 Spotify API Setup (Update with your credentials)
# ----------------------------------------
load_dotenv(dotenv_path='spotifycred.env')  # Load values from .env into environment


SPOTIFY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

# Setup Spotify client
sp = Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope="user-top-read"
))

# Setup Genius client
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], timeout=15, retries=3)
genius.verbose = False

# Step 1: Get Top 100 Tracks
def get_top_tracks(limit=100, time_range="medium_term"):
    tracks = []
    for offset in range(0, limit, 50):
        batch = sp.current_user_top_tracks(limit=50, time_range=time_range, offset=offset)
        tracks.extend(batch['items'])
    print(f"🎵 Retrieved {len(tracks)} top tracks.")
    return tracks

# Step 2: Fetch lyrics and analyze sentiment
def analyze_lyrics(tracks):
    results = []
    for i, track in enumerate(tracks, 1):
        title = track['name']
        artist = track['artists'][0]['name']
        print(f"🎤 ({i}/{len(tracks)}) Searching lyrics for: {title} - {artist}")

        try:
            song = genius.search_song(title, artist)
            if song and song.lyrics:
                lyrics = song.lyrics

                # VADER Sentiment
                sentiment = analyzer.polarity_scores(lyrics)
                results.append({
                    "track": title,
                    "artist": artist,
                    "lyrics": lyrics,
                    "compound": sentiment['compound'],
                    "positive": sentiment['pos'],
                    "neutral": sentiment['neu'],
                    "negative": sentiment['neg']
                })
            else:
                print(f"❌ No lyrics found for {title}")
        except Exception as e:
            print(f"⚠️ Error for {title} by {artist}: {e}")

        time.sleep(1)  # Respect Genius rate limit

    return pd.DataFrame(results)

# Step 3: Plot sentiment (placeholder)
def plot_sentiment(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['compound'], df['positive'], alpha=0.7)
    plt.title('VADER Sentiment Analysis of Top 100 Spotify Songs')
    plt.xlabel('Compound Score')
    plt.ylabel('Positive Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("vader_sentiment_plot.png")
    plt.show()


# Step 4: Generate Word Cloud colored by sentiment
def generate_sentiment_wordcloud(df):
    if df.empty or 'lyrics' not in df.columns:
        print("⚠️ No lyrics found for word cloud.")
        return

    all_lyrics = " ".join(df['lyrics'].dropna().tolist())
    words = all_lyrics.split()
    words = [word for word in words if word.lower() not in stop_words and word.isalpha()]

    # Get VADER sentiment for each unique word
    word_sentiment = {}
    for word in set(words):
        sentiment = analyzer.polarity_scores(word)['compound']
        word_sentiment[word] = sentiment

    # Color function for WordCloud
    def color_func(word, **kwargs):
        score = word_sentiment.get(word, 0)
        if score >= 0.2:
            return "green"
        elif score <= -0.2:
            return "red"
        else:
            return "gray"

    # Generate word cloud
    wordcloud = WordCloud(
        width=1000,
        height=600,
        background_color='white',
        max_words=200
    ).generate(" ".join(words))

    # Plot with custom coloring
    plt.figure(figsize=(12, 7))
    plt.imshow(wordcloud.recolor(color_func=color_func), interpolation='bilinear')
    plt.axis('off')
    plt.title("🎶 Word Cloud Colored by Sentiment", fontsize=16)
    plt.tight_layout()
    plt.savefig("wordcloud_sentiment.png")
    plt.show()

def classify_emotion(row):
    compound = row['compound']
    pos = row['positive']
    neg = row['negative']
    neu = row['neutral']

    if compound >= 0.5:
        return 'joy'
    elif compound <= -0.5:
        return 'anger'
    elif pos > 0.5 and neu < 0.2:
        return 'love'
    elif neg > 0.5:
        return 'sadness'
    elif neu > 0.7:
        return 'calm'
    else:
        return 'mixed'

def generate_mood_wheel(df):
    if df.empty:
        print("⚠️ No data available for mood wheel.")
        return

    # Classify emotion for each song
    df['emotion'] = df.apply(classify_emotion, axis=1)

    # Count frequency of each emotion
    mood_counts = df['emotion'].value_counts()
    labels = mood_counts.index.tolist()
    sizes = mood_counts.values

    # Plot as radial bar chart
    angles = [n / float(len(labels)) * 2 * 3.1416 for n in range(len(labels))]
    angles += angles[:1]  # close the circle
    sizes = list(sizes) + [sizes[0]]  # close the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(3.1416 / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, sizes, linewidth=2, linestyle='solid')
    ax.fill(angles, sizes, alpha=0.4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    ax.set_title("🌀 Mood Wheel – Emotional Profile of Your Top Tracks", fontsize=14)
    plt.tight_layout()
    plt.savefig("mood_wheel.png")
    plt.show()

def extract_genres(tracks):
    genre_list = []

    for i, track in enumerate(tracks, 1):
        try:
            artist_id = track['artists'][0]['id']
            artist_info = sp.artist(artist_id)
            genres = artist_info.get('genres', [])
            genre_list.extend(genres)
            print(f"🎸 ({i}/{len(tracks)}) {track['name']} - Genres: {genres}")
            time.sleep(0.3)  # avoid hitting rate limits
        except Exception as e:
            print(f"⚠️ Error fetching genres for {track['name']}: {e}")

    return genre_list

def generate_genre_wheel(genre_list):
    if not genre_list:
        print("⚠️ No genres available to plot.")
        return

    top_genres = Counter(genre_list).most_common(10)  # adjust to show more or fewer
    labels, counts = zip(*top_genres)

    angles = [n / float(len(labels)) * 2 * 3.1416 for n in range(len(labels))]
    angles += angles[:1]
    counts = list(counts) + [counts[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(3.1416 / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, counts, linewidth=2)
    ax.fill(angles, counts, alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)

    ax.set_title("🎧 Genre Wheel – Your Top Spotify Genres", fontsize=14)
    plt.tight_layout()
    plt.savefig("genre_wheel.png")
    plt.show()

#########Interactive_Viz Test
# Get Spotify tracks (already filtered to medium_term)


# Helper to fetch lyrics from Genius
def get_lyrics(title, artist):
    try:
        song = genius.search_song(title, artist)
        if song and song.lyrics:
            return song.lyrics
    except Exception as e:
        print(f"⚠️ Genius error for '{title}' by '{artist}': {e}")
    return ""

# Helper to map compound sentiment to a color
def sentiment_to_color(compound):
    if compound >= 0.5:
        return "green"
    elif compound <= -0.5:
        return "red"
    else:
        return "gray"





# Main pipeline
def main():
     print("🚀 Starting sentiment analysis...")
     tracks = get_top_tracks()
     df = analyze_lyrics(tracks)
    
     if not df.empty:
         df.to_csv("top_100_lyrics_vader_sentiment.csv", index=False)
         print("📁 CSV Exported: top_100_lyrics_vader_sentiment.csv")
         plot_sentiment(df)
         generate_sentiment_wordcloud(df)
         generate_mood_wheel(df)
         genre_list = extract_genres(tracks)
         generate_genre_wheel(genre_list)
     else:
         print("⚠️ No lyrics found for any tracks.")

    

if __name__ == "__main__":
    main()