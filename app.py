import pandas as pd
import webbrowser
from utils.detect_emotion import detect_emotion

def load_songs(emotion):
    try:
        if emotion == "happy":
            return pd.read_csv("songs/happy.csv")
        elif emotion == "sad":
            return pd.read_csv("songs/sad.csv")
        elif emotion == "angry":
            return pd.read_csv("songs/angry.csv")
        else:
            print("⚠️ Emotion not supported. Showing default songs.")
            return pd.read_csv("songs/happy.csv")
    except FileNotFoundError:
        print("❌ CSV file not found. Check songs folder.")
        return None


def display_songs(df):
    print("\n🎵 Recommended Songs:\n")
    for i, row in df.iterrows():
        print(f"{i+1}. {row['song_name']} - {row['artist']}")


def play_song(df):
    try:
        choice = int(input("\nEnter song number to play (0 to exit): "))

        if choice == 0:
            print("👋 Exiting without playing.")
            return

        if 1 <= choice <= len(df):
            url = df.iloc[choice - 1]['link']
            print("▶️ Opening song...")
            webbrowser.open(url)
        else:
            print("⚠️ Invalid choice.")
    except ValueError:
        print("⚠️ Please enter a valid number.")


def main():
    print("😊 Emotion-Based Music Recommendation System")
    print("📷 Look at the camera. Press 'q' to capture emotion.\n")

    # Step 1: Detect Emotion
    emotion = detect_emotion()
    print(f"\n✅ Detected Emotion: {emotion}")

    # Step 2: Load Songs
    df = load_songs(emotion)

    if df is None or df.empty:
        print("❌ No songs available.")
        return

    # Step 3: Display Songs
    display_songs(df)

    # Step 4: Play Song
    play_song(df)


if __name__ == "__main__":
    main()
