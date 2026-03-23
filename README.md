# Smart-Playlist-Matcher-
Project Overview
This project implements a 'Smart Playlist Matcher' that allows users to upload an audio clip and find similar tracks from a synthetic catalog based on mood and tempo. It uses machine learning for mood prediction and feature extraction to provide personalized music recommendations.

How to Run
Open the Colab Notebook: Ensure you are in the Google Colab environment with this notebook open.
Run All Cells: Go to Runtime -> Run all in the Colab menu. This will:
Generate a synthetic audio catalog.
Extract audio features and train a mood classification model.
Initialize an SQLite database for logging queries.
Launch the Gradio user interface.
Interact with the Gradio App: Once the last cell finishes executing, a Gradio interface will appear directly in the notebook output (or as a public share link). You can:
Upload an audio file (e.g., an MP3 or WAV clip).
Adjust the 'Weight: Tempo <---> Mood' slider to prioritize matching by tempo or mood.
Click 'Find Matches' to get recommendations.
The app will display the detected BPM, predicted mood, a spectrogram of your uploaded audio, and up to 5 recommended tracks from the catalog.
How to Add More Catalog Tracks
Option 1: Manually adding a new synthetic track (like the initial generation)

To add another synthetic track similar to how the initial catalog was created, you can modify or re-run the relevant code. For example, to add another 'happy' track
import numpy as np
import soundfile as sf
import pandas as pd

# --- Example of adding a new synthetic track --- #
new_track_idx = len(df_catalog) # Get a new unique index
new_mood = 'happy'
sr = 22050
duration = 5
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
y_new = np.sin(2 * np.pi * 440 * t) * np.exp(-t) # Example for 'happy'

new_file_path_synth = f"catalog/track_{new_track_idx}_{new_mood}.wav"
sf.write(new_file_path_synth, y_new, sr)

# Extract features for this new track
feats_synth = extract_features(new_file_path_synth)

if feats_synth is not None:
    feats_synth_scaled = scaler.transform([feats_synth])
    new_track_data_synth = {
        'track_id': f'track_{new_track_idx}',
        'file_path': new_file_path_synth,
        'true_mood': new_mood,
        'true_bpm': np.random.randint(100, 140), # Assign a realistic BPM for happy
        'features': feats_synth_scaled[0],
        'extracted_bpm': feats_synth[-2]
    }
    global df_catalog # Ensure we modify the global DataFrame
    df_catalog = pd.concat([df_catalog, pd.DataFrame([new_track_data_synth])], ignore_index=True)
    print(f"Added new synthetic track 'track_{new_track_idx}' to the catalog.")
else:
    print(f"Failed to extract features for synthetic track 'track_{new_track_idx}'.")

display(df_catalog.tail())

Option 2: Adding a track from an uploaded audio file (simulating Gradio upload)

If you have an audio file you want to add (e.g., downloaded, or one that was previously uploaded to Gradio), you can use the add_track_from_gradio_input function defined in a previous cell. This function will copy your file into the catalog/ directory, extract its features, predict its mood and BPM, and add it to df_catalog.

# Example of using the function to add a track from a local file
# Replace 'path/to/your/audio.wav' with the actual path to your audio file
# add_track_from_gradio_input('path/to/your/audio.wav')

# Or, to re-add a track already in the catalog as a *new* entry (for testing):
# add_track_from_gradio_input(df_catalog.iloc[5]['file_path']) # Adds a copy of an existing track as a new entry
After adding tracks using either method, they will be available for matching in the Gradio interface.

Limitations
This project serves as a conceptual demonstration and has several limitations. The audio catalog is entirely synthetic, meaning it doesn't represent the complexity or diversity of real-world music. The mood classification relies on a simple Logistic Regression model and basic audio features (MFCCs, tempo, energy), which may not accurately capture nuanced emotional content. The 'true_mood' and 'true_bpm' values in the catalog are assigned during synthetic creation rather than being accurately analyzed, potentially limiting the ground truth for matching. Lastly, the small size of the catalog and the synthetic nature of the data mean the recommendations are illustrative rather than robust for a large-scale, real-world application.
