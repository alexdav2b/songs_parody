# Preparation
- Download models in "songs_parody/front/core" : https://drive.google.com/file/d/1RD2O-f29m6KEA4MkVIc1HGkRSZAzHqxK/view?usp=sharing
- Download songs in "songs_parody/front/core/songs" : https://drive.google.com/file/d/1Z3r-3hIQ7iHr9C1YKyANDGkuUtKHos_h/view?usp=sharing

# Re-train models (optional)
- Download dataset in "songs_parody/front/core/emotion-detection-from-text" : https://drive.google.com/file/d/1FCR_Gpzsidnmq8iOVmsriISu7tyeleH9/view?usp=sharing
- Run "songs_parody/front/core/generate_dataset.py"
- Run "songs_parody/front/core/main.py"

# Run
- Run "songs_parody/front/core/generate_song_database"
- in console exec :
  - python manage.py makemigrations front
  - python manage.py migrate
- Run django project
- go to : http://127.0.0.1:8000/front/