import os


import django


# Recuperation et structuration des textes
from os import listdir
from os.path import isfile, join
import json
import re

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'songs_parody.settings')
django.setup()
from front.models import Song


def load_data(DATASET_FILEPATH):
    filenames = [f for f in listdir(DATASET_FILEPATH) if isfile(join(DATASET_FILEPATH, f))]
    lyrics = dict()

    for filename in filenames:
        if filename != "Beyonce.json":
            try:
                songs = dict()
                file = open(DATASET_FILEPATH + filename, encoding='utf-8')

                data = json.load(file)
                for i in data['songs']:
                    title = i['title'].strip(u'\u200b')
                    songs[title] = re.sub(r"[\[].*?[\]]", "", i["lyrics"])  # remove verses]
                lyrics[filename.split(".")[0]] = songs  # dictionnaires de paroles avec pour clé le nom de l'artiste pour valeur un dictionnaire qui a pour clé le titre de la chanson et pour valeur les paroles
            except:
                pass
                # print("error")
    return lyrics


def save_songs(songs):
    for author in songs:
        for song in songs[author]:
            title = song
            lyrics = songs[author][title]
            res = Song.objects.get_or_create(name=title, lyrics=lyrics)


if __name__ == "__main__":
    res = load_data("songs/")
    save_songs(res)
    # print(res["EdSheeran"].keys())
