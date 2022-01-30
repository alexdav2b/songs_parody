from django.http import JsonResponse
from django.shortcuts import render
from .models import Theme, Song
from .core.main import do_parody


def index(request):
    return None


def main_page(request):
    themes_list = Theme.objects.all()
    songs_list = Song.objects.all()
    context = {
        'themes': themes_list,
        'songs': songs_list,
    }
    return render(request, 'front/main_page.html', context)


def get_lyrics(request):
    if request.method == "GET":
        song_id = request.GET.get("song_id", None)
        if not song_id:
            return JsonResponse("need the id of the song to get lyrics", status=400)
        song = Song.objects.get(pk=song_id)
        lyrics = song.get_lyrics_by_line()
        return render(request, 'front/lyrics.html', {"lyrics": lyrics})

    return JsonResponse({}, status=400)


def get_parody(request):
    if request.method == "GET":
        song_id = request.GET.get("song_id", None)
        if not song_id:
            return JsonResponse("need the id of the song to get lyrics", status=400)

        theme_id = request.GET.get("theme_id", None)
        if not theme_id:
            return JsonResponse("need the id of the song to get lyrics", status=400)

        song = Song.objects.get(pk=song_id)
        lyrics = song.lyrics
        print(lyrics)
        theme = Theme.objects.get(pk=theme_id)
        theme = theme.name.lower()
        parody_lyrics = do_parody(lyrics, f"front/core/{theme}")
        print(parody_lyrics)
        return render(request, 'front/lyrics.html', {"lyrics": parody_lyrics.split('\n')})

    return JsonResponse({}, status=400)
