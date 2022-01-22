from django.db import models


class Theme(models.Model):
    name = models.CharField(max_length=70)

    def __str__(self):
        return self.name


class Song(models.Model):
    name = models.CharField(max_length=70)
    theme = models.ForeignKey(Theme, on_delete=models.CASCADE)
    lyrics = models.TextField()

    def get_lyrics_by_line(self):
        return self.lyrics.split('\n')

    def __str__(self):
        return self.name
