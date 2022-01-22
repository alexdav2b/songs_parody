"""songs_parody URL Configuration

The `urlpatterns` list routes URLs to front. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function front
    1. Add an import:  from my_app import front
    2. Add a URL to urlpatterns:  path('', front.home, name='home')
Class-based front
    1. Add an import:  from other_app.front import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('front/', include('front.urls')),
    path('admin/', admin.site.urls),
]
