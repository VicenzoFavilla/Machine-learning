from django.urls import path
from vistaprevia import views

ulrpatterns = [
    path('', views.index, name='index')
]