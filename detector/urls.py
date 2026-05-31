from django.urls import path
from . import views

urlpatterns = [
    path('', views.single_detection, name='single_detection'),
    path('multi/', views.multi_detection, name='multi_detection'),
]

