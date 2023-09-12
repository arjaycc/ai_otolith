

# Create your tests here.
from django.contrib import admin
from django.urls import path
from django.contrib.auth.views import LogoutView

from . import views

urlpatterns = [
    path('', views.index),
    path('images/', views.images),
    path('detail/<int:image_id>', views.detail),
    path('map/', views.map),
    path('analysis/', views.analysis),
    path('researchers/', views.researchers),
    path('ai/', views.ai),
    path('upload/', views.upload),
    path('visual/', views.visualize),
    path('northsea/', views.northsea),
    path('balticsea/', views.balticsea),
    path('detail/<str:image_name>', views.data_detail),
    path('experiments/', views.experiments),
    path('experiments/run_unet/', views.experiments_unet),
    path('logout/', LogoutView.as_view(), name="logout")
]
