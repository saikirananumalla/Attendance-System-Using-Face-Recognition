from django.contrib import admin
from django.urls import include, path
from . import views

urlpatterns = [
    path('', views.home),
    path('train', views.trainImage),
    path('test', views.testImage),
    path('adminT', views.AdminTrain),
    path('loadcsv',views.loadcsv),
]
