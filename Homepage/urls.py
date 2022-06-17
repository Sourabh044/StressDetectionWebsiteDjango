from django.urls import path,include

from . import views

urlpatterns = [
    path('', views.Homepage, name='Stress Detector'),
    path('AboutUS', views.Aboutus, name='Stress Detector'),

]