"""
URL configuration for random_face_generator project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.homepage, name="home"),
    path('eye', views.eye, name="nose"),
    path('nose', views.nose, name="nose"),
    path('mouth', views.mouth, name="mouth"),
    path('services', views.services, name="service"),
    path('about', views.about, name="about"),
    path('contact', views.contact, name="contact"),
    path('services/form1', views.select_eye, name="select_eye"),
    path('services/form2', views.select_nose, name="select_nose"),
    path('services/form3', views.select_mouth, name="select_mouth"),
    path('services/form4', views.final, name="final"),



]
