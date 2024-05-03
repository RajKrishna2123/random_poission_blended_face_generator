from django.core.files.storage import default_storage
from django.conf import settings
from . import urls
from django.shortcuts import render
from django.http import HttpResponse
import requests

def homepage(request):
    return render (request, "index.html")