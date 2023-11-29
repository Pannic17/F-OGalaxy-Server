# myapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('ue_request/', views.handle_ue_request, name='ue_request'),
    path('start-task/', views.start_task, name='start_task'),
    path('check-task/<task_id>/', views.check_task, name='check_task'),
]
