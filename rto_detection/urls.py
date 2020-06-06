from django.urls import path
from .views import RealTime, RealTimeScreen

app_name = 'rto_detection'

urlpatterns = [
    path('',  RealTimeScreen.as_view()),
    path('stream/',RealTime.as_view(), name='stream')  
]
