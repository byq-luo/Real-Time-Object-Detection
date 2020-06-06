from django.shortcuts import render
from django.views.generic import View
from django.http import StreamingHttpResponse
from .sort import simple_sort
from .video_stream import stream_video
# Create your views here.



class RealTime(View):
    def get(self, request): 
        try :    
            return StreamingHttpResponse(streaming_content=stream_video(),content_type="multipart/x-mixed-replace;boundary=frame")
        except :
            return "error"
        
class RealTimeScreen(View):
    def get(self, request):
        return render(request, 'rto_detection/index.html') 
   
   
 