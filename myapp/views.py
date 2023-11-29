import os

from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from .tasks import my_async_task, run_pipeline
from celery.result import AsyncResult

ABSOLUTE_PATH = os.path.abspath(__file__)  # Absolute directory path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root directory


def handle_ue_request(request):
    # 这里可以根据需要处理请求
    my_async_task.delay()
    data = {"message": "Hello from Django!"}
    return JsonResponse(data)


# myapp/views.py

# myapp/views.py

def start_task(request):
    num = request.GET.get('num')
    if num and num.isdigit():
        task = run_pipeline.delay(int(num))
        return JsonResponse({"task_id": task.id, 'code': 201, 'num': num, 'path': BASE_DIR})
    else:
        return JsonResponse({"error": "No valid number provided."})


# myapp/views.py (继续)


def check_task(request, task_id):
    task_result = AsyncResult(task_id)
    if task_result.state == 'PENDING':
        response_data = {'state': task_result.state,
                         'code': 202,
                         'message': "Task pending. Please wait a moment."}
    elif task_result.state != 'FAILURE':
        response_data = {'state': task_result.state,
                         'code': 200,
                         'result': task_result.get()}
    else:
        # 在这里处理失败的情况
        # code = task_result.get("status")
        response_data = {'state': task_result.state,
                         'code': 520,  # Errors
                         'message': "Check server log to see detail error.",
                         'error': str(task_result.result),
                         'info': str(task_result.info)}
    return JsonResponse(response_data)
