# myproject/celery.py

from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# 设置Django的默认设置模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'OGX_server.settings')

app = Celery('OGX_server')

# 使用字符串来配置Celery，这样它就能在使用Windows时避免死锁问题。
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自动从所有已注册的Django应用中加载任务。
app.autodiscover_tasks()
