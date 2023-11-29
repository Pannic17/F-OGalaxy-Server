from __future__ import absolute_import, unicode_literals

import os.path
import time

import cv2

# Create your tests here.
# myapp/tasks.py

from celery import shared_task

from myapp.sd_pipeline import StableDiffusionGenerate

ABSOLUTE_PATH = os.path.abspath(__file__)  # Absolute directory path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root directory


@shared_task
def my_async_task():
    # 执行长时间运行的任务
    result = BASE_DIR
    return result


@shared_task
def run_pipeline(num):
    try:
        print(BASE_DIR)
        print("START")
        pipeline = StableDiffusionGenerate.from_default()
        print("INIT")

        # TEST
        image = cv2.imread("myapp/test/ngc7635.jpg")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("READ IMAGE")
        pipeline.photo = image
        w, h, c = image.shape
        points = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
        print("SET POINTS")

        pipeline.cv_process_photo(points)
        pipeline.set_status(901, "process photo<>calculate centers")
        print(pipeline.message)

        centers = pipeline.cv_calculate_centers(num)
        pipeline.set_status(902, "calculate centers<>generate")
        print(pipeline.message)

        generated_path = pipeline.generate()
        pipeline.set_status(903, "generate<>inpaint")
        print(pipeline.message)

        inpainted_path = pipeline.inpaint()
        pipeline.set_status(904, "inpaint<>upscale")
        print(pipeline.message)

        upscale_path = pipeline.upscale()
        pipeline.set_status(905, "upscale<>save hdr")
        print(pipeline.message)

        hdri_path, hdra_path = pipeline.save_hdr()
        pipeline.set_status(906, "save hdr<>calculate colors")
        print(pipeline.message)

        colors = pipeline.cv_cluster_colors(5)
        pipeline.set_status(999, ">finish")
        print(pipeline.message)

        print("任务完成")
        return {
            "generated_path": BASE_DIR+'\\'+generated_path,
            "inpainted_path": BASE_DIR+'\\'+inpainted_path,
            "upscale_path": BASE_DIR+'\\'+upscale_path,
            "hdri_path": BASE_DIR+'\\'+hdri_path,
            "hdra_path": BASE_DIR+'\\'+hdra_path,
            "colors": colors,
            "centers": centers,
            "meta": {
                "status": pipeline.status,
                "message": pipeline.message,
                "num": num
            }
        }
    except Exception as e:
        print(f"任务出错：{str(e)}")
        return f"任务出错：{str(e)}"
