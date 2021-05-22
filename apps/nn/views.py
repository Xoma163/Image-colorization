# Create your views here.
import base64
import io

import numpy as np
from PIL import Image
from django.shortcuts import render
from django.views import View

from apps.nn.ImageHandler import DatasetImage
from apps.nn.apps import NnConfig
from apps.nn.consts import IMAGE_SIZE


class PredictImageTemplateView(View):
    template_name = 'index.html'

    def get(self, request, *args, **kwargs):
        input_file = request.FILES
        return render(request, self.template_name)

    @staticmethod
    def image_to_base64(image) -> str:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_base64 = base64.b64encode(image_bytes.getvalue()).decode()
        return f"data:image/png;base64, {image_base64}"

    @staticmethod
    def predict_image(image_file) -> Image:
        d_image = DatasetImage(np.array(image_file))
        NnConfig.nn.predict(d_image)
        predicted_image_lab_array = d_image.get_predicted_image()
        predicted_image_rgb_array = d_image.lab2rgb(predicted_image_lab_array)
        predicted_image_array = predicted_image_rgb_array.astype(np.uint8)
        predicted_image = Image.fromarray(predicted_image_array)

        return predicted_image

    def post(self, request, *args, **kwargs):
        input_file = request.FILES['input_image'].file
        img = Image.open(input_file).convert("RGB")
        # ToDo:
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeWarning()

        context_data = {
            'image_colored': self.image_to_base64(self.predict_image(img)),
            'image_black': self.image_to_base64(img),
        }
        return render(request, self.template_name, context_data)

    # def get_context_data(self, **kwargs):
    #     context = super().get_context_data(**kwargs)
    #     context['image_black'] = None
    #     context['image_colored'] = None
    #     return context
