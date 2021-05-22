from django.apps import AppConfig

from apps.nn.NeuralNetwork import NeuralNetwork


class NnConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.nn'
    nn = NeuralNetwork()

    def ready(self):
        NnConfig.nn.load_model()
