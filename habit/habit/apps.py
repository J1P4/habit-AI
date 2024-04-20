from django.apps import AppConfig

class HabitConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'habit'
    def ready(self):
        from habit.food_recommendation_v1 import food_recommendation
        food_recommendation()