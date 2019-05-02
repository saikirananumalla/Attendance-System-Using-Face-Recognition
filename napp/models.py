from django.db import models

# Create your models here.
class Image(models.Model):
    name = models.CharField(max_length=500,)
    #url = models.ImageField(upload_to='static/images/')
    def __str__(self):
        if self.name:
            return self.name
        else:
            return str(self.pk)
