from django import forms
from .models import Image

class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ('name',)
class UrlForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ('name',)
