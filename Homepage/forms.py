from dataclasses import field
from django import forms
from .models import Homepage
# class MyForm(forms.Form): #Note that it is not inheriting from forms.ModelForm
#     # a = forms.CharField(max_length=20)
#     message = forms.CharField(widget=forms.Textarea)

#     Algo = forms.CharField(max_length=6,choices=Algo_CHOICES, default='green')

class Myform(forms.ModelForm):
    class Meta:
        model = Homepage
        fields = ('__all__')