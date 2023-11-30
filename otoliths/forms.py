from django import forms


class UploadForm(forms.Form):
    folder = forms.CharField(widget=forms.TextInput(attrs={"class": "form-control"}))
    species = forms.CharField(widget=forms.TextInput(attrs={"class": "form-control"}))
    quarter = forms.CharField(widget=forms.TextInput(attrs={"class": "form-control"}))
    location = forms.CharField(widget=forms.TextInput(attrs={"class": "form-control"}))
    # image = forms.FileField()