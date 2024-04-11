from django import forms


class UploadForm(forms.Form):
    files = forms.FileField(required=False, widget=forms.ClearableFileInput(attrs={"class": "form-control", "multiple": True})) 
    folder = forms.CharField(required=True, widget=forms.TextInput(attrs={"class": "form-control"}))
    data_id = forms.CharField(initial="0", required=True, widget=forms.TextInput(attrs={"class": "form-control"}))
    # quarter = forms.CharField(required=False, widget=forms.TextInput(attrs={"class": "form-control"}))
    # location = forms.CharField(required=False, widget=forms.TextInput(attrs={"class": "form-control"}))
    ring_groundtruth = forms.FileField(required=False, widget=forms.ClearableFileInput(attrs={"class": "form-control", "multiple": True})) 
    outer_contour = forms.FileField(required=False, widget=forms.ClearableFileInput(attrs={"class": "form-control", "multiple": True})) 

    # image = forms.FileField()
    def __init__(self, *args, **kwargs):
        super(UploadForm, self).__init__(*args, **kwargs)

class DataFilterForm(forms.Form):
    subset = forms.ChoiceField(choices=[])

    def __init__(self, *args, **kwargs):
        request = kwargs.pop('request')
        choices = kwargs.pop('choices')
        super(DataFilterForm, self).__init__(*args, **kwargs)
        self.fields['subset'].widget.attrs.update({'class': 'form-control'})
        self.fields['subset'].choices = [(v, v) for x,v in enumerate(choices)]


class EnsembleFilterForm(forms.Form):
    subset = forms.ChoiceField(choices=[])

    def __init__(self, *args, **kwargs):
        request = kwargs.pop('request')
        choices = kwargs.pop('choices')
        super(EnsembleFilterForm, self).__init__(*args, **kwargs)
        self.fields['subset'].widget.attrs.update({'class': 'form-control'})
        self.fields['subset'].choices = [(v, v) for x,v in enumerate(choices)]


class UnetForm(forms.Form):
    unet_model = forms.ChoiceField(choices=[], widget=forms.Select(attrs={"class": "form-control"}))

    def __init__(self, *args, **kwargs):
        choices = kwargs.pop('choices')
        super(UnetForm, self).__init__(*args, **kwargs)
        self.fields['unet_model'].widget.attrs.update({'class': 'form-control'})
        self.fields['unet_model'].choices = [(v, v) for x,v in enumerate(choices)]

class RunFormPhase1(forms.Form):
    method = forms.ChoiceField(choices=[])
    dataset = forms.ChoiceField(choices=[], widget=forms.Select(attrs={"class": "form-control"}))
    run_type = forms.ChoiceField(choices=[], widget=forms.Select(attrs={"class": "form-control"}))
    run_label = forms.CharField(initial="uniquename", widget=forms.TextInput(attrs={"class": "form-control"}))
    idr = forms.CharField(initial="0", widget=forms.TextInput(attrs={"class": "form-control"}))
    selected = forms.CharField(initial="37", widget=forms.TextInput(attrs={"class": "form-control"}))
    split_name = forms.CharField(initial="ex0kfold", widget=forms.TextInput(attrs={"class": "form-control"}))

    age_limit = forms.CharField(required=False, widget=forms.TextInput(attrs={"class": "form-control"}))
    target_species = forms.CharField(required=False, widget=forms.TextInput(attrs={"class": "form-control"}))
    brighten = forms.CharField(required=False, widget=forms.TextInput(attrs={"class": "form-control"}))
    source_dataset = forms.CharField(required=False, widget=forms.TextInput(attrs={"class": "form-control"}))

    def __init__(self, *args, **kwargs):
        # request = kwargs.pop('request')
        # choices = kwargs.pop('choices')
        super(RunFormPhase1, self).__init__(*args, **kwargs)
        self.fields['method'].widget.attrs.update({'class': 'form-control'})
        self.fields['method'].choices = [('U-Net', 'U-Net'), ('Mask R-CNN', 'Mask R-CNN')]
        self.fields['dataset'].choices = [('datasets_north', 'datasets_north'), ('datasets_baltic', 'datasets_baltic'), ('datasets_user', 'datasets_user')]
        self.fields['run_type'].choices = [('test', 'test'), ('train', 'train'), ('both', 'both')]

class RunFormPhase2(forms.Form):
    method = forms.ChoiceField(choices=[])
    dataset = forms.ChoiceField(choices=[], widget=forms.Select(attrs={"class": "form-control"}))
    run_type = forms.ChoiceField(choices=[], widget=forms.Select(attrs={"class": "form-control"}))
    run_label = forms.CharField(initial="uniquename", widget=forms.TextInput(attrs={"class": "form-control"}))
    idr = forms.CharField(initial="0", widget=forms.TextInput(attrs={"class": "form-control"}))
    selected = forms.CharField(initial="37", widget=forms.TextInput(attrs={"class": "form-control"}))
    split_name = forms.CharField(initial="ex0kfold", widget=forms.TextInput(attrs={"class": "form-control"}))

    base = forms.CharField(widget=forms.TextInput(attrs={"class": "form-control"}))
    base_id = forms.CharField(widget=forms.TextInput(attrs={"class": "form-control"}))
    continual = forms.ChoiceField(choices=[(0, 'False'), (1, 'True')], widget=forms.Select(attrs={"class": "form-control"}))
    eval_balanced_set = forms.ChoiceField(choices=[(0, 'False'), (1, 'True')], widget=forms.Select(attrs={"class": "form-control"}))
    source_dataset = forms.CharField(required=False, widget=forms.TextInput(attrs={"class": "form-control"}))

    def __init__(self, *args, **kwargs):
        # request = kwargs.pop('request')
        # choices = kwargs.pop('choices')
        super(RunFormPhase2, self).__init__(*args, **kwargs)
        self.fields['method'].widget.attrs.update({'class': 'form-control'})
        self.fields['method'].choices = [('U-Net', 'U-Net'), ('Mask R-CNN', 'Mask R-CNN')]
        self.fields['dataset'].choices = [('datasets_north', 'datasets_north'), ('datasets_baltic', 'datasets_baltic'), ('datasets_user', 'datasets_user')]
        self.fields['run_type'].choices = [('test', 'test'), ('train', 'train'), ('both', 'both')]

