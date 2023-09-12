from django.db import models


    
# class FishGroup(models.Model):
#     name = models.CharField(max_length=100)

class Image(models.Model):
    name = models.CharField(max_length=100) # actual name
    file_path = models.CharField(max_length=300) # id
    data_path = models.CharField(max_length=300) # year
    thumbnail_path = models.CharField(max_length=300) # vessel
    fish_group = models.CharField(max_length=100) # quarter
    year_reading_manual = models.IntegerField(default=-1)
    year_reading_automated = models.IntegerField(default=-1)  

# class Age(models.Model):
#     year = models.IntegerField()
#     method = models.CharField(max_length=100)
#     image = models.ForeignKey(Image, on_delete=models.CASCADE)


class OtolithImage(models.Model):
    name = models.CharField(max_length=100) # actua
    file_path = models.CharField(max_length=300)
    data_path = models.CharField(max_length=300)
    thumbnail_path = models.CharField(max_length=300)
    fish_group = models.CharField(max_length=100)
    year_reading_manual = models.IntegerField(default=-1)
    year_reading_automated = models.IntegerField(default=-1)  



