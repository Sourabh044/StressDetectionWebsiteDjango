from django.db import models
Algo_CHOICES = [
    ('naive','NAIVE'),
    ('logistic', 'LOGISTIC'),
    ('decision','DECISION'),
    ('knn','KNN'),
    ('svm','SVM'),
    ('Rf','RandomForest'),
]
# Create your models here.
class Homepage(models.Model):
    dataText = models.CharField(max_length=300, null=False)
    algo = models.CharField(max_length=200, choices=Algo_CHOICES , default=" ", blank=False)