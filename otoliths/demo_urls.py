

# Create your tests here.
from django.contrib import admin
from django.urls import path
from django.contrib.auth.views import LogoutView, LoginView

from . import demo_views as views

urlpatterns = [
    path('index/', views.images),
    path('images/', views.images),
    path('detail/<int:image_id>', views.detail),
    path('map/', views.map),
    path('analysis/', views.analysis),
    path('researchers/', views.researchers),
    path('ai/', views.ai),
    path('upload/', views.upload),
    path('run/', views.run),
    path('unet_setup/', views.unet_setup),
    path('mrcnn_setup/', views.mrcnn_setup),
    path('start_run/', views.start_run),
    path('results/', views.show_results_table_from_file),

    #---------------------------------------

    path('dataview/', views.dataview),


    path('show_data/', views.show_data),
    path('show_runs/', views.show_runs),

    path('experiments/textbased/', views.experiments_textbased),

    path('experiments/transfer/', views.experiments_transfer_learning),
    path('experiments/ensemble/', views.experiments_ensemble_learning),
    path('experiments/continual', views.experiments_continual_learning),

    
    #----------------------------------------

    path('visual/', views.visualize),
    path('northsea/', views.northsea),
    # path('balticsea/', views.balticsea),
    path('balticsea_sets/', views.balticsea_sets),
    path('balticsea_sets/<str:folder>/', views.balticsea_images),
    path('detail/<str:image_name>', views.data_detail),
    path('experiments/', views.experiments),
    path('experiments/run_unet/', views.experiments_unet),
    path('experiments/run_mrcnn/', views.experiments_mrcnn),
    
    path('interact/', views.interact),
    path('ensemble/<str:dataset>/<str:subset>/', views.ensemble_sets),
    path('dataview/<str:dataset>/', views.dataview_sets),
    path('dataview/<str:dataset>/<str:folder>/', views.dataview_images),
    path('dataview/<str:dataset>/<str:folder>/<str:image_name>', views.data_detail),
    
    path('logout/', LogoutView.as_view(), name="logout"),
    path('', LoginView.as_view(template_name='accounts/login.html', redirect_authenticated_user=True), name="login"),

    #-----------------

]
