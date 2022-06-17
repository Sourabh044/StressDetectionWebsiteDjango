# from django.http import HttpResponse
from django.shortcuts import render
from .forms import Myform
from .Algos import naive,knn,logistict,rf,decisionTree,svm
# from django.contrib import messages
# def index(request):
#     return HttpResponse("Hello, world. Run Successfully")
def Homepage(request):
    form = Myform()
    if request.method=='POST':
        form = Myform(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            inputext = form.data.get('dataText')
            algo = form.data.get('algo')
            if algo == 'naive':
                result = naive(inputext)
                form = Myform()
                context = {'form':form ,'result': type(naive(inputext))}
                return render(request,'result.html',{'form':form ,'result': result, 'algo': algo})
                # if result == 'Stress':
                #     messages.info(request,'Stress')
                #     form = Myform()
                #     context = {'form': form,'result': 'Stress'}
                #     return render(request,'result.html',context)
                # else:
                #     messages.info(request,'No Stress')
                #     form = Myform()
                #     context = {'form': form,'result': 'No Stress' }
                #     return render(request,'result.html',context)
                
            if algo == 'logistic':
                result = logistict(inputext)
                context = {'form':form ,'result': logistict(inputext)}
                return render(request,'result.html',{'form':form ,'result': result, 'algo': algo})
            if algo == 'decision':
                result = decisionTree(inputext)
                context = {'form':form ,'result': decisionTree(inputext)}
                return render(request,'result.html',{'form':form ,'result': result, 'algo': algo})
            if algo == 'knn':
                result = knn(inputext)
                context = {'form':form ,'result': knn(inputext)}
                return render(request,'result.html',{'form':form ,'result': result, 'algo': algo})
            if algo == 'svm':
                result = svm(inputext)
                context = {'form':form ,'result': svm(inputext)}
                return render(request,'result.html',{'form':form ,'result': result, 'algo': algo})
            if algo == 'Rf':
                result = rf(inputext)
                context = {'form':form ,'result': rf(inputext)}
                return render(request,'result.html',{'form':form ,'result': result, 'algo': algo})
        else:
            form = Myform()

    context = {'form': form, }
    return render(request,'index.html',context)



def Aboutus(request):
    return render(request,'aboutus.html')

