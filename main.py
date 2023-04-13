from tkinter import *
from Model import *
from tkinter import ttk
from tkinter import messagebox

window = Tk()
window.geometry('700x500')
window.title('Penguins')
label = Label(window,
              text = "Select 1st Feature :  ",
              font = ("Times New Roman", 10))
label.grid(row=0,column=0)

# Combobox creation
feature1 = StringVar()
choosen1 = ttk.Combobox(window, width=27, textvariable=feature1)

# Adding combobox drop down list
choosen1['values'] = ('bill_length_mm',
                          'bill_depth_mm',
                          'flipper_length_mm',
                          'gender',
                          'body_mass_g',
                          )

choosen1.grid( row=1,column=0)
choosen1.current()

lbl = Label(window,
              text = "Select 2nd Feature:  ",
              font = ("Times New Roman", 10))
lbl.grid(row=2,column=0)

feature2 = StringVar()
choosen2 = ttk.Combobox(window, width=27, textvariable=feature2)

# Adding combobox drop down list
choosen2['values'] = ('bill_length_mm',
                          'bill_depth_mm',
                          'flipper_length_mm',
                          'gender',
                          'body_mass_g',
                          )

choosen2.grid(row=3,column=0)
choosen2.current()

#select class
label1 = Label(window,
              text = "Select Only Two Classes:  ",
              font = ("Times New Roman", 10))
label1.grid(row=4,column=0)
listbox1 = Listbox(window, width=20, height=5, selectmode=MULTIPLE,bg="white",fg="blue")
listbox1.grid(row=5,column=0)
#listbox.pack(side=LEFT, anchor=NONE, ipadx=10, ipady=10)
# Inserting the listbox items
listbox1.insert(1, "C1")
listbox1.insert(2, "C2")
listbox1.insert(3, "C3")
selected_classes=[]
def selected_class():
    counter=0
    for i in listbox1.curselection():
        counter+=1
        if (len(selected_classes)<2):
            selected_classes.append(listbox1.get(i))

    if(counter==3 or counter==1):
        msg= messagebox.showinfo("Error","Please Select 2 Classes Only")
        selected_classes.clear()
btn1 = Button(window, text='Select Classes', command=selected_class)

btn1.grid(row=5,column=1)


#learning rate

label2 = Label(window,
              text = "Enter learning rate :  ",
              font = ("Times New Roman", 10))
label2.grid(row=6,column=0)
learningRate=Entry(window)
learningRate.grid(row=7,column=0)
learningRate.focus_set()

#number of epochs

label3 = Label(window,
              text = "Enter Number Of Epochs :  ",
              font = ("Times New Roman", 10))
label3.grid(row=8,column=0)
NumberOfEpoces=Entry(window)
NumberOfEpoces.grid(row=9,column=0)
NumberOfEpoces.focus_set()

    # cheakbox
Bias = IntVar()
Checkbutton(window, text="Bias ", variable=Bias, onvalue=1, offvalue=0).grid(row=12, column=0)


btn3= Button(window, text='Train', command=lambda: [train()])
btn3.grid(row=9,column=1)


l1 = Label(window,
              text = "Feature 1 Value:  ",
              font = ("Times New Roman", 10))
l1.grid(row=1,column=2)
textbox3=Entry(window)
textbox3.grid(row=1,column=3)
textbox3.focus_set()
l2 = Label(window,
              text = "Feature 2 Value:  ",
              font = ("Times New Roman", 10))
l2.grid(row=3,column=2)

textbox4=Entry(window)
textbox4.grid(row=3,column=3)
textbox4.focus_set()
#mse
l4 = Label(window,
              text = "MSE threshold:  ",
              font = ("Times New Roman", 10))
l4.grid(row=10,column=0)

MSE=Entry(window)
MSE.grid(row=11,column=0)
MSE.focus_set()


l3 = Label(window,
              text = "            ",
              font = ("Times New Roman", 10))
l3.grid(row=2,column=4)

def predict():
    x=textbox3.get()
    y=textbox4.get()
    if x==''or y=='':
        msg = messagebox.showinfo("Error", "Enter The Missing Information ")
    else:
        if(Feature1!="gender"):
            x=float(textbox3.get())
        if (Feature2 != "gender"):
            y = float(textbox4.get())
        sampleData = [[x, y]]#Take From Gui
        sampleDF = pd.DataFrame(sampleData, columns=[Feature1, Feature2])
        sampleDF = samplePreprocessing(sampleDF, minMaxDF)
        p=predictSample(sampleDF, weights, Feature1, Feature2)
        if(p==1):
            p=selected_classes[0]
        else:
            p=selected_classes[1]
        dict = {'C1':'Adelie', 'C2': 'Gentoo', 'C3': 'Chinstrap'}
        messagebox.showinfo("Prediction", "The predicted class for this sample is: "+dict[p])



btn4= Button(window, text='Predict', command=predict)
btn4.grid(row=2,column=5)


def train():
    global minMaxDF, Feature1, Feature2, weights
    dict = {'C1': 0, 'C2': 1, 'C3': 2}
    if feature1.get()=='' or feature2.get()=='':
        messagebox.showinfo("Error", "Select Two Features")
    elif feature1.get()==feature2.get() :
        messagebox.showinfo("Error", "Select Two Different Features")
    else:

        if NumberOfEpoces.get()==''or MSE.get()=='' or learningRate.get()==''or len(selected_classes)==0 :
            msg = messagebox.showinfo("Error", "Enter  Missing Information ")
        else:
         minMaxDF, Feature1, Feature2, weights = main(feature1.get(), feature2.get(), dict[selected_classes[0]],dict[selected_classes[1]],
                                                      int(NumberOfEpoces.get()), float(learningRate.get()), Bias.get(),float(MSE.get()))

window.mainloop()

# sampleData = [[13.2, 4500]]#Take From Gui
# sampleDF = pd.DataFrame(sampleData, columns=[feature1, feature2])
# sampleDF = samplePreprocessing(sampleDF, minMaxDF)
# # print(sampleDF[feature1][0])
# predictSample(sampleDF, weights, feature1, feature2)